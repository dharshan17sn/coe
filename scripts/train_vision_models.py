"""
Training Pipeline for Vision & Multimodal Deep Learning Models.
Models:
1. Custom CNN (Slide 9 & 11)
2. Modified AlexNet (Slide 9 & 11)
3. Multimodal EfficientNet-B0 [Physical] (Slide 13 & 14)
4. Multimodal EfficientNet-B0 [No Pressure] (Slide 13 & 14)

Evaluates on:
- Validation Set (2020 - 2022)
- Test Holdout Set (2023 - 2026, unseen cyclones)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import CustomCycloneCNN, ModifiedAlexNet, MultimodalEfficientNet
from src.data import MultimodalCycloneDataset

MODELS_DIR = "models"
REPORTS_DIR = "reports"


def prepare_tabular_inputs(df, use_pressure=True, scaler=None, imputer_vals=None):
    df = df.copy()
    df["SUBBASIN_CODE"] = (df["SUBBASIN"] == "BB").astype(float)
    if "DIST2LAND" not in df.columns:
        df["DIST2LAND"] = 0.0

    if use_pressure:
        feature_cols = ["PRES_HPA", "LAT", "LON", "DIST2LAND", "SUBBASIN_CODE"]
    else:
        feature_cols = ["LAT", "LON", "DIST2LAND", "SUBBASIN_CODE"]

    if imputer_vals is None:
        imputer_vals = {col: df[col].median() for col in feature_cols}

    for col in feature_cols:
        df[col] = df[col].fillna(imputer_vals[col])

    X = df[feature_cols].values
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    y_wind = df["WIND_KTS"].values
    y_class = df["IMD_CLASS_CODE"].clip(0, 7).values
    return X_scaled, y_wind, y_class, scaler, imputer_vals, feature_cols


def train_vision_model(model_type, model, train_loader, val_loader, test_loader,
                       y_val_wind, y_test_wind, y_val_class, y_test_class,
                       wind_mean, wind_std, epochs=15, lr=1e-3, is_multimodal=False):
    print(f"\n{'='*65}")
    print(f"TRAINING VISION MODEL: {model_type} (Epochs: {epochs})")
    print(f"{'='*65}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    reg_criterion = nn.SmoothL1Loss()
    cls_criterion = nn.CrossEntropyLoss()

    best_val_rmse = float("inf")
    best_weights = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            img, tab, by_wind, by_cls = batch
            optimizer.zero_grad()

            if is_multimodal:
                p_wind, p_cls = model(img, tab)
            else:
                p_wind, p_cls = model(img)

            loss = reg_criterion(p_wind, by_wind) + 0.3 * cls_criterion(p_cls, by_cls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(img)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation evaluation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                img, tab, _, _ = batch
                if is_multimodal:
                    pw, _ = model(img, tab)
                else:
                    pw, _ = model(img)
                val_preds.extend((pw.squeeze(1).numpy() * wind_std + wind_mean).tolist())

        val_rmse = float(np.sqrt(mean_squared_error(y_val_wind, val_preds)))
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        print(f" Epoch [{epoch:2d}/{epochs:2d}] | Loss: {epoch_loss:.4f} | Val RMSE: {val_rmse:.2f} kt (Best: {best_val_rmse:.2f} kt @ ep {best_epoch})")

    # Load best weights & evaluate test
    model.load_state_dict(best_weights)
    model.eval()

    # Val Metrics
    val_preds, val_cls = [], []
    with torch.no_grad():
        for batch in val_loader:
            img, tab, _, _ = batch
            pw, pc = model(img, tab) if is_multimodal else model(img)
            val_preds.extend((pw.squeeze(1).numpy() * wind_std + wind_mean).tolist())
            val_cls.extend(torch.argmax(pc, dim=1).numpy().tolist())

    val_rmse_final = float(np.sqrt(mean_squared_error(y_val_wind, val_preds)))
    val_mae_final = float(mean_absolute_error(y_val_wind, val_preds))
    val_r2_final = float(r2_score(y_val_wind, val_preds))

    # Test Metrics
    test_preds, test_cls = [], []
    with torch.no_grad():
        for batch in test_loader:
            img, tab, _, _ = batch
            pw, pc = model(img, tab) if is_multimodal else model(img)
            test_preds.extend((pw.squeeze(1).numpy() * wind_std + wind_mean).tolist())
            test_cls.extend(torch.argmax(pc, dim=1).numpy().tolist())

    test_rmse = float(np.sqrt(mean_squared_error(y_test_wind, test_preds)))
    test_mae = float(mean_absolute_error(y_test_wind, test_preds))
    test_r2 = float(r2_score(y_test_wind, test_preds))
    test_bias = float(np.mean(np.array(test_preds) - y_test_wind))
    test_acc = float(accuracy_score(y_test_class, test_cls))

    # Save checkpoint
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, f"{model_type.lower().replace(' ', '_').replace('-', '_')}.pt")
    torch.save(best_weights, save_path)
    print(f" Saved best weights to {save_path}")

    return {
        "Model": model_type,
        "Epochs run": best_epoch,
        "Val RMSE": round(val_rmse_final, 3),
        "Val MAE": round(val_mae_final, 3),
        "Val R2": round(val_r2_final, 3),
        "Test 2025 RMSE": round(test_rmse, 3),
        "Test 2025 MAE": round(test_mae, 3),
        "Test 2025 R2": round(test_r2, 3),
        "Test 2025 Bias": round(test_bias, 3),
        "Test Acc": round(test_acc * 100, 1),
    }


def run_all_vision_experiments():
    train_df = pd.read_csv(os.path.join("datasets", "splits", "train.csv"))
    val_df = pd.read_csv(os.path.join("datasets", "splits", "validation.csv"))
    test_df = pd.read_csv(os.path.join("datasets", "splits", "test.csv"))

    # Sample a representative subset for efficient CPU training (1500 train samples)
    np.random.seed(42)
    train_subset = train_df.sample(n=min(1500, len(train_df)), random_state=42).reset_index(drop=True)
    val_subset = val_df.sample(n=min(300, len(val_df)), random_state=42).reset_index(drop=True)
    test_subset = test_df.sample(n=min(300, len(test_df)), random_state=42).reset_index(drop=True)

    # 1. Physical features (with pressure)
    X_train_p, y_train_w, y_train_c, scaler_p, imputer_p, f_p = prepare_tabular_inputs(train_subset, use_pressure=True)
    X_val_p, y_val_w, y_val_c, _, _, _ = prepare_tabular_inputs(val_subset, use_pressure=True, scaler=scaler_p, imputer_vals=imputer_p)
    X_test_p, y_test_w, y_test_c, _, _, _ = prepare_tabular_inputs(test_subset, use_pressure=True, scaler=scaler_p, imputer_vals=imputer_p)

    wind_mean = float(y_train_w.mean())
    wind_std = float(y_train_w.std())

    y_train_w_scaled = (y_train_w - wind_mean) / wind_std
    y_val_w_scaled = (y_val_w - wind_mean) / wind_std
    y_test_w_scaled = (y_test_w - wind_mean) / wind_std

    # Physical Datasets
    ds_train_p = MultimodalCycloneDataset(train_subset, X_train_p, y_train_w_scaled, y_train_c, is_train=True)
    ds_val_p = MultimodalCycloneDataset(val_subset, X_val_p, y_val_w_scaled, y_val_c, is_train=False)
    ds_test_p = MultimodalCycloneDataset(test_subset, X_test_p, y_test_w_scaled, y_test_c, is_train=False)

    loader_train_p = DataLoader(ds_train_p, batch_size=32, shuffle=True)
    loader_val_p = DataLoader(ds_val_p, batch_size=32, shuffle=False)
    loader_test_p = DataLoader(ds_test_p, batch_size=32, shuffle=False)

    results = []

    # Model 1: Custom CNN (Slide 9 & 11)
    cnn_model = CustomCycloneCNN(in_channels=3, num_classes=8)
    res_cnn = train_vision_model(
        "Custom CNN", cnn_model,
        loader_train_p, loader_val_p, loader_test_p,
        y_val_w, y_test_w, y_val_c, y_test_c,
        wind_mean, wind_std, epochs=10, lr=1e-3, is_multimodal=False
    )
    results.append(res_cnn)

    # Model 2: Modified AlexNet (Slide 9 & 11)
    alexnet_model = ModifiedAlexNet(in_channels=3, num_classes=8, pretrained=False)
    res_alexnet = train_vision_model(
        "Modified AlexNet", alexnet_model,
        loader_train_p, loader_val_p, loader_test_p,
        y_val_w, y_test_w, y_val_c, y_test_c,
        wind_mean, wind_std, epochs=10, lr=1e-3, is_multimodal=False
    )
    results.append(res_alexnet)

    # Model 3: Multimodal EfficientNet-B0 [Physical] (Slide 13 & 14)
    effnet_phys = MultimodalEfficientNet(tabular_in_dim=len(f_p), num_classes=8, pretrained=False)
    res_eff_phys = train_vision_model(
        "Multimodal EfficientNet-B0 [Physical]", effnet_phys,
        loader_train_p, loader_val_p, loader_test_p,
        y_val_w, y_test_w, y_val_c, y_test_c,
        wind_mean, wind_std, epochs=12, lr=1e-3, is_multimodal=True
    )
    results.append(res_eff_phys)

    # Model 4: Multimodal EfficientNet-B0 [No Pressure] (Slide 13 & 14)
    X_train_np, _, _, scaler_np, imputer_np, f_np = prepare_tabular_inputs(train_subset, use_pressure=False)
    X_val_np, _, _, _, _, _ = prepare_tabular_inputs(val_subset, use_pressure=False, scaler=scaler_np, imputer_vals=imputer_np)
    X_test_np, _, _, _, _, _ = prepare_tabular_inputs(test_subset, use_pressure=False, scaler=scaler_np, imputer_vals=imputer_np)

    ds_train_np = MultimodalCycloneDataset(train_subset, X_train_np, y_train_w_scaled, y_train_c, is_train=True)
    ds_val_np = MultimodalCycloneDataset(val_subset, X_val_np, y_val_w_scaled, y_val_c, is_train=False)
    ds_test_np = MultimodalCycloneDataset(test_subset, X_test_np, y_test_w_scaled, y_test_c, is_train=False)

    loader_train_np = DataLoader(ds_train_np, batch_size=32, shuffle=True)
    loader_val_np = DataLoader(ds_val_np, batch_size=32, shuffle=False)
    loader_test_np = DataLoader(ds_test_np, batch_size=32, shuffle=False)

    effnet_np = MultimodalEfficientNet(tabular_in_dim=len(f_np), num_classes=8, pretrained=False)
    res_eff_np = train_vision_model(
        "Multimodal EfficientNet-B0 [No Pressure]", effnet_np,
        loader_train_np, loader_val_np, loader_test_np,
        y_val_w, y_test_w, y_val_c, y_test_c,
        wind_mean, wind_std, epochs=12, lr=1e-3, is_multimodal=True
    )
    results.append(res_eff_np)

    # Summary Table
    df_results = pd.DataFrame(results)
    out_csv = os.path.join(REPORTS_DIR, "vision_models_performance_comparison.csv")
    df_results.to_csv(out_csv, index=False)

    print("\n" + "=" * 85)
    print("ALL VISION & MULTIMODAL EXPERIMENTAL RESULTS (REPRODUCING SLIDES 9-14)")
    print("=" * 85)
    print(df_results.to_string(index=False))
    print("=" * 85)


if __name__ == "__main__":
    run_all_vision_experiments()

