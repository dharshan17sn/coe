"""
Training and Evaluation Pipeline for Tabular MLP (Baseline Experiment).
Reproduces:
1. Tabular MLP [physical] - with central pressure and coordinates
2. Tabular MLP [no_pressure] - ablation excluding central pressure

Evaluates on:
- Validation Set (2020 - 2022)
- Test Holdout Set (2023 - 2026, unseen cyclones)
Computes: RMSE, MAE, R2, Bias, and IMD Class Accuracy.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.tabular_mlp import TabularMLP

MODELS_DIR = "models"
REPORTS_DIR = "reports"


class CycloneDataset(Dataset):
    def __init__(self, X, y_wind, y_class):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_wind = torch.tensor(y_wind, dtype=torch.float32).unsqueeze(1)
        self.y_class = torch.tensor(y_class, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_wind[idx], self.y_class[idx]


def prepare_features(df, use_pressure=True, scaler=None, imputer_vals=None):
    """
    Extracts numerical features and scales them cleanly without data leakage.
    """
    df = df.copy()

    # Subbasin encoding: BB = 1 (Bay of Bengal), AS = 0 (Arabian Sea)
    df["SUBBASIN_CODE"] = (df["SUBBASIN"] == "BB").astype(float)

    # Missing distance to land filled with mean
    if "DIST2LAND" not in df.columns:
        df["DIST2LAND"] = 0.0

    if use_pressure:
        feature_cols = ["PRES_HPA", "LAT", "LON", "DIST2LAND", "SUBBASIN_CODE"]
    else:
        feature_cols = ["LAT", "LON", "DIST2LAND", "SUBBASIN_CODE"]

    # Impute missing values using train statistics
    if imputer_vals is None:
        imputer_vals = {}
        for col in feature_cols:
            imputer_vals[col] = df[col].median()

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


def train_single_model(config_name, use_pressure, train_df, val_df, test_df, epochs=120, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"TRAINING EXPERIMENT: {config_name}")
    print(f"{'='*60}")

    # Prepare data
    X_train, y_train_wind, y_train_class, scaler, imputer, f_cols = prepare_features(
        train_df, use_pressure=use_pressure
    )
    X_val, y_val_wind, y_val_class, _, _, _ = prepare_features(
        val_df, use_pressure=use_pressure, scaler=scaler, imputer_vals=imputer
    )
    X_test, y_test_wind, y_test_class, _, _, _ = prepare_features(
        test_df, use_pressure=use_pressure, scaler=scaler, imputer_vals=imputer
    )

    # Target scaling for stable SmoothL1 convergence (scale by train std)
    wind_mean = float(y_train_wind.mean())
    wind_std = float(y_train_wind.std())

    y_train_wind_scaled = (y_train_wind - wind_mean) / wind_std
    y_val_wind_scaled = (y_val_wind - wind_mean) / wind_std

    train_dataset = CycloneDataset(X_train, y_train_wind_scaled, y_train_class)
    val_dataset = CycloneDataset(X_val, y_val_wind_scaled, y_val_class)
    test_dataset = CycloneDataset(X_test, (y_test_wind - wind_mean) / wind_std, y_test_class)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    in_dim = len(f_cols)
    model = TabularMLP(in_features=in_dim, hidden_dims=[128, 64, 32], dropout_rate=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)
    reg_criterion = nn.SmoothL1Loss()
    cls_criterion = nn.CrossEntropyLoss()

    best_val_rmse = float("inf")
    best_weights = None
    best_epoch = 0

    history = {"train_loss": [], "val_rmse": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for bx, by_wind, by_cls in train_loader:
            optimizer.zero_grad()
            p_wind, p_cls = model(bx)
            loss_wind = reg_criterion(p_wind, by_wind)
            loss_cls = cls_criterion(p_cls, by_cls)
            loss = loss_wind + 0.3 * loss_cls
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(bx)

        epoch_loss = running_loss / len(train_dataset)

        # Validation evaluation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for bx, _, _ in val_loader:
                pw, _ = model(bx)
                # Unscale back to knots
                val_preds.extend((pw.squeeze(1).numpy() * wind_std + wind_mean).tolist())

        val_rmse = float(np.sqrt(mean_squared_error(y_val_wind, val_preds)))
        scheduler.step(val_rmse)

        history["train_loss"].append(epoch_loss)
        history["val_rmse"].append(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if epoch % 20 == 0 or epoch == epochs or epoch == 1:
            print(f" Epoch [{epoch:3d}/{epochs:3d}] | Train Loss: {epoch_loss:.4f} | Val RMSE: {val_rmse:.2f} kt (Best: {best_val_rmse:.2f} kt @ ep {best_epoch})")

    # Load best weights
    model.load_state_dict(best_weights)

    # Final Validation Metrics
    model.eval()
    with torch.no_grad():
        v_pw, v_pc = model(torch.tensor(X_val, dtype=torch.float32))
        val_final_wind = v_pw.squeeze(1).numpy() * wind_std + wind_mean
        val_final_cls = torch.argmax(v_pc, dim=1).numpy()

    val_rmse_final = float(np.sqrt(mean_squared_error(y_val_wind, val_final_wind)))
    val_mae_final = float(mean_absolute_error(y_val_wind, val_final_wind))
    val_r2_final = float(r2_score(y_val_wind, val_final_wind))
    val_acc_final = float(accuracy_score(y_val_class, val_final_cls))

    # Test Metrics (Holdout 2023 - 2026)
    with torch.no_grad():
        t_pw, t_pc = model(torch.tensor(X_test, dtype=torch.float32))
        test_final_wind = t_pw.squeeze(1).numpy() * wind_std + wind_mean
        test_final_cls = torch.argmax(t_pc, dim=1).numpy()

    test_rmse = float(np.sqrt(mean_squared_error(y_test_wind, test_final_wind)))
    test_mae = float(mean_absolute_error(y_test_wind, test_final_wind))
    test_r2 = float(r2_score(y_test_wind, test_final_wind))
    test_bias = float(np.mean(test_final_wind - y_test_wind))
    test_acc = float(accuracy_score(y_test_class, test_final_cls))

    # Save model checkpoint
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_save_path = os.path.join(MODELS_DIR, f"{config_name.lower().replace(' ', '_')}.pt")
    torch.save({
        "model_state_dict": best_weights,
        "features": f_cols,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "imputer_vals": imputer,
        "wind_mean": wind_mean,
        "wind_std": wind_std,
        "best_epoch": best_epoch,
        "val_rmse": val_rmse_final,
        "test_rmse": test_rmse,
    }, model_save_path)
    print(f"Model saved to: {model_save_path}")

    return {
        "Model": config_name,
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


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join("datasets", "splits", "train.csv"))
    val_df = pd.read_csv(os.path.join("datasets", "splits", "validation.csv"))
    test_df = pd.read_csv(os.path.join("datasets", "splits", "test.csv"))

    print(f"Loaded Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    results = []

    # Experiment 1: Tabular MLP [physical] (with pressure)
    res_physical = train_single_model(
        "Tabular MLP [physical]",
        use_pressure=True,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        epochs=150,
        lr=2e-3
    )
    results.append(res_physical)

    # Experiment 2: Tabular MLP [no_pressure] (ablation)
    res_no_pres = train_single_model(
        "Tabular MLP [no_pressure]",
        use_pressure=False,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        epochs=100,
        lr=2e-3
    )
    results.append(res_no_pres)

    # Format into comparison table matching Slide 14!
    results_df = pd.DataFrame(results)
    results_table_path = os.path.join(REPORTS_DIR, "tabular_mlp_results_comparison.csv")
    results_df.to_csv(results_table_path, index=False)

    print("\n" + "=" * 80)
    print("FINAL RESULTS TABLE - MODEL PERFORMANCE COMPARISON (SLIDE 14 REPRODUCED)")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

