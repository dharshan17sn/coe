@echo off
title Cyclone Eye — Vision-Based Real-Time Monitoring & Prediction Platform
echo =====================================================================
echo  CYCLONE EYE: VISION & METEOROLOGICAL PREDICTION PLATFORM (INDIA)
echo =====================================================================
echo.
echo Activating Python virtual environment...
call .\.venv\Scripts\activate.bat
echo.
echo Starting FastAPI Backend Server on http://127.0.0.1:8000 ...
echo Opening interactive dashboard in your browser...
start "" "http://127.0.0.1:8000"
echo.
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
pause

