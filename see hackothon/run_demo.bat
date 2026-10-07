@echo off
title Linklens - Phishing URL Detection
echo ===================================================
echo     Starting Linklens Phishing Detection Service
echo ===================================================
echo [1/2] Starting Flask ML Backend on http://127.0.0.1:5000...
start /b python app.py
timeout /t 2 /nobreak >nul
echo [2/2] Opening Linklens Frontend in default browser...
start "" "index.html"
echo.
echo [OK] Linklens is live! Press any key to run verification tests...
pause >nul
python test_urls.py
pause
