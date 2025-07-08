@echo off
echo Starting Phishing Email Detector UI...
echo.
echo The web interface will be available at:
echo   http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.
set OTEL_SDK_DISABLED=true
python app.py