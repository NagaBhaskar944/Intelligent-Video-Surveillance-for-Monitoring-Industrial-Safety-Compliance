@echo off
echo ===========================================
echo       Starting PPE Detection System       
echo ===========================================

:: Use explicit python executable to bypass venv corrupted launcher issues
echo Starting Daphne Server...
start http://127.0.0.1:8000/
venv\Scripts\python.exe -m daphne -b 0.0.0.0 -p 8000 ppe.asgi:application

pause
