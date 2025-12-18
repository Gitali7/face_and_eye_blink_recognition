@echo off
echo Starting Face & Eye Blink Recognition App...
cd /d "%~dp0"
call venv\Scripts\activate
cd backend
python main.py
pause
