@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Running image processing script...
python main.py

pause
