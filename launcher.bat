@echo off

echo Attempting to activate virtual environment .venv...

REM Check if the virtual environment exists
if not exist .\.venv\Scripts\activate.bat (
    echo Error: Virtual environment .venv not found or is incomplete.
    echo Please run install_dependencies.bat first to create and configure the environment.
    pause
    exit /b
)

call .\.venv\Scripts\activate.bat
echo Virtual environment activated.

echo.
echo Starting the application (main.py)...
echo If the application doesn't show an interface immediately, please check the command line output for error messages.
echo Gradio will usually print a local URL (e.g., http://127.0.0.1:7860).
echo The first launch may take a while.
echo.

python main.py

echo.
echo Application has been closed or has finished.
pause
