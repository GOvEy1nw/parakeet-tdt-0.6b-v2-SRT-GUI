@echo off
echo Starting environment configuration and dependency installation...
REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
echo Error: Python not found. Please install Python 3.12.2 or higher and add it to the PATH.
echo You can download it from https://www.python.org/downloads/.
pause
exit /b
)
echo Python has been detected.


REM Check if the virtual environment exists
if not exist .\.venv\Scripts\activate.bat (
	REM Create the virtual environment
	echo Creating virtual environment .venv...
	python -m venv .venv
	if %errorlevel% neq 0 (
	echo Error: Failed to create the virtual environment.
	pause
	exit /b
	)
	echo Virtual environment .venv created successfully.
)

REM Activate the virtual environment
call .\.venv\Scripts\activate.bat
echo Virtual environment activated.

echo.
echo ===============================================================================
echo About PyTorch Installation (Important!):
echo ===============================================================================
echo If you have an NVIDIA GPU and want to use CUDA acceleration, it is highly recommended that you:
echo    1. Visit the PyTorch official website (https://pytorch.org/get-started/locally/)
echo    2. Get the PyTorch installation command suitable for your CUDA version.
echo    3. In a [new command line window, first activate this virtual environment (.venv\Scripts\activate)],
echo       then [manually execute] that PyTorch installation command.
echo    4. After manually installing the PyTorch GPU version, return to [this window] and press 'n' to skip the automatic installation.
echo.
echo If you are not sure, only want to use the CPU, or have already manually installed the GPU version of PyTorch,
echo the script can try to install a general PyTorch (usually the CPU version), or skip it.
echo ===============================================================================
echo.


set /p choice="Do you want to install torch (cpu version) and its dependencies? (y/n): "
if /i "%choice%"=="y" (
	echo Installing torch and its dependencies

	pip install -r requirements.txt

	if %errorlevel% neq 0 (
		echo Error: Failed to install dependencies.
		pause
		exit /b
	)
	echo Dependencies installed successfully.
) else (
		pause
		exit /b
)
 

echo requirements dependencies installation complete.

echo.
echo ===============================================================================
echo The installation process for all dependencies has been completed.
echo.
echo Next steps:
echo    Double-click 'launcher.bat' to start the application.
echo.
pause
exit /b
