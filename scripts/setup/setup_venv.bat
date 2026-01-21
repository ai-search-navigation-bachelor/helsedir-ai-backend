@echo off
REM Setup script for Windows

echo ========================================
echo Setting up Helsedirektoratet AI Backend
echo ========================================

REM Check if Python is installed
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python version:
py --version

REM Create virtual environment
echo.
echo Creating virtual environment...
if exist venv (
    echo Virtual environment already exists. Skipping creation.
) else (
    py -m venv venv
    echo Virtual environment created successfully!
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist .env (
    echo.
    echo Creating .env file from template...
    copy .env.example .env
    echo .env file created. Please edit it with your settings.
) else (
    echo .env file already exists.
)

REM Create necessary directories
echo.
echo Creating necessary directories...
if not exist data mkdir data
if not exist logs mkdir logs

REM Run setup test
echo.
echo Testing setup...
python scripts\test_setup.py

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To activate the virtual environment, run:
echo   venv\Scripts\activate
echo.
echo To start the server, run:
echo   python scripts\run.py
echo.
echo ========================================

