@echo off
REM Script to set up the environment for the MLP Language Model on Windows

echo Setting up environment for MLP Language Model...

REM Create virtual environment if it doesn't exist
if not exist mlp_env (
    echo Creating virtual environment...
    python -m venv mlp_env
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment
echo Activating virtual environment...
call mlp_env\Scripts\activate

REM Install required dependencies
echo Installing required dependencies...
pip install -r requirements.txt

REM Ask if user wants to install optional dependencies
set /p REPLY=Do you want to install optional dependencies for enhanced features? (y/n)
if /i "%REPLY%"=="y" (
    echo Installing optional dependencies...
    pip install gensim transformers torch tokenizers
)

REM Create screenshots directory if it doesn't exist
if not exist screenshots (
    echo Creating screenshots directory...
    mkdir screenshots
)

REM Print success message
echo.
echo Setup complete! You can now run the UI applications:
echo   python complete_mlp_ui.py    # Complete UI with all features
echo   python basic_mlp_ui.py       # Basic UI with core functionality
echo   python run_standard_mlp.py   # Standard MLP UI
echo.
echo See README.md and UI_GUIDE.md for more information.

REM Keep the window open
pause