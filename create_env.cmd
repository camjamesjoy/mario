@echo off
@setlocal

set ROOT=%~dp0
set ROOT=%ROOT:~0,-1%
set PACKAGE_NAME=mario
set ENV_DIR=%ROOT%\env\%PACKAGE_NAME%
set ACTIVATE_SCRIPT=%ENV_DIR%\Scripts\activate.bat
set REQ_FILE=%ROOT%\requirements.txt

echo.
echo making environment "%ENV_DIR%"
python -m venv --clear "%ENV_DIR%"
if ERRORLEVEL 1 (
    echo.
    echo Failed to create environment 1>&2
    exit /b 1
)

if not exist "%ENV_DIR%\." (
    echo.
    echo Failed to create environment 1>&2
    exit /b 1
)

echo.
echo activating environment
if not exist "%ACTIVATE_SCRIPT%" (
    echo.
    echo activation script not found 1>&2
    exit /b 1
)
call "%ACTIVATE_SCRIPT%"
if ERRORLEVEL 1 (
    echo.
    echo Failed to activate environment 1>&2
    exit /b 1
)

echo.
echo updating pip
python -m pip install --upgrade pip

echo.
echo installing requirements
python -m pip install -r "%REQ_FILE%"
python -m pip install --no-index --no-deps -e "%ROOT%"
