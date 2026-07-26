@echo off
rem Build the frozen Spectrum Explorer (onedir, windowed).
rem Output: dist\spectrum_explorer\spectrum_explorer.exe
rem
rem Close any running spectrum_explorer.exe first — a locked file under
rem dist\ makes the rebuild fail with "Access is denied" (if it still
rem fails with nothing running, an antivirus scan of the fresh .pyd
rem files is usually holding the handle; retry after a few seconds).

cd /d "%~dp0"

rem Pick the interpreter: SPECTRA_PYTHON if set, else a virtualenv beside
rem the repo, else the launcher. The build must run against an environment
rem carrying the full stack — a build from one without specutils SUCCEEDS
rem while silently baking in a crippled continuum auto-fit, so the check
rem below is what stops that.
set "PY=%SPECTRA_PYTHON%"
if not defined PY if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY set "PY=py -3.13"

rem Hard-fail rather than ship the silent degradation.
%PY% -c "import specutils, astroquery" 2>nul
if errorlevel 1 (
    echo.
    echo BUILD ABORTED: specutils / astroquery missing from %PY%
    echo   %PY% -m pip install "specutils>=2.0" "astroquery>=0.4.8"
    echo.
    pause
    exit /b 1
)
%PY% -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo.
    echo BUILD ABORTED: PyInstaller missing from %PY%
    echo   %PY% -m pip install "pyinstaller>=6.10"
    echo.
    pause
    exit /b 1
)

if exist dist\spectrum_explorer rmdir /s /q dist\spectrum_explorer
%PY% -m PyInstaller spectrum_explorer.spec --noconfirm
pause
