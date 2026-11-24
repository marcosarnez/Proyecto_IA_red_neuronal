@echo off
setlocal
cd /d "%~dp0"
py -3.12 --version >nul 2>&1
if errorlevel 1 (
  echo Instala Python 3.12 para Windows antes de continuar.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv ".venv"
if errorlevel 1 (
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 (
  echo La instalacion de dependencias no se completo.
  pause
  exit /b 1
)
echo Entorno preparado. Ya puedes abrir iniciar.bat.
pause
