@echo off
setlocal
set "PYTHON_LOCAL=%~dp0.runtime\python312\python.exe"
if exist "%PYTHON_LOCAL%" goto :local
if not exist "%~dp0.venv\Scripts\python.exe" goto :sin_entorno
"%~dp0.venv\Scripts\python.exe" -I -c "import sys, tensorflow, cv2, PIL, numpy, matplotlib; assert sys.version_info[:2] == (3, 12)" >nul 2>&1
if errorlevel 1 goto :sin_entorno
exit /b 0

:local
"%PYTHON_LOCAL%" -I "%~dp0herramientas\preparar_entorno_local.py"
exit /b %errorlevel%

:sin_entorno
echo Falta el entorno de Python 3.12. Ejecuta instalar_dependencias.bat una vez.
exit /b 1
