@echo off
cd /d "%~dp0"
call "%~dp0preparar_entorno.bat"
if errorlevel 1 (
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u "Residuos_Medicos_Pedagogico.py" --entrenar --epocas 24
pause
