@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto :error

title Clasificador de residuos hospitalarios
echo Preparando el entorno local...
call "%~dp0preparar_entorno.bat"
if errorlevel 1 goto :error

echo Cargando las bibliotecas. La ventana puede tardar unos segundos en aparecer.
".venv\Scripts\python.exe" -u -X faulthandler "lanzar.pyw" %*
if errorlevel 1 goto :error
exit /b 0

:error
echo No se pudo ejecutar el clasificador. Revisa el mensaje anterior.
pause
exit /b 1
