@echo off
setlocal

set "WORKSPACE=%~dp0.."
for %%I in ("%WORKSPACE%") do set "WORKSPACE=%%~fI"

subst W: /D >nul 2>nul
subst W: "%WORKSPACE%"
if errorlevel 1 exit /b 1

set "VSWHERE_DIR=C:\Program Files (x86)\Microsoft Visual Studio\Installer"
set "VSDEV=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
set "VSCMAKE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"

set "PATH=%VSWHERE_DIR%;%PATH%"
call "%VSDEV%" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1

W:
cd \external\Colosseum
if errorlevel 1 exit /b 1

set "PATH=%VSCMAKE%;%PATH%"
call .\build.cmd --no-full-poly-car --Release
exit /b %ERRORLEVEL%
