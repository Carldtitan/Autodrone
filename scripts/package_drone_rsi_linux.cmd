@echo off
setlocal

set "WORKSPACE=%~dp0.."
for %%I in ("%WORKSPACE%") do set "WORKSPACE=%%~fI"

set "UE_ROOT=C:\Program Files\Epic Games\UE_5.3"
set "VSDEV=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
set "LINUX_MULTIARCH_ROOT=C:\UnrealToolchains\v22_clang-16.0.6-centos7\"

if not exist "%UE_ROOT%\Engine\Build\BatchFiles\RunUAT.bat" (
  echo RunUAT.bat not found under "%UE_ROOT%".
  exit /b 1
)

if not exist "%LINUX_MULTIARCH_ROOT%" (
  echo Linux cross-compile toolchain not found at "%LINUX_MULTIARCH_ROOT%".
  exit /b 1
)

call "%VSDEV%" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1

subst W: /D >nul 2>nul
subst W: "%WORKSPACE%"
if errorlevel 1 exit /b 1

set "PROJECT=W:\unreal\DroneRSI\Blocks.uproject"
set "ARCHIVE=W:\packages\DroneRSI-Linux"

if not exist "%PROJECT%" (
  echo Project not found at "%PROJECT%".
  subst W: /D >nul 2>nul
  exit /b 1
)

if not exist "%ARCHIVE%" mkdir "%ARCHIVE%"

call "%UE_ROOT%\Engine\Build\BatchFiles\RunUAT.bat" BuildCookRun ^
  -project="%PROJECT%" ^
  -noP4 ^
  -nocompileeditor ^
  -platform=Linux ^
  -clientconfig=Development ^
  -serverconfig=Development ^
  -build ^
  -cook ^
  -stage ^
  -pak ^
  -archive ^
  -archivedirectory="%ARCHIVE%" ^
  -map=/Game/FlyingCPP/Maps/FlyingExampleMap ^
  -unattended ^
  -utf8output

set "RESULT=%ERRORLEVEL%"
subst W: /D >nul 2>nul
exit /b %RESULT%
