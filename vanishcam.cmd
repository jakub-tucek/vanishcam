@echo off
rem Windows launcher. macOS and Linux: ./vanishcam. Setup lives in bootstrap.py.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" bootstrap.py %*
  exit /b !errorlevel!
)
for %%v in (3.12 3.13 3.11) do (
  py -%%v -c "" >nul 2>&1
  if !errorlevel! equ 0 (
    py -%%v bootstrap.py %*
    exit /b !errorlevel!
  )
)

echo [VanishCam] Python 3.11-3.13 is required: install it from https://www.python.org/downloads/ 1>&2
exit /b 1
