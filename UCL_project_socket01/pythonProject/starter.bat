@echo off
setlocal


set "CURRENT_DIR=%~dp0"


set "CURRENT_DIR=%CURRENT_DIR:~0,-1%"


set "TARGET_FILE=%CURRENT_DIR%\static\address.txt"


if not exist "%CURRENT_DIR%\static" mkdir "%CURRENT_DIR%\static"


echo %CURRENT_DIR% > "%TARGET_FILE%"

echo Address saved to: %TARGET_FILE%


cd /d CURRENT_DIR


call C:\Users\dell\anaconda3\Scripts\activate.bat
call conda activate pythonProject311


start "Socket Server" cmd /k python server_s.py


timeout /t 3


start "Socket Client" cmd /k python socket_cli.py

exit
