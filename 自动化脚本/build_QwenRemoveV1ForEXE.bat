@echo off
call E:\anaconda3\Scripts\activate.bat sam2Mask
cd /d "%~dp0"
echo Starting packaging...
pyinstaller QwenRemoveV1ForEXE.spec
echo.
echo Packaging complete!
echo Executable file location: dist\QwenRemoveV1ForEXE.exe
pause
