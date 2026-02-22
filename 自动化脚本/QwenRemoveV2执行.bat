@echo off
call E:\anaconda3\Scripts\activate.bat sam2Mask
cd /d "%~dp0"
python QwenRemoveV2.py
pause
