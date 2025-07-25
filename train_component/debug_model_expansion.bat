@echo off
chcp 65001 >nul
echo Starting model expansion script (debug mode)...
echo.
echo Auto-selecting parameters:
echo - Model: 1 (models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B)
echo - Expansion: 1 (preset size)
echo - Target size: 2 (1.8b)
echo - Epochs: 2
echo - Batch size: 2
echo - Limit data: y (1 line)
echo - Start training: y
echo.
echo Note: Using gradient accumulation strategy (no more force training)
echo.

REM Create temporary input file with proper line endings
(
echo 1
echo 1
echo 2
echo 1
echo 1
echo y
echo 1
echo y
) > temp_input.txt

REM Run script with input redirection
C:/Python313/python.exe -u model_expansion.py < temp_input.txt

REM Clean up temporary file
if exist temp_input.txt del temp_input.txt

echo.
echo Script execution completed, press any key to exit...
pause >nul 