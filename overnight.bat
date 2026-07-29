@echo off
title N2LN-QEM Overnight Runner (4.2 -> 4.3 -> 5.1)
echo ============================================================
echo Starting Overnight Run: Step 4.2, 4.3, and 5.1
echo ============================================================
echo Start Time: %time%
echo.

echo [1/3] Training SN-D Head (Step 4.2)...
echo This will take 3-6 hours. Please wait...
python experiments/exp1_snd/train.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ ERROR in Step 4.2! Exiting.
    pause
    exit /b %errorlevel%
)

echo.
echo ✅ Step 4.2 completed successfully!
echo.

echo [2/3] Evaluating SN-D (Step 4.3)...
echo This will take 10-15 minutes...
python experiments/exp1_snd/evaluate.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ ERROR in Step 4.3! Exiting.
    pause
    exit /b %errorlevel%
)

echo.
echo ✅ Step 4.3 completed successfully!
echo.

echo [3/3] Generating HN-E Dataset (Step 5.1)...
echo This will take 4-6 hours. Please wait...
python experiments/exp2_hne/generate_data.py

if %errorlevel% neq 0 (
    echo.
    echo ❌ ERROR in Step 5.1! Exiting.
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo ✅ All tasks completed successfully!
echo End Time: %time%
echo ============================================================
echo.
echo Completed:
echo   1. Step 4.2: SN-D Training
echo   2. Step 4.3: SN-D Evaluation
echo   3. Step 5.1: HN-E Dataset Generation
echo.
pause