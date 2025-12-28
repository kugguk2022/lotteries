@echo off
setlocal
echo ==================================================
echo [EUROMILLIONS] Starting Pipeline
echo ==================================================

:: 1. FETCH DRAWS
echo.
echo [1/6] Fetching EuroMillions draws...
python -m euromillions.get_draws --out data/euromillions.csv --append --source auto --allow-stale --allow-partial
if %errorlevel% neq 0 (
    echo [ERROR] get_draws failed.
    exit /b %errorlevel%
)


:: 2. LOTTO LAB (Agent/Discriminator Analysis)
echo.
echo [2/6] Running Lotto Lab Agent...
:: We use the 'all' mode for full analysis
python euromillions_agent/lotto_lab.py --csv data/euromillions.csv --mode all --outdir outputs/euromillions/lottolab --debug
if %errorlevel% neq 0 (
    echo [ERROR] lotto_lab failed.
    exit /b %errorlevel%
)

:: 3. PHASE 2 SOBOL - FEATURES
echo.
echo [3/6] Extracting Phase 2 Features (Sobol)...
python euromillions_agent/phase2_sobol.py features --infile data/euromillions.csv --outdir outputs/euromillions/features --main-k 5 --star-k 2 --include-current
if %errorlevel% neq 0 (
    echo [ERROR] phase2_sobol features failed.
    exit /b %errorlevel%
)

:: 4. GROK (Transformer Training)
echo.
echo [4/6] Training Grok Transformer Model...
:: Using the features extracted in step 3
python euromillions/grok.py --g outputs/euromillions/features/g1.csv --poi outputs/euromillions/features/poi.csv --out-dir outputs/euromillions/grok --epochs 80
if %errorlevel% neq 0 (
    echo [ERROR] grok failed.
    exit /b %errorlevel%
)

:: 5. PHASE 2 SOBOL - TICKETS
echo.
echo [5/6] Generating Sobol Tickets...
python euromillions_agent/phase2_sobol.py tickets --out outputs/euromillions/sobol_tickets.csv --n 50 --main-n 50 --main-k 5 --star-n 12 --star-k 2
if %errorlevel% neq 0 (
    echo [ERROR] phase2_sobol tickets failed.
    exit /b %errorlevel%
)

:: 6. INFER (Frequency Candidates)
echo.
echo [6/6] Generating Frequency Candidates (Infer)...
python euromillions/infer.py --history data/euromillions.csv --n 20 --out outputs/euromillions/infer_candidates.csv --max-ball 50 --max-star 12 --num-balls 5 --num-stars 2 --ball-prefix ball_ --star-prefix star_
if %errorlevel% neq 0 (
    echo [ERROR] infer failed.
    exit /b %errorlevel%
)

echo.
echo ==================================================
echo [SUCCESS] Pipeline Completed!
echo Check outputs in outputs/euromillions/
echo ==================================================
endlocal
pause
