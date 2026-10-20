param(
    [string]$RunId = "experiment-v2-002",
    [string]$SmokeRunId = "smoke-v2-006",
    [string]$BaselineSmokeRunId = "smoke-v2-004",
    [string]$DataRunId = "data-v2-001",
    [switch]$ForceSmokeRerun = $false
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

# Ensure clean Python environment isolation
$env:PYTHONPATH = ""
$env:PYTHONNOUSERSITE = "1"
$BaseVenvScripts = Join-Path $ProjectRoot ".venv\Scripts"
$BasePython = Join-Path $BaseVenvScripts "python.exe"
$env:PATH = "$BaseVenvScripts;$env:PATH"

$AccDir = Join-Path $ProjectRoot "artifacts\acceptance"
if (Test-Path $AccDir) {
    Remove-Item -LiteralPath $AccDir -Recurse -Force
}
New-Item -ItemType Directory -Path $AccDir -Force | Out-Null
$LogFile = Join-Path $AccDir "acceptance.log"

$StepTimings = @()

function Invoke-AcceptanceStep {
    param(
        [string]$StepId,
        [string]$Description,
        [string]$CommandDisplay,
        [scriptblock]$Action
    )
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host "[$StepId] $Description" -ForegroundColor Cyan
    Write-Host "Command: $CommandDisplay" -ForegroundColor Gray
    Add-Content -Path $LogFile -Value "=================================================================="
    Add-Content -Path $LogFile -Value "[$StepId] $Description"
    Add-Content -Path $LogFile -Value "Command: $CommandDisplay"

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        & $Action 2>&1 | Tee-Object -FilePath $LogFile -Append | Out-Host
        if ($LASTEXITCODE -ne 0) {
            throw "Step $StepId exited with non-zero exit code: $LASTEXITCODE"
        }
        $sw.Stop()
        $script:StepTimings += [ordered]@{
            step_id = $StepId
            description = $Description
            command = $CommandDisplay
            status = "PASSED"
            duration_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 4)
        }
        Write-Host "[$StepId] PASSED ($([math]::Round($sw.Elapsed.TotalSeconds, 2))s)" -ForegroundColor Green
    }
    catch {
        $sw.Stop()
        $script:StepTimings += [ordered]@{
            step_id = $StepId
            description = $Description
            command = $CommandDisplay
            status = "FAILED"
            duration_seconds = [math]::Round($sw.Elapsed.TotalSeconds, 4)
            error = $_.Exception.Message
        }
        Write-Host "[$StepId] FAILED: $($_.Exception.Message)" -ForegroundColor Red
        throw
    }
}

$AccVenvDir = Join-Path $ProjectRoot ".venv-acceptance"
$AccPython = Join-Path $AccVenvDir "Scripts\python.exe"

# G01: Clean .venv-acceptance setup & lockfile verification (A17)
Invoke-AcceptanceStep -StepId "G01_CLEAN_VENV" -Description "Create clean .venv-acceptance and verify locked dependencies + editable install" -CommandDisplay "python -m venv .venv-acceptance && pip install -r requirements-lock.txt && pip install --no-deps -e . && pip check" -Action {
    & $BasePython -m venv --clear --without-pip --system-site-packages $AccVenvDir
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $AccPython -m pip install --no-deps --no-index -r requirements-lock.txt
    if ($LASTEXITCODE -ne 0) {
        & $AccPython -m pip install -r requirements-lock.txt
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }

    & $AccPython -m pip install --no-deps --no-build-isolation -e .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & $AccPython -m pip check
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# G02: Inspect frozen v2 configurations (A01, A17)
Invoke-AcceptanceStep -StepId "G02_INSPECT_CONFIGS" -Description "Validate research-v2, smoke-v2, and experiment-v2 configurations via CLI" -CommandDisplay "python -m olist_delay inspect-config --config configs/{research,smoke,experiment}-v2.json" -Action {
    & $AccPython -m olist_delay inspect-config --config configs/research-v2.json
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $AccPython -m olist_delay inspect-config --config configs/smoke-v2.json
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $AccPython -m olist_delay inspect-config --config configs/experiment-v2.json
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# G03: Unit test suite (zero-skip, zero-fail)
$JunitUnit = Join-Path $AccDir "junit-unit.xml"
Invoke-AcceptanceStep -StepId "G03_PYTEST_UNIT" -Description "Run unit test suite (-m unit) with JUnit XML capture" -CommandDisplay "python -m pytest -ra -m unit --junitxml=artifacts/acceptance/junit-unit.xml" -Action {
    & $AccPython -m pytest -ra -m unit "--junitxml=$JunitUnit"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# G04: Integration test suite (zero-skip, zero-fail)
$JunitInteg = Join-Path $AccDir "junit-integration.xml"
Invoke-AcceptanceStep -StepId "G04_PYTEST_INTEGRATION" -Description "Run end-to-end integration test suite (-m integration) with JUnit XML capture" -CommandDisplay "python -m pytest -ra -m integration --junitxml=artifacts/acceptance/junit-integration.xml" -Action {
    & $AccPython -m pytest -ra -m integration "--junitxml=$JunitInteg"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# G05: Verify prepared data snapshot & clean-environment smoke backtest (A18, W7-01)
Invoke-AcceptanceStep -StepId "G05_PREPARE_AND_SMOKE" -Description "Verify prepared data snapshot ($DataRunId) and smoke backtest ($SmokeRunId)" -CommandDisplay "python -m olist_delay verify --run-id $SmokeRunId --allow-smoke --no-update-manifest" -Action {
    $DataDir = Join-Path $ProjectRoot "artifacts\runs\$DataRunId"
    if (-not (Test-Path $DataDir)) {
        & $AccPython -m olist_delay prepare --config configs/research-v2.json --run-id $DataRunId
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    $SmokeDir = Join-Path $ProjectRoot "artifacts\runs\$SmokeRunId"
    if ((-not (Test-Path $SmokeDir)) -or $ForceSmokeRerun) {
        & $AccPython -m olist_delay backtest --config configs/smoke-v2.json --run-id $SmokeRunId --parent-run-id $BaselineSmokeRunId --rerun-reason "W7 clean .venv-acceptance reproducibility verification"
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    & $AccPython -m olist_delay verify --run-id $SmokeRunId --allow-smoke --no-update-manifest
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# G06: Verify formal run and report narrative sync (A16, A18)
Invoke-AcceptanceStep -StepId "G06_VERIFY_FORMAL_RUN" -Description "Verify formal run ($RunId) and report/presentation claim synchronization (--check-docs)" -CommandDisplay "python -m olist_delay verify --run-id $RunId --no-update-manifest --check-docs" -Action {
    & $AccPython -m olist_delay verify --run-id $RunId --no-update-manifest --check-docs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# Record pre-evaluate hashes of formal run files to prove read-only evaluate invariance (A13)
$PreEvalHashesJson = Join-Path $AccDir "pre_eval_hashes.json"
& $AccPython -c "import json, pathlib, hashlib; r = pathlib.Path('artifacts/runs/$RunId'); files = ['manifest.json', 'metrics_summary.json', 'backtest_predictions.csv', 'validation_predictions.csv', 'daily_policy.csv', 'claims.json']; h = {f: hashlib.sha256((r/f).read_bytes()).hexdigest() for f in files}; pathlib.Path(r'$PreEvalHashesJson').write_text(json.dumps(h, indent=2), encoding='utf-8')"

# G07: Dual independent read-only evaluations (A13, W7-01)
$EvalDir1 = Join-Path $AccDir "eval-run1"
$EvalDir2 = Join-Path $AccDir "eval-run2"
Invoke-AcceptanceStep -StepId "G07_DUAL_EVALUATE" -Description "Run two independent read-only evaluations on $RunId to verify deterministic policy & bootstrap output" -CommandDisplay "python -m olist_delay evaluate --run-id $RunId --output-dir artifacts/acceptance/eval-run{1,2}" -Action {
    & $AccPython -m olist_delay evaluate --run-id $RunId --output-dir $EvalDir1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $AccPython -m olist_delay evaluate --run-id $RunId --output-dir $EvalDir2
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# G08: Independent recomputation of claims and metrics (A15)
$RecomputeJson = Join-Path $AccDir "recompute_report.json"
Invoke-AcceptanceStep -StepId "G08_INDEPENDENT_RECOMPUTE" -Description "Recompute all 38 metrics and claims independently from per-order predictions" -CommandDisplay "python scripts/recompute_results.py --run-id $RunId --output artifacts/acceptance/recompute_report.json" -Action {
    & $AccPython scripts/recompute_results.py --run-id $RunId --output $RecomputeJson
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# G09: Dual independent offline batch scoring runs (A09, A10, W7-01)
$ScoreOut1 = Join-Path $AccDir "score-run1\predictions.csv"
$ScoreOut2 = Join-Path $AccDir "score-run2\predictions.csv"
Invoke-AcceptanceStep -StepId "G09_DUAL_OFFLINE_SCORE" -Description "Run two independent offline batch scoring invocations on 2018-06-01 snapshot" -CommandDisplay "python -m olist_delay score --run-id $RunId --as-of 2018-06-01T23:59:59.999999 --output artifacts/acceptance/score-run{1,2}/predictions.csv" -Action {
    & $AccPython -m olist_delay score --run-id $RunId --input "artifacts/runs/$DataRunId/sample_orders.csv" --history-events "artifacts/runs/$DataRunId/sample_events.csv" --decision-events "artifacts/runs/$DataRunId/sample_decision_events.csv" --as-of "2018-06-01T23:59:59.999999" --output $ScoreOut1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $AccPython -m olist_delay score --run-id $RunId --input "artifacts/runs/$DataRunId/sample_orders.csv" --history-events "artifacts/runs/$DataRunId/sample_events.csv" --decision-events "artifacts/runs/$DataRunId/sample_decision_events.csv" --as-of "2018-06-01T23:59:59.999999" --output $ScoreOut2
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# G10: Rebuild the report analysis layer in the clean environment (compared with the committed facts in G12)
$RecheckDir = "artifacts/analysis/acceptance-recheck"
$RobustRecheckDir = "artifacts/analysis/acceptance-robustness-recheck"
Invoke-AcceptanceStep -StepId "G10_REPORT_ANALYSIS_RECHECK" -Description "Rebuild report facts, tables and figures from the frozen run" -CommandDisplay "python scripts/report_analysis.py --run-id $RunId --output-dir $RecheckDir --replace" -Action {
    & $AccPython scripts/report_analysis.py --run-id $RunId --output-dir $RecheckDir --figures-dest "artifacts/acceptance/figures-recheck" --replace
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $AccPython scripts/robustness_analysis.py --main-run $RunId --output-dir $RobustRecheckDir --figures-dest "artifacts/acceptance/figures-recheck" --replace
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# G11: Every number in the narrative documents traces to a fact, every figure link resolves, the deck embeds every image
$DocCheckJson = "artifacts/acceptance/doc_consistency.json"
Invoke-AcceptanceStep -StepId "G11_DOC_CONSISTENCY" -Description "Check narrative documents against report facts and figure files" -CommandDisplay "python scripts/check_docs_consistency.py --output $DocCheckJson" -Action {
    & $AccPython scripts/check_docs_consistency.py --output $DocCheckJson
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$StepTimingsJson = Join-Path $AccDir "step_timings.json"
$StepTimings | ConvertTo-Json -Depth 5 | Set-Content -Path $StepTimingsJson -Encoding UTF8

# G12: Finalize acceptance summary, cross-run hash comparisons, and delivery/ package (W7-02)
Invoke-AcceptanceStep -StepId "G12_PACKAGE_DELIVERY" -Description "Verify cross-run determinism, source/input hash invariance, report-layer reproducibility, and assemble delivery/ package" -CommandDisplay "python scripts/finalize_acceptance.py --run-id $RunId --smoke-run-id $SmokeRunId" -Action {
    & $AccPython scripts/finalize_acceptance.py --run-id $RunId --smoke-run-id $SmokeRunId --baseline-smoke-run-id $BaselineSmokeRunId --data-run-id $DataRunId --acceptance-dir "artifacts/acceptance" --pre-eval-hashes-json $PreEvalHashesJson --step-timings-json $StepTimingsJson --report-recheck-dir $RecheckDir --robustness-recheck-dir $RobustRecheckDir --doc-check-json $DocCheckJson
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "==================================================================" -ForegroundColor Green
Write-Host "ALL W7 ACCEPTANCE GATES PASSED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "Summary JSON : delivery/acceptance-summary.json" -ForegroundColor Green
Write-Host "Summary MD   : delivery/acceptance-summary.md" -ForegroundColor Green
Write-Host "Checksums    : delivery/file-checksums.json" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
