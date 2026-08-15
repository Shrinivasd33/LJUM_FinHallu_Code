# Full GPT-4o (OpenAI, PAID - billed per call) experiment runner - RESUMABLE.
# Run this any time; it skips everything already recorded in results/*.jsonl,
# so hitting OpenAI rate limits just means "run it again tomorrow".
#
#   powershell -ExecutionPolicy Bypass -File run_gpt4_full.ps1
#
# Order: zero_shot for all four dimensions first (breadth), then few_shot,
# then the conditional RAG phase (only task-model pairs that erred in zero-shot).

$py = "C:\Users\Sonali\anaconda3\envs\ai_env\python.exe"
Set-Location $PSScriptRoot

foreach ($cond in @("zero_shot", "few_shot", "rag")) {
    foreach ($dim in @("TAI", "CRI", "EFS", "NPI")) {
        Write-Host ""
        Write-Host "=================================================="
        Write-Host " $dim / $cond  ($(Get-Date -Format 'HH:mm:ss'))"
        Write-Host "=================================================="
        & $py src\run_experiments.py --dimension $dim --models gpt-4 --condition $cond
        if ($LASTEXITCODE -ne 0) {
            Write-Host "$dim/$cond stopped (likely rate cap) - re-run this script later to resume." -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "Runner pass complete. Score whatever has been collected so far with:"
Write-Host "  & $py src\run_scoring.py"
