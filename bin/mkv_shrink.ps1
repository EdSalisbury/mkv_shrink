# Launch helper for mkv_shrink on Windows.
# Activates the local .venv and runs mkv_shrink/watcher.py with any additional arguments.

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$projectRoot = "D:\Dev\mkv_shrink"
$scriptRel   = "mkv_shrink\watcher.py"
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate)) {
    Write-Error "Virtual environment not found at: $venvActivate"
    exit 1
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& $venvActivate

$scriptPath = Join-Path $projectRoot $scriptRel
if (-not (Test-Path $scriptPath)) {
    Write-Error "Python script not found: $scriptPath"
    exit 1
}

# Clean up PowerShell auto-quoting and resolve paths when possible
$cleanArgs = foreach ($a in $Args) {
    $fixed = $a.Trim('"', "'") -replace "''", "'"
    if (Test-Path $fixed) { (Resolve-Path $fixed).Path } else { $fixed }
}

Push-Location $projectRoot
Write-Host "Running: python $scriptRel $cleanArgs" -ForegroundColor Green
python $scriptPath @cleanArgs
Pop-Location
