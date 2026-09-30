param([switch]$CPUOnly)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$taskRequirements = if ($CPUOnly) { 'requirements.txt' } else { 'requirements-gpu.txt' }
if (Get-Command uv -ErrorAction SilentlyContinue) {
    if (-not (Test-Path -LiteralPath $taskPython)) {
        & uv venv .venv --python 3.12
        if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
    }
    & uv pip install --python $taskPython -r $taskRequirements
} else {
    if (-not (Test-Path -LiteralPath $taskPython)) {
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 or uv first.' }
    }
    & $taskPython -m pip install -r $taskRequirements
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $taskPython -X utf8 -c "from pathlib import Path; from plotly.offline import get_plotlyjs; p=Path('static/vendor/plotly.min.js'); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(get_plotlyjs(),encoding='utf-8')"
if ($LASTEXITCODE -ne 0) { throw 'Plotly setup failed.' }
Write-Host 'Ready. Run start_web.ps1 or launch.bat.'
