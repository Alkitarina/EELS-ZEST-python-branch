param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8878,
    [switch]$NoBrowser,
    [ValidateRange(1, 300)]
    [int]$StartupTimeoutSeconds = 60
)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run setup.ps1 first to install the local environment.' }
$taskUrl = "http://127.0.0.1:$Port"
$taskExisting = $null
try { $taskExisting = Invoke-RestMethod -Uri "$taskUrl/api/health" -TimeoutSec 2 } catch {}
if ($taskExisting.app -eq 'eels-zest-parallel' -and $taskExisting.status -eq 'ok') {
    Write-Host "EELS-ZEST Parallel is already running at $taskUrl"
    if (-not $NoBrowser) { Start-Process $taskUrl }
    exit 0
}
if ($null -ne $taskExisting -or (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)) {
    throw "Port $Port is already occupied, but the backend is not healthy. Check it before starting another instance."
}
$taskRuntime = if ($env:ZEST_PARALLEL_RUNTIME) { [System.IO.Path]::GetFullPath($env:ZEST_PARALLEL_RUNTIME) } else { Join-Path $PSScriptRoot 'runtime' }
$taskLogDir = Join-Path $taskRuntime 'logs'
$null = New-Item -ItemType Directory -Path $taskLogDir -Force
$taskStamp = (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6)
$taskStdout = Join-Path $taskLogDir "server-$Port-$taskStamp.stdout.log"
$taskStderr = Join-Path $taskLogDir "server-$Port-$taskStamp.stderr.log"
$taskPreviousEnv = @{}
foreach ($taskKey in @('PYTHONUTF8', 'PYTHONFAULTHANDLER', 'PYTHONUNBUFFERED')) {
    $taskPreviousEnv[$taskKey] = [Environment]::GetEnvironmentVariable($taskKey, 'Process')
    [Environment]::SetEnvironmentVariable($taskKey, '1', 'Process')
}
try {
    # A separate hidden process survives exit/crash of the launching terminal.
    # Keep the uvicorn command and project venv identifiable by stop_web.ps1.
    $taskServer = Start-Process -FilePath $taskPython -WorkingDirectory $PSScriptRoot -WindowStyle Hidden `
        -ArgumentList @('-u', '-X', 'utf8', '-X', 'faulthandler', '-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', "$Port", '--no-access-log') `
        -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr -PassThru
} finally {
    foreach ($taskKey in $taskPreviousEnv.Keys) {
        [Environment]::SetEnvironmentVariable($taskKey, $taskPreviousEnv[$taskKey], 'Process')
    }
}
[ordered]@{
    started_at = (Get-Date).ToString('o'); launcher_pid = $taskServer.Id; port = $Port
    stdout = $taskStdout; stderr = $taskStderr
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskLogDir "latest-$Port.json") -Encoding UTF8
Write-Host "Starting EELS-ZEST Parallel: $taskUrl"
Write-Host "Backend log: $taskStderr"
$taskDeadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
do {
    $taskServer.Refresh()
    if ($taskServer.HasExited) {
        Get-Content -LiteralPath $taskStderr -Tail 25 -Encoding UTF8 -ErrorAction SilentlyContinue | Write-Host
        throw "Backend exited during startup (code $($taskServer.ExitCode)). See $taskStderr"
    }
    $taskHealth = $null
    try { $taskHealth = Invoke-RestMethod -Uri "$taskUrl/api/health" -TimeoutSec 1 } catch {}
    if ($taskHealth.app -eq 'eels-zest-parallel' -and $taskHealth.status -eq 'ok') {
        Write-Host "EELS-ZEST Parallel is ready: $taskUrl"
        Write-Host 'Backend runs independently. Closing this window will not stop computation.'
        Write-Host "To stop and save completed pixels: .\stop_web.ps1 -Port $Port"
        if (-not $NoBrowser) { Start-Process $taskUrl }
        exit 0
    }
    Start-Sleep -Milliseconds 250
} while ([DateTime]::UtcNow -lt $taskDeadline)
throw "Backend did not become ready within $StartupTimeoutSeconds seconds. Its process was kept running; check $taskStderr or use stop_web.ps1 -Port $Port."
