[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8878,
    [ValidateRange(1, 3600)]
    [int]$TimeoutSeconds = 900,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$taskUrl = "http://127.0.0.1:$Port"
$taskPython = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '.venv\Scripts\python.exe'))

function Get-BackendListener {
    $owners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique)
    if ($owners.Count -eq 0) { return $null }
    if ($owners.Count -ne 1) { throw "端口 $Port 对应多个进程，无法确定后端，未终止任何进程。" }
    return Get-CimInstance Win32_Process -Filter "ProcessId = $($owners[0])"
}

function Assert-ProjectBackend($Process) {
    $portPattern = '(?i)(?:^|\s)--port(?:\s+|=)"?' + $Port + '"?(?:\s|$)'
    if ($Process.CommandLine -notmatch '(?i)(?:^|\s)-m\s+uvicorn\s+"?app:app"?(?:\s|$)' -or
        $Process.CommandLine -notmatch $portPattern) {
        throw "端口 $Port 的进程不是本启动脚本使用的 Uvicorn 后端，未终止任何进程。"
    }
    # On Windows, the venv python.exe may launch the shared base interpreter.
    # Check its ancestors as well, never identify the project by 'python' alone.
    $ancestor = $Process
    for ($depth = 0; $depth -lt 8 -and $null -ne $ancestor; $depth++) {
        if ([string]::Equals($ancestor.ExecutablePath, $taskPython,
                [System.StringComparison]::OrdinalIgnoreCase)) { return }
        if ($ancestor.ParentProcessId -le 0) { break }
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($ancestor.ParentProcessId)"
        if ($null -eq $parent -or $parent.CreationDate -gt $ancestor.CreationDate) { break }
        $ancestor = $parent
    }
    throw "端口 $Port 的后端不属于本目录的 Python 环境，未终止任何进程。"
}

function Test-SameBackend($Expected) {
    $current = Get-BackendListener
    if ($null -eq $current) { return $false }
    if ($current.ProcessId -ne $Expected.ProcessId -or
        $current.CreationDate -ne $Expected.CreationDate) {
        throw "等待期间端口 $Port 的进程已更换，已停止操作，不会终止新进程。"
    }
    return $true
}

$taskBackend = Get-BackendListener
if ($null -eq $taskBackend) {
    Write-Host "端口 $Port 没有正在监听的后端，无需停止。"
    return
}
Assert-ProjectBackend $taskBackend

$taskHealth = $null
try { $taskHealth = Invoke-RestMethod -Uri "$taskUrl/api/health" -TimeoutSec 5 } catch {
    if (-not $Force) {
        throw '后端接口没有响应，无法确认保存状态，未终止进程。需要立即终止时请显式使用 -Force。'
    }
}
if ($null -ne $taskHealth -and $taskHealth.app -ne 'eels-zest-parallel') {
    throw "端口 $Port 返回的服务标识不是 EELS-ZEST Parallel，未终止任何进程。"
}

if (-not $PSCmdlet.ShouldProcess("EELS-ZEST Parallel · $taskUrl · PID $($taskBackend.ProcessId)",
        '停止后端及其子进程')) { return }

if ($Force) {
    Write-Warning '强制停止：不会等待计算和导出完成，尚未保存的结果可能丢失。'
} else {
    $taskDeadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    $taskCancelled = @{}
    try {
        while ($true) {
            if (-not (Test-SameBackend $taskBackend)) {
                Write-Host '后端已退出。'
                return
            }
            try { $taskJobs = Invoke-RestMethod -Uri "$taskUrl/api/jobs" -TimeoutSec 5 } catch {
                throw '无法读取任务状态，已停止操作。需要立即终止时请显式使用 -Force。'
            }
            $taskActive = @($taskJobs | Where-Object { $_.status -in @('queued', 'running') })
            if ($taskActive.Count -eq 0) { break }
            foreach ($job in $taskActive) {
                if ($job.phase -ne 'export' -and -not $taskCancelled.ContainsKey($job.id)) {
                    $null = Invoke-RestMethod -Method Post -Uri "$taskUrl/api/jobs/$($job.id)/cancel" `
                        -ContentType 'application/json' -Body '{}' -TimeoutSec 5
                    $taskCancelled[$job.id] = $true
                    Write-Host "已请求取消任务 $($job.id)，等待保存已完成的像素…"
                }
            }
            if ([DateTime]::UtcNow -ge $taskDeadline) {
                throw "等待保存超过 $TimeoutSeconds 秒，后端仍保留运行。可稍后重试、增加 -TimeoutSeconds，或显式使用 -Force。"
            }
            Write-Progress -Activity '等待任务保存与导出完成' -Status ([string]$taskActive[0].message)
            Start-Sleep -Seconds 1
        }
    } finally {
        Write-Progress -Activity '等待任务保存与导出完成' -Completed
    }
}

if (-not (Test-SameBackend $taskBackend)) {
    Write-Host '后端已退出。'
    return
}
# /T also stops multiprocessing workers. The root PID and creation time were
# checked immediately above; never stop all python.exe processes by name.
Write-Host "正在停止 EELS-ZEST Parallel · PID $($taskBackend.ProcessId) · 端口 $Port"
$taskKill = Join-Path $env:SystemRoot 'System32\taskkill.exe'
& $taskKill /PID $taskBackend.ProcessId /T /F
$taskKillCode = $LASTEXITCODE
$taskExitDeadline = [DateTime]::UtcNow.AddSeconds(5)
do {
    $taskRemaining = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskBackend.ProcessId)"
    if ($null -eq $taskRemaining -or $taskRemaining.CreationDate -ne $taskBackend.CreationDate) {
        Write-Host "并行版后端已停止。已有数据与结果文件保留，可重新运行 start_web.ps1。"
        exit 0
    }
    Start-Sleep -Milliseconds 200
} while ([DateTime]::UtcNow -lt $taskExitDeadline)
throw "停止后端失败（taskkill 返回 $taskKillCode），请检查当前账户是否有权管理该进程。"
