"""On-demand host inventory. No background sampler and no persisted machine spec."""

import json
import os
import platform
import subprocess


def cpu_info():
    info = dict(cpu_name=platform.processor() or platform.machine() or "CPU",
                logical_cpus=os.cpu_count() or 1, physical_cores=None,
                memory_bytes=None, gpu_names=[], detection_warning=None)
    if os.name == "nt":
        # CIM queries run only at service startup or an explicit refresh. Keep
        # a bounded, hidden process and UTF-8 output, including non-English OSes.
        command = """
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$ErrorActionPreference = 'Stop'
$processors = @(Get-CimInstance Win32_Processor)
$machine = Get-CimInstance Win32_ComputerSystem
$graphics = @(Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name })
@{cpu_name = ($processors.Name -join ' / ')
  physical_cores = ($processors | Measure-Object NumberOfCores -Sum).Sum
  memory_bytes = $machine.TotalPhysicalMemory
  gpu_names = $graphics} | ConvertTo-Json -Compress
"""
        try:
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
                capture_output=True, check=True, timeout=10, encoding="utf-8-sig",
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            detected = json.loads(result.stdout)
            info.update(cpu_name=detected["cpu_name"] or info["cpu_name"],
                        physical_cores=int(detected["physical_cores"]),
                        memory_bytes=int(detected["memory_bytes"]),
                        gpu_names=detected.get("gpu_names") or [])
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
            info["detection_warning"] = f"CPU / 内存详细信息检测未完成：{type(exc).__name__}"
    else:
        try:
            info["memory_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (AttributeError, OSError, ValueError):
            pass
    return info
