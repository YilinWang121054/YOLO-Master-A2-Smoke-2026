param(
    [string]$TaskName = "YOLO-Master-A2-P1-Seed3-Chain"
)

$ErrorActionPreference = "Stop"
$PythonExe = "F:\conda-envs\yolo-master\python.exe"
$ChainScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "chain_p1_seed3.py")).Path

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$Arguments = '"{0}" --poll-seconds 60' -f $ChainScript
$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument $Arguments -WorkingDirectory (Split-Path $ChainScript)
$Principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name)
$Trigger.Delay = "PT1M"
# A three-mode 120-epoch chain can exceed 72 hours on a 6 GB GPU. Keep the
# logon recovery supervisor alive for a full week so a reboot does not turn
# into an automatic timeout before the final mode starts.
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Days 7) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
