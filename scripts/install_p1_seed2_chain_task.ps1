param(
    [string]$TaskName = "YOLO-Master-A2-P1-Seed2-Chain"
)

$ErrorActionPreference = "Stop"
$PythonExe = "F:\conda-envs\yolo-master\python.exe"
$ChainScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "chain_p1_seed2.py")).Path
$Arguments = '"{0}" --poll-seconds 60' -f $ChainScript
$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument $Arguments -WorkingDirectory (Split-Path $ChainScript)
$Principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name)
$Trigger.Delay = "PT3M"
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
