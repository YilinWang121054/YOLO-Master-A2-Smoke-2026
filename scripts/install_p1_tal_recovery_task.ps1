param(
    [string]$TaskName = "YOLO-Master-A2-P1-TAL-Resume"
)

$ErrorActionPreference = "Stop"
$PythonExe = "F:\conda-envs\yolo-master\python.exe"
$ResumeScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "resume_p1_training.py")).Path
$ConfigPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\configs\p1-tal-s20260824.resume.json")).Path

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$Arguments = '"{0}" --config "{1}"' -f $ResumeScript, $ConfigPath
$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument $Arguments -WorkingDirectory (Split-Path $ResumeScript)
$Principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) -LogonType Interactive -RunLevel Limited
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name)
$Trigger.Delay = "PT1M"
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
