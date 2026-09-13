$ErrorActionPreference = 'Stop'
$taskName = 'YOLO-Master-A2-P1-V2-Fixed-Resume'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { throw 'Task already exists; inspect before changing it.' }
$pythonPath = 'F:\conda-envs\yolo-master\python.exe'
$scriptPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'resume_p1_training.py')).Path
$configPath = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../configs/p1-v2-fixed-s20260825.resume.json')).Path
$taskArgs = '-X utf8 "{0}" --config "{1}"' -f $scriptPath, $configPath
$taskAction = New-ScheduledTaskAction -Execute $pythonPath -Argument $taskArgs -WorkingDirectory $PSScriptRoot
$taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$taskPrincipal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$taskTrigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser
$taskTrigger.Delay = 'PT1M'
$taskSettings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName $taskName -Action $taskAction -Trigger $taskTrigger -Principal $taskPrincipal -Settings $taskSettings | Out-Null
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName,State
