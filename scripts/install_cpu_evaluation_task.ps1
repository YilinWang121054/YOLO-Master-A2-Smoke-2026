param([string]$TaskName = "YOLO-Master-A2-CPU-Evaluation")
$ErrorActionPreference = "Stop"
$PythonExe = "F:\conda-envs\yolo-master\python.exe"
$ChainScript = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "chain_cpu_evaluation.py")).Path
$User = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Action = New-ScheduledTaskAction -Execute $PythonExe -Argument ('"{0}"' -f $ChainScript) -WorkingDirectory $PSScriptRoot
$Principal = New-ScheduledTaskPrincipal -UserId $User -LogonType Interactive -RunLevel Limited
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$Trigger.Delay = "PT2M"
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Days 7)
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
