$ErrorActionPreference = 'Stop'
$a2Shell = (Get-Command powershell.exe -ErrorAction Stop).Source
$a2Script = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'start_p1_screen_worker.ps1')).Path
$a2User = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
foreach ($a2Worker in @('gpu', 'cpu')) {
    $a2Task = 'YOLO-Master-A2-Screen-V2-' + $a2Worker.ToUpper()
    if (Get-ScheduledTask -TaskName $a2Task -ErrorAction SilentlyContinue) { throw "Task already exists: $a2Task" }
}
foreach ($a2Worker in @('gpu', 'cpu')) {
    $a2Task = 'YOLO-Master-A2-Screen-V2-' + $a2Worker.ToUpper()
    $a2Args = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -File "{0}" -Worker {1}' -f $a2Script, $a2Worker
    $a2Action = New-ScheduledTaskAction -Execute $a2Shell -Argument $a2Args -WorkingDirectory $PSScriptRoot
    $a2Principal = New-ScheduledTaskPrincipal -UserId $a2User -LogonType Interactive -RunLevel Limited
    $a2Trigger = New-ScheduledTaskTrigger -AtLogOn -User $a2User
    $a2Trigger.Delay = 'PT1M'
    $a2Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Days 7) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $a2Task -Action $a2Action -Trigger $a2Trigger -Principal $a2Principal -Settings $a2Settings | Out-Null
    Get-ScheduledTask -TaskName $a2Task | Select-Object TaskName, State
}
