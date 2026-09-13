# Recoverable form of MathWorks' documented manual Service Host reinstall.
$ErrorActionPreference = 'Stop'
$repairRoot = 'D:\C_Migrated\Users\12105\AppData\Local\MathWorks'
$repairNames = @('ServiceHost', 'MATLABConnector')
$repairSuffix = '.a2-backup-20260913-evening'
$activeMatlab = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^(MATLAB|MathWorksServiceHost|MATLABConnector)\.exe$' })
if ($activeMatlab.Count -gt 0) { throw 'MATLAB services are active; refusing to move their files.' }
$repairParent = Get-Item -LiteralPath $repairRoot -Force
if ($repairParent.LinkType) { throw 'Expected resolved native parent, not a junction.' }
# Validate every target and destination before making any move.
foreach ($repairName in $repairNames) {
    $repairSource = Get-Item -LiteralPath (Join-Path $repairRoot $repairName) -Force
    $repairDestination = Join-Path $repairRoot ($repairName + $repairSuffix)
    if ($repairSource.Parent.FullName -ne $repairParent.FullName -or $repairSource.LinkType) {
        throw 'Target is outside the expected parent or is a reparse point.'
    }
    if (Test-Path -LiteralPath $repairDestination) { throw 'Backup exists; do not repeat this repair.' }
}
foreach ($repairName in $repairNames) {
    Rename-Item -LiteralPath (Join-Path $repairRoot $repairName) -NewName ($repairName + $repairSuffix) -Force
    Write-Output ('Preserved backup: ' + (Join-Path $repairRoot ($repairName + $repairSuffix)))
}
Write-Output 'No licenses or credentials were changed. Next MATLAB launch will reinstall Service Host.'
