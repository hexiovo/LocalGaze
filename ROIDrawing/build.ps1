param([string]$Destination = '')
$ErrorActionPreference = 'Stop'
$project = $PSScriptRoot
$projectRoot = Split-Path -Parent $project
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $projectRoot 'build-artifacts\windows-2026.10.31.0\roidrawing'
}
$python = Join-Path $project '.build-venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    py -3.12 -m venv (Join-Path $project '.build-venv')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create Python 3.12 build environment' }
}
& $python -m pip install -r (Join-Path $project 'requirements-build.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $python (Join-Path $project 'build_exe.py')
if ($LASTEXITCODE -ne 0) { throw 'EXE build failed' }
New-Item -ItemType Directory -Path $Destination -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $project 'dist\ROIDrawing.exe') -Destination (Join-Path $Destination 'ROIDrawing.exe')
Get-FileHash -LiteralPath (Join-Path $Destination 'ROIDrawing.exe') -Algorithm SHA256
