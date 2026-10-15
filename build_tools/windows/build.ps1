param(
    [string]$PythonExe = '',
    [string]$ArtifactsRoot = '',
    [string]$ReleaseName = 'LocalGaze-2026.10.15.0',
    [string]$DeliveryRoot = '',
    [switch]$SkipArchive
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
if ([string]::IsNullOrWhiteSpace($ArtifactsRoot)) {
    $ArtifactsRoot = Join-Path $projectRoot 'build-artifacts\windows-2026.10.15.0'
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe. Run launch_localgaze.bat first, then install requirements-build.txt in .venv."
}
$PythonExe = (Resolve-Path -LiteralPath $PythonExe).Path
$pythonVersionText = (& $PythonExe -c "import sys; print('.'.join(str(part) for part in sys.version_info[:3]))").Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($pythonVersionText)) {
    throw 'Unable to determine the Python version used for the portable build.'
}
if ($pythonVersionText -eq '3.12.0') {
    throw 'CPython 3.12.0 is not supported for portable builds because it can produce invalid frozen SciPy code. Use Python 3.12.14 or a newer Python 3.12 maintenance release.'
}
$baseArtifactDir = [IO.Path]::GetFullPath($ArtifactsRoot)
$artifactParent = Split-Path -Parent $baseArtifactDir
$artifactLeaf = Split-Path -Leaf $baseArtifactDir
$attempt = 0
while ($true) {
    if ($attempt -eq 0) {
        $artifactDir = $baseArtifactDir
        $resolvedReleaseName = $ReleaseName
    }
    else {
        $artifactDir = Join-Path $artifactParent ($artifactLeaf + "-r$attempt")
        $resolvedReleaseName = "$ReleaseName-r$attempt"
    }
    if ($DeliveryRoot) {
        $releaseRoot = Join-Path ([IO.Path]::GetFullPath($DeliveryRoot)) $resolvedReleaseName
    }
    else {
        $releaseRoot = Join-Path $projectRoot ('release\' + $resolvedReleaseName)
    }
    $zipPath = "$releaseRoot.zip"
    if (-not (Test-Path -LiteralPath $artifactDir) -and
        -not (Test-Path -LiteralPath $releaseRoot) -and
        -not (Test-Path -LiteralPath $zipPath)) {
        break
    }
    $attempt += 1
}
New-Item -ItemType Directory -Path $artifactDir | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonDir = Split-Path -Parent ([IO.Path]::GetFullPath($PythonExe))
$pythonBaseDir = (& $PythonExe -c "import sys; print(sys.base_prefix)").Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($pythonBaseDir)) { throw 'Unable to determine the Python base directory.' }
$env:PATH = "$pythonDir;$pythonDir\Library\bin;$pythonDir\DLLs;$pythonBaseDir;$pythonBaseDir\Library\bin;$pythonBaseDir\DLLs;$pythonDir\Scripts;" + $env:PATH
& $PythonExe -B (Join-Path $PSScriptRoot 'collect_licenses.py') --project $projectRoot --output (Join-Path $projectRoot 'third_party') *> (Join-Path $artifactDir 'license-collection.log')
if ($LASTEXITCODE -ne 0) { throw "Third-party license collection failed; see $artifactDir\license-collection.log" }
& $PythonExe -B -m PyInstaller --noconfirm `
    --distpath (Join-Path $artifactDir 'dist') `
    --workpath (Join-Path $artifactDir 'work') `
    (Join-Path $PSScriptRoot 'LocalGaze.spec') *> (Join-Path $artifactDir 'build.log')
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed; see $artifactDir\build.log" }
$builtDir = Join-Path $artifactDir 'dist\LocalGaze'
$reportPath = Join-Path $artifactDir 'installation-check.json'
& (Join-Path $builtDir 'LocalGazeWorker.exe') --check-installation $reportPath `
    *> (Join-Path $artifactDir 'installation-check.log')
if ($LASTEXITCODE -ne 0) { throw "Bundle checks failed; see $reportPath" }
$check = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($check.status -ne 'passed') { throw 'Bundle checks did not pass.' }
# Windows MediaPipe native loading must also work from a Unicode installation.
$unicodeCheckDir = Join-Path $artifactDir '中文路径检查'
New-Item -ItemType Directory -Path $unicodeCheckDir | Out-Null
Copy-Item -LiteralPath $builtDir -Destination $unicodeCheckDir -Recurse
$unicodeExe = Join-Path $unicodeCheckDir 'LocalGaze\LocalGaze.exe'
$unicodeReport = Join-Path $artifactDir 'unicode-windowed-check.json'
$windowedProcess = Start-Process -FilePath $unicodeExe -ArgumentList @('--check-installation', ('"' + $unicodeReport + '"')) -WindowStyle Hidden -PassThru -Wait
if ($windowedProcess.ExitCode -ne 0) { throw "Unicode windowed bundle check failed: $unicodeReport" }
$unicodeResult = Get-Content -LiteralPath $unicodeReport -Raw -Encoding UTF8 | ConvertFrom-Json
if ($unicodeResult.status -ne 'passed') { throw 'Unicode windowed check did not pass.' }
New-Item -ItemType Directory -Path $releaseRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $builtDir 'LocalGaze.exe'), (Join-Path $builtDir 'LocalGazeWorker.exe'), (Join-Path $builtDir 'ROIDrawing.exe') -Destination $releaseRoot
Copy-Item -LiteralPath (Join-Path $builtDir '_internal') -Destination $releaseRoot -Recurse
Copy-Item -LiteralPath (Join-Path $projectRoot 'assets\guide') -Destination $releaseRoot -Recurse
Copy-Item -LiteralPath (Join-Path $projectRoot 'third_party') -Destination $releaseRoot -Recurse
New-Item -ItemType Directory -Path (Join-Path $releaseRoot 'docs') | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot 'docs\LocalGaze产品介绍与使用手册.pdf') -Destination (Join-Path $releaseRoot 'docs')
Copy-Item -LiteralPath (Join-Path $projectRoot 'README.md'), (Join-Path $projectRoot 'version.md') -Destination $releaseRoot
Copy-Item -LiteralPath (Join-Path $projectRoot 'THIRD_PARTY_NOTICES.md') -Destination $releaseRoot
$sourceInventoryPath = Join-Path $artifactDir 'source-inventory.json'
& $PythonExe -B (Join-Path $PSScriptRoot 'release_manifest.py') $projectRoot $releaseRoot $reportPath $sourceInventoryPath
if ($LASTEXITCODE -ne 0) { throw 'Release manifest generation failed.' }
$verificationPath = Join-Path $artifactDir 'delivery-verification.json'
& $PythonExe -B (Join-Path $PSScriptRoot 'verify_app_delivery.py') $projectRoot $releaseRoot (Join-Path $releaseRoot 'LocalGaze.exe') $verificationPath $sourceInventoryPath `
    *> (Join-Path $artifactDir 'delivery-verification.log')
if ($LASTEXITCODE -ne 0) { throw "Portable delivery verification failed: $verificationPath" }
if (-not $SkipArchive) {
    Compress-Archive -LiteralPath $releaseRoot -DestinationPath $zipPath
    $deliveryManifest = Join-Path $artifactDir 'delivery-manifest.json'
    & $PythonExe -B (Join-Path $PSScriptRoot 'delivery_manifest.py') $releaseRoot $zipPath $deliveryManifest
    if ($LASTEXITCODE -ne 0) { throw 'External delivery manifest generation failed.' }
    Write-Output "Archive: $zipPath"
    Write-Output "Delivery manifest: $deliveryManifest"
}
Write-Output "Release: $releaseRoot"
