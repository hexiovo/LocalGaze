param(
    [Parameter(Mandatory=$true)][string]$PythonExe,
    [string]$ArtifactsRoot = 'E:\CodexData\localgaze-builds\20261001-v2',
    [string]$ReleaseName = 'LocalGaze-20261001-v2'
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
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
    $releaseRoot = Join-Path $projectRoot ('release\' + $resolvedReleaseName)
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
$env:PATH = "$pythonDir;$pythonDir\Library\bin;$pythonDir\DLLs;$pythonDir\Scripts;" + $env:PATH
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
Copy-Item -LiteralPath (Join-Path $PSScriptRoot '使用说明.txt') -Destination $releaseRoot
Copy-Item -LiteralPath (Join-Path $projectRoot 'LocalGaze使用指导语及流程.txt') -Destination $releaseRoot
Copy-Item -LiteralPath (Join-Path $projectRoot 'README.md'), (Join-Path $projectRoot 'version.md') -Destination $releaseRoot
& $PythonExe -B (Join-Path $PSScriptRoot 'release_manifest.py') $projectRoot $releaseRoot $reportPath
if ($LASTEXITCODE -ne 0) { throw 'Release manifest generation failed.' }
Compress-Archive -LiteralPath $releaseRoot -DestinationPath $zipPath
$deliveryManifest = Join-Path $artifactDir 'delivery-manifest.json'
& $PythonExe -B (Join-Path $PSScriptRoot 'delivery_manifest.py') $releaseRoot $zipPath $deliveryManifest
if ($LASTEXITCODE -ne 0) { throw 'External delivery manifest generation failed.' }
Write-Output "Release: $releaseRoot"
Write-Output "Archive: $zipPath"
Write-Output "Delivery manifest: $deliveryManifest"
