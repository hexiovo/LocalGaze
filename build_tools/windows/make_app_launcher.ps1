param(
    [Parameter(Mandatory=$true)][string]$BundlePath,
    [Parameter(Mandatory=$true)][string]$LauncherPath,
    [Parameter(Mandatory=$true)][string]$ArtifactsRoot
)
$ErrorActionPreference = 'Stop'
$bundle = [IO.Path]::GetFullPath($BundlePath)
$launcher = [IO.Path]::GetFullPath($LauncherPath)
if (-not (Test-Path -LiteralPath (Join-Path $bundle 'LocalGaze.exe'))) { throw 'Validated bundle is missing.' }
if (Test-Path -LiteralPath $launcher) { throw 'Launcher path already exists; do not overwrite it.' }
$launcherDirectory = Split-Path -Parent $launcher
if ([IO.Path]::GetPathRoot($bundle) -eq [IO.Path]::GetPathRoot($launcher)) {
    $launcherUri = [Uri]::new($launcherDirectory + [IO.Path]::DirectorySeparatorChar)
    $bundleUri = [Uri]::new($bundle)
    $bundleLocation = [Uri]::UnescapeDataString($launcherUri.MakeRelativeUri($bundleUri).ToString()).Replace('/', '\')
}
else { $bundleLocation = $bundle }
$bundleLiteral = $bundleLocation.Replace('\', '\\').Replace('"', '\"')
$source = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'PortableLauncher.cs') -Raw -Encoding UTF8
$source = $source.Replace('__BUNDLE_DIRECTORY__', $bundleLiteral)
$generatedSource = Join-Path $ArtifactsRoot 'portable-launcher-generated.cs'
[IO.File]::WriteAllText($generatedSource, $source, [Text.UTF8Encoding]::new($true))
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$arguments = @('/nologo', '/target:winexe', '/platform:x64', '/codepage:65001',
    '/reference:System.Windows.Forms.dll', ('/win32icon:' + (Join-Path $projectRoot 'assets\localgaze.ico')),
    ('/out:' + $launcher), $generatedSource)
& $compiler @arguments *> (Join-Path $ArtifactsRoot 'launcher-build.log')
if ($LASTEXITCODE -ne 0) { throw 'Launcher compilation failed; see launcher-build.log.' }
Write-Output "Launcher: $launcher"
