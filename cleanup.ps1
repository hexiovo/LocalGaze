$ErrorActionPreference = 'Continue'
$projectRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path.TrimEnd('\')
$projectPrefix = $projectRoot + '\'
$targets = @(
    '__pycache__',
    '.pytest_cache',
    'deployment\installation-check.json',
    'deployment\installation-check.stderr.log',
    'deployment\installation-check.stdout.log',
    'deployment\pip-install.log',
    'deployment\requirements-installed.txt',
    'deployment\source-installation-check.json',
    'downloads\pip-cache',
    'downloads\delivery-manifest.json',
    'downloads\LocalGaze-2026.10.16.0.zip',
    'downloads\LocalGaze-2026.10.16.0',
    'portable\LocalGaze-2026.10.16.0',
    'deliverables\localgaze-study-config.zip',
    'deliverables\localgaze-study-config-paste.md',
    'ROIDrawing\__pycache__',
    'ROIDrawing\build',
    'ROIDrawing\dist',
    'ROIDrawing\.build-venv',
    'docs\LocalGaze产品介绍与使用手册.pdf',
    'assets\guide\01-workbench.png',
    'assets\guide\02-analysis-selection.png',
    'assets\guide\03-analysis-parameters.png',
    'assets\guide\04-known-target.png',
    'assets\guide\05-environment-check.png',
    'Eye_Tracking_Preprocessing_Pipeline.md'
)
$failed = [System.Collections.Generic.List[string]]::new()

Write-Host "清理范围：$projectRoot"
Write-Host '仅处理脚本列出的项目文件。请先关闭 LocalGaze、ROI 工具和编辑器。'
foreach ($relative in $targets) {
    $target = [IO.Path]::GetFullPath((Join-Path $projectRoot $relative))
    if (-not $target.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝处理项目目录以外的路径：$relative"
    }
    $current = $target
    while ($current.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase) -or
           $current.Equals($projectRoot, [StringComparison]::OrdinalIgnoreCase)) {
        if (Test-Path -LiteralPath $current -ErrorAction Stop) {
            $component = Get-Item -LiteralPath $current -Force -ErrorAction Stop
            if ($component.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "拒绝经过符号链接或联接目录：$relative（$current）"
            }
        }
        if ($current.Equals($projectRoot, [StringComparison]::OrdinalIgnoreCase)) { break }
        $parent = [IO.Path]::GetDirectoryName($current)
        if ([string]::IsNullOrWhiteSpace($parent) -or
            $parent.Equals($current, [StringComparison]::OrdinalIgnoreCase)) {
            throw "无法安全检查路径组件：$relative（$current）"
        }
        $current = $parent
    }
    if (-not (Test-Path -LiteralPath $target)) { continue }
    $resolved = (Resolve-Path -LiteralPath $target).Path
    if (-not $resolved.StartsWith($projectPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝处理解析后位于项目目录以外的路径：$relative"
    }
    $item = Get-Item -LiteralPath $resolved -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "拒绝处理符号链接或联接目录：$relative"
    }
    if ($item.PSIsContainer) {
        $nestedReparsePoint = Get-ChildItem -LiteralPath $resolved -Force -Recurse -ErrorAction Stop |
            Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint } |
            Select-Object -First 1
        if ($nestedReparsePoint) { throw "拒绝处理包含符号链接或联接目录的目录：$relative" }
        $bytes = (Get-ChildItem -LiteralPath $resolved -File -Force -Recurse | Measure-Object -Property Length -Sum).Sum
        if ($null -eq $bytes) { $bytes = 0 }
        Write-Host ("将删除目录 {0}（{1} 字节）" -f $relative, $bytes)
        try { Remove-Item -LiteralPath $resolved -Recurse -Force -ErrorAction Stop }
        catch { Write-Warning ("无法删除 {0}：{1}" -f $relative, $_.Exception.Message); $failed.Add($relative) }
    }
    else {
        Write-Host ("将删除文件 {0}（{1} 字节）" -f $relative, $item.Length)
        try { Remove-Item -LiteralPath $resolved -Force -ErrorAction Stop }
        catch { Write-Warning ("无法删除 {0}：{1}" -f $relative, $_.Exception.Message); $failed.Add($relative) }
    }
}
if ($failed.Count -gt 0) {
    Write-Host ('未删除：' + ($failed -join ', '))
    exit 1
}
Write-Host '清理完成。'
