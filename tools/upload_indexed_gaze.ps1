<# Upload only the exact NPY/CSV names in a successful rebuild manifest.
DestinationRoot corresponds to the local data-root (/dvss).
Use -DryRun for robocopy /L (listing only).
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Manifest,
    [Parameter(Mandatory = $true)][string]$DestinationRoot,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$selection = Get-Content -LiteralPath $Manifest -Raw -Encoding UTF8 | ConvertFrom-Json
$sourceRoot = [IO.Path]::GetFullPath($selection.data_root).TrimEnd('\')
$destination = [IO.Path]::GetFullPath($DestinationRoot).TrimEnd('\')
if ($destination -eq $sourceRoot -or $destination.StartsWith($sourceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Destination must be outside the local data root.'
}
$entries = @($selection.files | ForEach-Object {
    $relative = [string]$_
    if ([IO.Path]::IsPathRooted($relative) -or $relative -match '(^|[/\\])\.\.([/\\]|$)' -or
        $relative.IndexOfAny([char[]]'*?:') -ge 0 -or $relative -notmatch '\.(npy|csv)$') {
        throw "Invalid manifest path: $relative"
    }
    $localPath = [IO.Path]::GetFullPath((Join-Path $sourceRoot $relative))
    if (-not $localPath.StartsWith($sourceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Path escapes source root: $relative"
    }
    if (-not (Test-Path -LiteralPath $localPath -PathType Leaf)) {
        throw "Missing source file: $localPath"
    }
    [PSCustomObject]@{
        Directory = [IO.Path]::GetDirectoryName($relative)
        Name = [IO.Path]::GetFileName($relative)
    }
})
if ($entries.Count -eq 0) { throw 'Manifest contains no files.' }
foreach ($group in ($entries | Group-Object Directory)) {
    $source = Join-Path $sourceRoot $group.Name
    $target = Join-Path $destination $group.Name
    # Chunk exact names to stay below Windows native command-line length limits.
    $names = @($group.Group.Name | Sort-Object -Unique)
    for ($offset = 0; $offset -lt $names.Count; $offset += 20) {
        $last = [Math]::Min($offset + 19, $names.Count - 1)
        $copyArguments = @($source, $target) + @($names[$offset..$last]) + @(
            '/Z', '/COPY:DAT', '/R:3', '/W:5', '/IS', '/IT', '/XX', '/NP'
        )
        if ($DryRun) { $copyArguments += '/L' }
        & robocopy @copyArguments
        if ($LASTEXITCODE -ge 8) { throw "Robocopy failed with exit code $LASTEXITCODE for $source" }
    }
}
Write-Host "Finished. Selected files: $($entries.Count). DryRun: $DryRun"
