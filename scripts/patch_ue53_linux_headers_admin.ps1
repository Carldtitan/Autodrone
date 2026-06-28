$ErrorActionPreference = "Stop"

$edits = @(
    @{
        Path = "C:\Program Files\Epic Games\UE_5.3\Engine\Source\Runtime\Core\Public\Async\AsyncWork.h"
        From = "check(TimeLimitSeconds > 0.0f)`r`n`t`tFPlatformMisc::MemoryBarrier();"
        To = "check(TimeLimitSeconds > 0.0f);`r`n`t`tFPlatformMisc::MemoryBarrier();"
    },
    @{
        Path = "C:\Program Files\Epic Games\UE_5.3\Engine\Source\Runtime\Experimental\Chaos\Public\Chaos\ChaosArchive.h"
        From = "check(!Ar.IsLoading())`t//SerializationFactory must construct new object on load"
        To = "check(!Ar.IsLoading());`t//SerializationFactory must construct new object on load"
    }
)

$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

foreach ($edit in $edits) {
    if (-not (Test-Path -LiteralPath $edit.Path)) {
        throw "Missing Unreal header: $($edit.Path)"
    }

    $backup = "$($edit.Path).codex-bak"
    if (-not (Test-Path -LiteralPath $backup)) {
        Copy-Item -LiteralPath $edit.Path -Destination $backup
    }

    $item = Get-Item -LiteralPath $edit.Path
    if ($item.IsReadOnly) {
        $item.IsReadOnly = $false
    }

    $text = [System.IO.File]::ReadAllText($edit.Path)
    if ($text.Contains($edit.To)) {
        Write-Host "Already patched: $($edit.Path)"
        continue
    }
    if (-not $text.Contains($edit.From)) {
        throw "Expected text not found in $($edit.Path)"
    }

    $text = $text.Replace($edit.From, $edit.To)
    [System.IO.File]::WriteAllText($edit.Path, $text, $utf8NoBom)
    Write-Host "Patched: $($edit.Path)"
}

Write-Host "UE 5.3 Linux header patch complete."
