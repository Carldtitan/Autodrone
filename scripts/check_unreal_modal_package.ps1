param(
  [Parameter(Mandatory=$true)]
  [string]$PackageDir
)

$resolved = Resolve-Path -LiteralPath $PackageDir -ErrorAction Stop
$scripts = Get-ChildItem -LiteralPath $resolved.Path -Filter "*.sh" -File -ErrorAction SilentlyContinue
$binaries = Get-ChildItem -LiteralPath $resolved.Path -Recurse -File -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match "(Linux|Shipping|Development)" -or $_.DirectoryName -match "Binaries" }

[pscustomobject]@{
  PackageDir = $resolved.Path
  HasShellLauncher = [bool]$scripts
  ShellLaunchers = $scripts.FullName
  CandidateBinaries = $binaries.FullName
  ModalUploadCommand = "scripts\\upload_sim_package_to_modal.ps1 -PackageDir `"$($resolved.Path)`" -Force"
} | ConvertTo-Json -Depth 4
