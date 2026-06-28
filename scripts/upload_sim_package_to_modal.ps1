param(
  [Parameter(Mandatory=$true)]
  [string]$PackageDir,

  [string]$VolumeName = "drone-rsi-sim-package",
  [string]$RemotePath = "/package",
  [switch]$Force
)

$resolved = Resolve-Path -LiteralPath $PackageDir -ErrorAction Stop
$putArgs = @("volume", "put")
if ($Force) {
  $putArgs += "--force"
}
$putArgs += @($VolumeName, $resolved.Path, $RemotePath)

python -m modal @putArgs
