$ErrorActionPreference = "Stop"

$workspace = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$colosseum = Join-Path $workspace "external\Colosseum"
$pluginDeps = Join-Path $workspace "unreal\DroneRSI\Plugins\AirSim\Source\AirLib\deps"
$buildRoot = Join-Path $workspace "build\airsim_linux_deps"

$toolchain = "C:\UnrealToolchains\v22_clang-16.0.6-centos7\x86_64-unknown-linux-gnu"
$clang = Join-Path $toolchain "bin\clang++.exe"
$ar = Join-Path $toolchain "bin\llvm-ar.exe"
$sysroot = $toolchain.Replace("\", "/")
$ueLibCxx = "C:/Program Files/Epic Games/UE_5.3/Engine/Source/ThirdParty/Unix/LibCxx"

if (-not (Test-Path -LiteralPath $clang)) { throw "Missing clang: $clang" }
if (-not (Test-Path -LiteralPath $ar)) { throw "Missing llvm-ar: $ar" }
if (-not (Test-Path -LiteralPath $colosseum)) { throw "Missing Colosseum checkout: $colosseum" }

function New-CleanDirectory([string] $path) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

function Get-ObjectPath([string] $base, [string] $source, [string] $objectRoot) {
    $relative = $source.Substring($base.Length).TrimStart("\", "/")
    $name = ($relative -replace '[\\/:*?"<>| ]', "_") -replace '\.cc$|\.cpp$', ".o"
    return Join-Path $objectRoot $name
}

function Compile-StaticLibrary(
    [string] $name,
    [string] $base,
    [string[]] $sources,
    [string[]] $includeDirs,
    [string[]] $defines,
    [string] $outputArchive
) {
    $objectRoot = Join-Path $buildRoot $name
    New-CleanDirectory $objectRoot

    $commonArgs = @(
        "--driver-mode=g++",
        "-target", "x86_64-unknown-linux-gnu",
        "--sysroot=$sysroot",
        "-nostdinc++",
        "-isystem", "$ueLibCxx/include",
        "-isystem", "$ueLibCxx/include/c++/v1",
        "-std=c++17",
        "-O3",
        "-fPIC",
        "-pthread",
        "-fexceptions",
        "-Wno-deprecated-declarations",
        "-Wno-undefined-var-template"
    )

    foreach ($define in $defines) {
        $commonArgs += "-D$define"
    }
    foreach ($include in $includeDirs) {
        $commonArgs += "-I$include"
    }

    $objects = New-Object System.Collections.Generic.List[string]
    foreach ($source in $sources) {
        $object = Get-ObjectPath $base $source $objectRoot
        Write-Host "Compiling ${name}: $source"
        & $clang @commonArgs -c $source -o $object
        if ($LASTEXITCODE -ne 0) {
            throw "Compile failed for $source"
        }
        $objects.Add($object)
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outputArchive) | Out-Null
    if (Test-Path -LiteralPath $outputArchive) {
        Remove-Item -LiteralPath $outputArchive -Force
    }

    Write-Host "Archiving $outputArchive"
    & $ar rcs $outputArchive @objects
    if ($LASTEXITCODE -ne 0) {
        throw "Archive failed: $outputArchive"
    }
}

New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null

$rpcRoot = Join-Path $colosseum "external\rpclib\rpclib-2.3.0"
$rpcSources = @(
    "lib\rpc\dispatcher.cc",
    "lib\rpc\server.cc",
    "lib\rpc\client.cc",
    "lib\rpc\this_handler.cc",
    "lib\rpc\this_session.cc",
    "lib\rpc\this_server.cc",
    "lib\rpc\rpc_error.cc",
    "lib\rpc\detail\server_session.cc",
    "lib\rpc\detail\response.cc",
    "lib\rpc\detail\client_error.cc",
    "lib\rpc\nonstd\optional.cc",
    "dependencies\src\format.cc",
    "dependencies\src\posix.cc"
) | ForEach-Object { Join-Path $rpcRoot $_ }

Compile-StaticLibrary `
    -name "rpclib" `
    -base $rpcRoot `
    -sources $rpcSources `
    -includeDirs @(
        (Join-Path $rpcRoot "include"),
        (Join-Path $rpcRoot "dependencies\include")
    ) `
    -defines @(
        "MSGPACK_PP_VARIADICS_MSVC=0",
        "ASIO_STANDALONE",
        "RPCLIB_ASIO=clmdep_asio",
        "RPCLIB_FMT=clmdep_fmt",
        "RPCLIB_MSGPACK=clmdep_msgpack",
        "RPCLIB_LINUX"
    ) `
    -outputArchive (Join-Path $pluginDeps "rpclib\lib\librpc.a")

$mavRoot = Join-Path $colosseum "MavLinkCom"
$mavSources = @(
    "common_utils\FileSystem.cpp",
    "common_utils\ThreadUtils.cpp",
    "src\AdHocConnection.cpp",
    "src\MavLinkConnection.cpp",
    "src\MavLinkFtpClient.cpp",
    "src\MavLinkLog.cpp",
    "src\MavLinkMessageBase.cpp",
    "src\MavLinkMessages.cpp",
    "src\MavLinkNode.cpp",
    "src\MavLinkTcpServer.cpp",
    "src\MavLinkVehicle.cpp",
    "src\MavLinkVideoStream.cpp",
    "src\Semaphore.cpp",
    "src\UdpSocket.cpp",
    "src\impl\AdHocConnectionImpl.cpp",
    "src\impl\MavLinkConnectionImpl.cpp",
    "src\impl\MavLinkFtpClientImpl.cpp",
    "src\impl\MavLinkNodeImpl.cpp",
    "src\impl\MavLinkTcpServerImpl.cpp",
    "src\impl\MavLinkVehicleImpl.cpp",
    "src\impl\MavLinkVideoStreamImpl.cpp",
    "src\impl\UdpSocketImpl.cpp",
    "src\serial_com\SerialPort.cpp",
    "src\serial_com\TcpClientPort.cpp",
    "src\serial_com\UdpClientPort.cpp",
    "src\serial_com\SocketInit.cpp",
    "src\serial_com\wifi.cpp",
    "src\impl\linux\MavLinkFindSerialPorts.cpp"
) | ForEach-Object { Join-Path $mavRoot $_ }

Compile-StaticLibrary `
    -name "MavLinkCom" `
    -base $mavRoot `
    -sources $mavSources `
    -includeDirs @(
        $mavRoot,
        (Join-Path $mavRoot "common_utils"),
        (Join-Path $mavRoot "include")
    ) `
    -defines @() `
    -outputArchive (Join-Path $pluginDeps "MavLinkCom\lib\libMavLinkCom.a")

Write-Host "Built AirSim Linux dependencies:"
Get-Item (Join-Path $pluginDeps "rpclib\lib\librpc.a"), (Join-Path $pluginDeps "MavLinkCom\lib\libMavLinkCom.a") |
    Select-Object FullName, Length, LastWriteTime |
    Format-Table -AutoSize
