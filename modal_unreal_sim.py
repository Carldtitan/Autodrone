import json
import math
import os
import uuid
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import modal


APP_NAME = "drone-rsi-sim"
SECRET_NAME = os.getenv("MODAL_SECRET_NAME", "drone-rsi-secrets")
REGION = "us"
SIM_GPU = "L40S"
SIM_VOLUME_NAME = "drone-rsi-sim-package"
NVIDIA_VULKAN_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"
LAVAPIPE_VULKAN_ICD = "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json"
os.environ.setdefault("VK_ICD_FILENAMES", LAVAPIPE_VULKAN_ICD)

app = modal.App(APP_NAME)
secrets = [modal.Secret.from_name(SECRET_NAME)]
sim_volume = modal.Volume.from_name(SIM_VOLUME_NAME, create_if_missing=True)

sim_image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.10")
    .apt_install(
        "bash",
        "ca-certificates",
        "curl",
        "libasound2",
        "libatk1.0-0",
        "libcairo2",
        "libcups2",
        "libdbus-1-3",
        "libdrm2",
        "libegl1",
        "libgbm1",
        "libgl1",
        "libglib2.0-0",
        "libnss3",
        "libpulse0",
        "libvulkan1",
        "libx11-6",
        "libxcb1",
        "libxcomposite1",
        "libxcursor1",
        "libxdamage1",
        "libxext6",
        "libxfixes3",
        "libxi6",
        "libxinerama1",
        "libxkbcommon0",
        "libxrandr2",
        "libxrender1",
        "imagemagick",
        "mesa-vulkan-drivers",
        "procps",
        "vulkan-tools",
        "x11-apps",
        "xvfb",
    )
    .pip_install(
        "fastapi[standard]==0.118.0",
        "msgpack-rpc-python==0.4.1",
        "numpy<2",
        "pymongo==4.17.0",
        "setuptools",
        "wheel",
    )
    .run_commands("useradd -m -u 1000 unreal")
    .run_commands(
        "mkdir -p /usr/share/vulkan/icd.d && "
        "echo '{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"libGLX_nvidia.so.0\",\"api_version\":\"1.3.239\"}}' "
        "> /usr/share/vulkan/icd.d/nvidia_icd.json"
    )
    .run_commands("python -m pip install --no-build-isolation airsim==1.8.1 && python -m pip install 'numpy<2'")
)

_unreal_proc: subprocess.Popen | None = None
_xvfb_proc: subprocess.Popen | None = None
_unreal_log_tail: list[str] = []
_unreal_log_lock = threading.Lock()
_mission_lock = threading.Lock()
_mission_state: dict[str, Any] = {
    "flight_id": None,
    "status": "idle",
    "request": None,
    "goal": None,
    "resolved_goal": None,
    "route": [],
    "samples": [],
    "started_at": None,
    "completed_at": None,
    "error": None,
}


LANDMARKS = {
    "ferry": {"name": "Ferry Building", "lat": 37.79549, "lon": -122.39374},
    "bay bridge": {"name": "Bay Bridge", "lat": 37.79836, "lon": -122.37780},
    "bridge": {"name": "Bay Bridge", "lat": 37.79836, "lon": -122.37780},
    "salesforce": {"name": "Salesforce Tower", "lat": 37.78973, "lon": -122.39608},
    "coit": {"name": "Coit Tower", "lat": 37.80239, "lon": -122.40582},
    "moscone": {"name": "Moscone Center", "lat": 37.78417, "lon": -122.40156},
    "union": {"name": "Union Square", "lat": 37.78799, "lon": -122.40744},
    "market": {"name": "Market Street", "lat": 37.77670, "lon": -122.41620},
    "street": {"name": "Market Street", "lat": 37.77670, "lon": -122.41620},
    "golden gate": {"name": "Golden Gate Bridge", "lat": 37.81993, "lon": -122.47826},
    "pier": {"name": "Pier 39", "lat": 37.80867, "lon": -122.40982},
    "downtown": {"name": "Downtown San Francisco", "lat": 37.7897, "lon": -122.4011},
}

SPATIAL_OBSTACLES = [
    {"id": "salesforce-tower", "name": "Salesforce Tower", "lat": 37.78973, "lon": -122.39608, "radius_m": 85, "height_m": 335},
    {"id": "transamerica-pyramid", "name": "Transamerica Pyramid", "lat": 37.79516, "lon": -122.40279, "radius_m": 70, "height_m": 260},
    {"id": "555-california", "name": "555 California", "lat": 37.79210, "lon": -122.40372, "radius_m": 70, "height_m": 240},
    {"id": "ferry-building", "name": "Ferry Building", "lat": 37.79549, "lon": -122.39374, "radius_m": 55, "height_m": 75},
    {"id": "coit-tower", "name": "Coit Tower", "lat": 37.80239, "lon": -122.40582, "radius_m": 65, "height_m": 145},
    {"id": "bay-bridge-west-tower", "name": "Bay Bridge West Tower", "lat": 37.79875, "lon": -122.38690, "radius_m": 95, "height_m": 175},
    {"id": "bay-bridge-east-span", "name": "Bay Bridge East Span", "lat": 37.80120, "lon": -122.37160, "radius_m": 125, "height_m": 180},
]

SAFETY_CONFIG = {
    "min_clearance_m": 10.0,
    "min_altitude_m": 0.8,
    "cruise_altitude_m": 1.83,
    "max_demo_altitude_m": 1.83,
    "control_step_m": 0.75,
    "control_tick_s": 0.6,
    "max_speed_mps": 20.12,
}


def _append_unreal_log(line: str) -> None:
    with _unreal_log_lock:
        _unreal_log_tail.append(line.rstrip())
        del _unreal_log_tail[:-200]


def _read_unreal_stdout(proc: subprocess.Popen) -> None:
    if not proc.stdout:
        return
    try:
        for line in proc.stdout:
            _append_unreal_log(line)
    except Exception as exc:
        _append_unreal_log(f"log reader failed: {type(exc).__name__}: {exc}")


def _unreal_logs() -> list[str]:
    with _unreal_log_lock:
        return list(_unreal_log_tail)


def _package_dir() -> Path:
    return Path(os.getenv("SIM_PACKAGE_DIR", "/sim/package"))


def _start_script() -> Path:
    configured = os.getenv("SIM_START_SCRIPT")
    if configured:
        return Path(configured)
    return _package_dir() / "DroneRSI.sh"


def _unreal_args() -> list[str]:
    configured = os.getenv("UNREAL_ARGS")
    if configured:
        return configured.split()
    args = [
        "-AllowRunningAsRoot",
        "-AllowCPUDevices",
        "-nosound",
        "-unattended",
        "-NoSplash",
        "-windowed",
        "-ResX=1280",
        "-ResY=720",
    ]
    if os.getenv("USE_XVFB", "1") != "1":
        args.insert(0, "-RenderOffscreen")
    return args


def _airsim_settings_text() -> str:
    return json.dumps(
        {
            "SettingsVersion": 1.2,
            "SimMode": "Multirotor",
            "ClockSpeed": 1.0,
            "ViewMode": os.getenv("AIRSIM_VIEW_MODE", "FlyWithMe"),
            "CameraDefaults": {
                "CaptureSettings": [
                    {
                        "ImageType": 0,
                        "Width": int(os.getenv("AIRSIM_CAPTURE_WIDTH", "320")),
                        "Height": int(os.getenv("AIRSIM_CAPTURE_HEIGHT", "180")),
                        "FOV_Degrees": 90,
                        "MotionBlurAmount": 0,
                    }
                ]
            },
        },
        separators=(",", ":"),
    )


def _find_launch_script() -> Path | None:
    script = _start_script()
    if script.exists():
        return script
    candidates = sorted(_package_dir().glob("*.sh"))
    return candidates[0] if candidates else None


def _start_xvfb() -> None:
    global _xvfb_proc
    if os.getenv("USE_XVFB", "1") != "1":
        return
    if _xvfb_proc and _xvfb_proc.poll() is None:
        return
    _xvfb_proc = subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1280x720x24", "-nolisten", "tcp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(0.5)
    _append_unreal_log("started Xvfb on DISPLAY=:99")


def _start_unreal() -> dict[str, Any]:
    global _unreal_proc
    if _unreal_proc and _unreal_proc.poll() is None:
        return {"started": True, "pid": _unreal_proc.pid, "already_running": True}

    script = _find_launch_script()
    if not script:
        return {
            "started": False,
            "reason": "missing_linux_package",
            "expected": str(_start_script()),
            "volume": SIM_VOLUME_NAME,
        }

    run_user = os.getenv("SIM_RUN_USER", "unreal")
    _start_xvfb()
    script.chmod(script.stat().st_mode | 0o111)
    binary = script.parent / "Blocks" / "Binaries" / "Linux" / "Blocks"
    if binary.exists():
        binary.chmod(binary.stat().st_mode | 0o111)
    airsim_settings_dir = Path(f"/home/{run_user}/Documents/AirSim")
    airsim_settings_dir.mkdir(parents=True, exist_ok=True)
    settings_file = airsim_settings_dir / "settings.json"
    if not settings_file.exists():
        settings_file.write_text(_airsim_settings_text(), encoding="utf-8")
    launch_settings_file = script.parent / "settings.json"
    launch_settings_file.write_text(_airsim_settings_text(), encoding="utf-8")
    for writable in (script.parent / "Blocks" / "Saved", script.parent / "Engine" / "Saved"):
        writable.mkdir(parents=True, exist_ok=True)
        writable.chmod(0o777)
    subprocess.run(["chown", "-R", f"{run_user}:{run_user}", f"/home/{run_user}"], check=False)
    subprocess.run(["chown", "-R", f"{run_user}:{run_user}", str(script.parent)], check=False)

    cmd = [str(script), *_unreal_args(), f"-settings={launch_settings_file}"]
    env = os.environ.copy()
    env.setdefault("HOME", f"/home/{run_user}")
    env.setdefault("VK_ICD_FILENAMES", LAVAPIPE_VULKAN_ICD)
    if os.getenv("USE_XVFB", "1") == "1":
        env["DISPLAY"] = ":99"
    else:
        env.pop("SDL_VIDEODRIVER", None)
        env.pop("DISPLAY", None)
    _append_unreal_log("starting: " + " ".join(cmd))
    _unreal_proc = subprocess.Popen(
        cmd,
        cwd=str(script.parent),
        env=env,
        user=run_user,
        group=run_user,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    threading.Thread(target=_read_unreal_stdout, args=(_unreal_proc,), daemon=True).start()
    return {"started": True, "pid": _unreal_proc.pid, "cmd": cmd}


def _airsim_ping() -> dict[str, Any]:
    try:
        import airsim

        client = airsim.MultirotorClient(
            ip=os.getenv("AIRSIM_INTERNAL_HOST", "127.0.0.1"),
            port=int(os.getenv("AIRSIM_INTERNAL_PORT", "41451")),
            timeout_value=5,
        )
        client.confirmConnection()
        return {"ok": True, "error": None}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _airsim_client(timeout_value: int = 10):
    import airsim

    return airsim.MultirotorClient(
        ip=os.getenv("AIRSIM_INTERNAL_HOST", "127.0.0.1"),
        port=int(os.getenv("AIRSIM_INTERNAL_PORT", "41451")),
        timeout_value=timeout_value,
    )


def _sf_origin() -> tuple[float, float, float]:
    return (
        float(os.getenv("SF_ORIGIN_LAT", "37.7749")),
        float(os.getenv("SF_ORIGIN_LON", "-122.4194")),
        float(os.getenv("SF_ORIGIN_ALT", "20")),
    )


def _sim_to_city_scale() -> float:
    return max(1.0, float(os.getenv("SF_METERS_PER_SIM_METER", "22")))


def _mission_altitude_m() -> float:
    return float(os.getenv("MISSION_ALTITUDE_METERS", str(SAFETY_CONFIG["cruise_altitude_m"])))


def _estimate_agl_m(position: Any) -> tuple[float, str]:
    raw_agl_m = -float(position.z_val)
    if raw_agl_m > 0:
        return min(SAFETY_CONFIG["max_demo_altitude_m"], raw_agl_m), "airsim_ned"
    return min(SAFETY_CONFIG["max_demo_altitude_m"], _mission_altitude_m()), "mission_command_cesium_frame"


def _state_to_geo(position: Any) -> dict[str, float]:
    origin_lat, origin_lon, origin_alt = _sf_origin()
    scale = _sim_to_city_scale()
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = 111_320.0 * max(0.1, math.cos(math.radians(origin_lat)))
    lat = origin_lat + ((float(position.x_val) * scale) / meters_per_degree_lat)
    lon = origin_lon + ((float(position.y_val) * scale) / meters_per_degree_lon)
    agl_m, altitude_source = _estimate_agl_m(position)
    alt = origin_alt + agl_m
    return {
        "lat": lat,
        "lon": lon,
        "alt": alt,
        "agl_m": agl_m,
        "altitude_source": altitude_source,
        "raw_airsim_z_m": float(position.z_val),
    }


def _geo_to_sim(lat: float, lon: float) -> tuple[float, float]:
    origin_lat, origin_lon, _ = _sf_origin()
    scale = _sim_to_city_scale()
    meters_per_degree_lat = 111_320.0
    meters_per_degree_lon = 111_320.0 * max(0.1, math.cos(math.radians(origin_lat)))
    x = ((lat - origin_lat) * meters_per_degree_lat) / scale
    y = ((lon - origin_lon) * meters_per_degree_lon) / scale
    return x, y


def _resolve_landmark(request: str, goal: str | None = None) -> dict[str, Any]:
    text = f"{request} {goal or ''}".lower()
    for key, landmark in LANDMARKS.items():
        if key in text:
            return landmark
    return LANDMARKS["ferry"]


def _obstacle_with_sim_coords(obstacle: dict[str, Any]) -> dict[str, Any]:
    x, y = _geo_to_sim(float(obstacle["lat"]), float(obstacle["lon"]))
    return {**obstacle, "x": x, "y": y, "radius_sim": float(obstacle["radius_m"]) / _sim_to_city_scale()}


def _distance_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple[float, float, float]:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-6:
        return math.hypot(px - ax, py - ay), ax, ay
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    cx = ax + abx * t
    cy = ay + aby * t
    return math.hypot(px - cx, py - cy), cx, cy


def _avoid_obstacles(route: list[dict[str, float]]) -> list[dict[str, float]]:
    if not route:
        return route
    margin_sim = SAFETY_CONFIG["min_clearance_m"] / _sim_to_city_scale()
    adjusted: list[dict[str, float]] = []
    prev = {"x": 0.0, "y": 0.0, "z": -SAFETY_CONFIG["cruise_altitude_m"], "label": "home"}
    for waypoint in route:
        detours: list[dict[str, float]] = []
        for obstacle in (_obstacle_with_sim_coords(item) for item in SPATIAL_OBSTACLES):
            clearance, closest_x, closest_y = _distance_point_to_segment(
                float(obstacle["x"]),
                float(obstacle["y"]),
                float(prev["x"]),
                float(prev["y"]),
                float(waypoint["x"]),
                float(waypoint["y"]),
            )
            required = float(obstacle["radius_sim"]) + margin_sim
            if clearance < required:
                seg_x = float(waypoint["x"]) - float(prev["x"])
                seg_y = float(waypoint["y"]) - float(prev["y"])
                seg_len = max(1e-6, math.hypot(seg_x, seg_y))
                side = 1.0 if (seg_x * (float(obstacle["y"]) - float(prev["y"])) - seg_y * (float(obstacle["x"]) - float(prev["x"]))) >= 0 else -1.0
                perp_x = -seg_y / seg_len * side
                perp_y = seg_x / seg_len * side
                avoid_dist = required + 14.0
                detour_alt = max(SAFETY_CONFIG["cruise_altitude_m"], float(obstacle["height_m"]) + SAFETY_CONFIG["min_clearance_m"])
                detours.append(
                    {
                        "x": float(obstacle["x"]) + perp_x * avoid_dist,
                        "y": float(obstacle["y"]) + perp_y * avoid_dist,
                        "z": -min(SAFETY_CONFIG["max_demo_altitude_m"], detour_alt),
                        "label": f"avoid {obstacle['name']}",
                        "avoidance_for": obstacle["id"],
                    }
                )
        detours.sort(key=lambda item: math.hypot(float(item["x"]) - float(prev["x"]), float(item["y"]) - float(prev["y"])))
        adjusted.extend(detours)
        adjusted.append(waypoint)
        prev = waypoint
    return adjusted


def _constraint_status(position: Any) -> dict[str, Any]:
    city_x = float(position.x_val) * _sim_to_city_scale()
    city_y = float(position.y_val) * _sim_to_city_scale()
    altitude_m, _ = _estimate_agl_m(position)
    nearest = None
    violations: list[dict[str, Any]] = []
    for obstacle in SPATIAL_OBSTACLES:
        ox, oy = _geo_to_sim(float(obstacle["lat"]), float(obstacle["lon"]))
        ox *= _sim_to_city_scale()
        oy *= _sim_to_city_scale()
        horizontal_clearance = math.hypot(city_x - ox, city_y - oy) - float(obstacle["radius_m"])
        vertical_clearance = altitude_m - float(obstacle["height_m"])
        clearance = horizontal_clearance
        candidate = {
            "id": obstacle["id"],
            "name": obstacle["name"],
            "horizontal_clearance_m": round(horizontal_clearance, 2),
            "vertical_clearance_m": round(vertical_clearance, 2),
            "required_clearance_m": SAFETY_CONFIG["min_clearance_m"],
        }
        if nearest is None or horizontal_clearance < nearest["horizontal_clearance_m"]:
            nearest = candidate
        if horizontal_clearance < SAFETY_CONFIG["min_clearance_m"] and vertical_clearance < SAFETY_CONFIG["min_clearance_m"]:
            violations.append(candidate)
    altitude_violation = altitude_m < SAFETY_CONFIG["min_altitude_m"]
    return {
        "nearest_obstacle": nearest,
        "violations": violations,
        "altitude_violation": altitude_violation,
        "min_clearance_m": SAFETY_CONFIG["min_clearance_m"],
    }


def _wants_return_home(request: str) -> bool:
    text = f" {request.lower()} "
    return any(phrase in text for phrase in (" come back", " return home", " return to home", " go back home", " back to home"))


def _build_route(request: str, goal: str | None = None) -> tuple[dict[str, Any], list[dict[str, float]]]:
    landmark = _resolve_landmark(request, goal)
    target_x, target_y = _geo_to_sim(landmark["lat"], landmark["lon"])
    cruise_z = -_mission_altitude_m()
    return_home = _wants_return_home(request)
    # Keep prompted flights readable: one resolved destination, plus an optional
    # explicit return-home leg. Detours are inserted only when the straight line
    # intersects a known obstacle volume.
    route = [{"x": target_x, "y": target_y, "z": cruise_z, "label": landmark["name"]}]
    if return_home:
        route.append({"x": 0.0, "y": 0.0, "z": cruise_z, "label": "home"})
    return landmark, _avoid_obstacles(route)


def _mission_snapshot() -> dict[str, Any]:
    with _mission_lock:
        return json.loads(json.dumps(_mission_state))


def _mission_status() -> str:
    with _mission_lock:
        return str(_mission_state.get("status") or "idle")


def _update_mission(**values: Any) -> dict[str, Any]:
    with _mission_lock:
        _mission_state.update(values)
        return json.loads(json.dumps(_mission_state))


def _record_sample(position: Any, action: dict[str, Any] | None = None) -> None:
    constraints = _constraint_status(position)
    sample = {
        "position": {"x": position.x_val, "y": position.y_val, "z": position.z_val},
        "geo": _state_to_geo(position),
        "constraints": constraints,
        "rung3_action": action,
        "time": time.time(),
    }
    with _mission_lock:
        samples = _mission_state.setdefault("samples", [])
        samples.append(sample)
        del samples[:-500]
        _mission_state["last_constraints"] = constraints
        if action:
            _mission_state["last_rung3_action"] = action


def _rung3_action_for_target(position: Any, target: dict[str, float], previous_distance: float | None) -> tuple[dict[str, Any], float]:
    dx = float(target["x"]) - float(position.x_val)
    dy = float(target["y"]) - float(position.y_val)
    distance = math.hypot(dx, dy)
    desired_yaw = math.atan2(dy, dx) if distance > 0.01 else 0.0
    progress_error = 0.0 if previous_distance is None else max(-10.0, min(10.0, previous_distance - distance))
    pitch = max(-0.16, min(0.16, dx * 0.004 + progress_error * 0.002))
    roll = max(-0.16, min(0.16, -dy * 0.004))
    if abs(dx) < 1.5:
        pitch = 0.0
    if abs(dy) < 1.5:
        roll = 0.0
    target_z = float(target["z"])
    vertical_error = abs(target_z) - max(0.0, -float(position.z_val))
    estimated_thrust_n = max(1.0, 14.71 - vertical_error * 0.05)
    base_pwm = max(0.48, min(0.68, 0.56 + estimated_thrust_n * 0.003))
    motor_pwms = [
        max(0.35, min(0.85, base_pwm + pitch * 0.35 + roll * 0.18)),
        max(0.35, min(0.85, base_pwm + pitch * 0.35 - roll * 0.18)),
        max(0.35, min(0.85, base_pwm - pitch * 0.22 + roll * 0.18)),
        max(0.35, min(0.85, base_pwm - pitch * 0.22 - roll * 0.18)),
    ]
    action = {
        "control_rung": 3,
        "api": "rung3_force_policy",
        "execution_api": "moveByVelocityZAsync AirSim physics velocity control",
        "roll_rad": round(roll, 4),
        "pitch_rad": round(pitch, 4),
        "yaw_rad": round(desired_yaw, 4),
        "target_altitude_agl_m": round(abs(target_z), 2),
        "target_z_ned_m": round(target_z, 2),
        "duration_s": SAFETY_CONFIG["control_tick_s"],
        "speed_limit_mps": SAFETY_CONFIG["max_speed_mps"],
        "estimated_total_thrust_n": round(estimated_thrust_n, 2),
        "motor_pwm_estimate": [round(value, 3) for value in motor_pwms],
        "distance_to_waypoint_m": round(distance * _sim_to_city_scale(), 2),
    }
    return action, distance


def _fly_rung3_segment(client: Any, target: dict[str, float]) -> str:
    import airsim

    timeout_at = time.time() + float(os.getenv("MISSION_SEGMENT_TIMEOUT_SECONDS", "300"))
    target_x = float(target["x"])
    target_y = float(target["y"])
    target_z = float(target["z"])
    previous_distance: float | None = None
    speed_mps = float(os.getenv("MISSION_SPEED_MPS", str(SAFETY_CONFIG["max_speed_mps"])))
    speed_sim_mps = max(0.01, speed_mps / _sim_to_city_scale())
    close_enough = max(0.03, float(os.getenv("MISSION_WAYPOINT_TOLERANCE_METERS", "2.0")) / _sim_to_city_scale())
    stagnant_ticks = 0
    while time.time() < timeout_at:
        if _mission_status() in {"paused", "stopped"}:
            client.hoverAsync().join()
            return _mission_status()
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        action, distance = _rung3_action_for_target(pos, target, previous_distance)
        _record_sample(pos, action)
        dx = target_x - float(pos.x_val)
        dy = target_y - float(pos.y_val)
        if distance < close_enough:
            return "reached"
        last_distance = previous_distance
        previous_distance = distance
        vx = (dx / max(distance, 0.001)) * speed_sim_mps
        vy = (dy / max(distance, 0.001)) * speed_sim_mps
        projected = type(
            "P",
            (),
            {
                "x_val": float(pos.x_val) + vx * action["duration_s"],
                "y_val": float(pos.y_val) + vy * action["duration_s"],
                "z_val": target_z,
            },
        )()
        projected_constraints = _constraint_status(projected)
        if projected_constraints["violations"]:
            _update_mission(
                status="obstacle_hold",
                error=f"Known obstacle volume ahead near {projected_constraints['violations'][0]['name']}; holding instead of passing through.",
            )
            _record_sample(pos, action)
            return "blocked"
        client.moveByVelocityZAsync(
            vx,
            vy,
            target_z,
            action["duration_s"],
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, math.degrees(action["yaw_rad"])),
        ).join()
        collision = client.simGetCollisionInfo()
        if getattr(collision, "has_collided", False):
            _update_mission(
                status="collision_hold",
                error=f"Collision detected; stopped instead of passing through geometry: {getattr(collision, 'object_name', 'unknown')}",
            )
            _record_sample(client.getMultirotorState().kinematics_estimated.position, action)
            return "collision"
        if last_distance is not None and distance >= last_distance - 0.001:
            stagnant_ticks += 1
            if stagnant_ticks >= 4:
                _update_mission(status="holding_near_target", error="No progress toward waypoint; holding instead of oscillating.")
                return "stalled"
        else:
            stagnant_ticks = 0
    _update_mission(status="timed_out_before_waypoint", error=f"Timed out before {target.get('label', 'waypoint')}; holding position at realistic speed.")
    return "timeout"


def _reset_vehicle_to_low_home(client: Any) -> None:
    import airsim

    cruise_altitude = _mission_altitude_m()
    pose = airsim.Pose(
        airsim.Vector3r(0.0, 0.0, -cruise_altitude),
        airsim.to_quaternion(0.0, 0.0, 0.0),
    )
    client.simSetVehiclePose(pose, False)
    client.hoverAsync().join()
    time.sleep(0.4)
    _record_sample(client.getMultirotorState().kinematics_estimated.position)


def _run_flight_worker(flight_id: str, request: str, goal: str, route: list[dict[str, float]]) -> None:
    try:
        import airsim

        _update_mission(status="connecting", error=None)
        client = _airsim_client(timeout_value=20)
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)

        _update_mission(status="low_home_reset")
        _reset_vehicle_to_low_home(client)

        for index, waypoint in enumerate(route):
            _update_mission(status="flying", active_waypoint=index + 1, active_label=waypoint.get("label"))
            result = _fly_rung3_segment(client, waypoint)
            if result != "reached":
                client.hoverAsync().join()
                state = client.getMultirotorState()
                _record_sample(state.kinematics_estimated.position)
                _update_mission(completed_at=time.time(), active_waypoint=index + 1)
                return

        client.hoverAsync().join()
        state = client.getMultirotorState()
        _record_sample(state.kinematics_estimated.position)
        _update_mission(status="complete", completed_at=time.time(), active_waypoint=len(route))
    except Exception as exc:
        _update_mission(status="error", error=f"{type(exc).__name__}: {exc}", completed_at=time.time())


def _scene_png() -> bytes:
    import airsim

    client = _airsim_client(timeout_value=int(os.getenv("AIRSIM_IMAGE_TIMEOUT_SECONDS", "60")))
    png = client.simGetImage(os.getenv("AIRSIM_CAMERA_NAME", "0"), airsim.ImageType.Scene)
    if not png:
        raise RuntimeError("AirSim returned no camera image.")
    if isinstance(png, str):
        return png.encode("latin1")
    return bytes(png)


def _screen_png() -> bytes:
    if os.getenv("USE_XVFB", "1") != "1":
        raise RuntimeError("Xvfb screen capture is disabled.")
    env = os.environ.copy()
    env["DISPLAY"] = ":99"
    result = subprocess.run(
        ["sh", "-lc", "xwd -root -silent -display :99 | convert xwd:- png:-"],
        env=env,
        capture_output=True,
        timeout=int(os.getenv("XVFB_SCREENSHOT_TIMEOUT_SECONDS", "20")),
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"screen capture failed: {stderr}")
    if not result.stdout:
        raise RuntimeError("screen capture returned no bytes.")
    return result.stdout


@app.function(
    image=sim_image,
    gpu=SIM_GPU,
    secrets=secrets,
    volumes={"/sim": sim_volume},
    min_containers=1,
    max_containers=1,
    scaledown_window=900,
    timeout=24 * 60 * 60,
    startup_timeout=10 * 60,
    region=REGION,
)
@modal.asgi_app()
def sim_api():
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response, StreamingResponse
    from pydantic import BaseModel

    api = FastAPI(title="Drone RSI Modal Unreal Sim")

    class MissionRequest(BaseModel):
        request: str = "Fly to the Ferry Building and return safely"
        goal: str | None = None

    class StartPositionRequest(BaseModel):
        lat: float
        lon: float
        altitude_m: float = 1.83
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    launch = _start_unreal()

    @api.get("/")
    def root():
        return {
            "service": APP_NAME,
            "gpu": SIM_GPU,
            "package_dir": str(_package_dir()),
            "launch": launch,
            "airsim": _airsim_ping() if launch.get("started") else {"ok": False, "error": launch.get("reason")},
        }

    @api.get("/api/sim/status")
    def status():
        running = bool(_unreal_proc and _unreal_proc.poll() is None)
        current_launch = launch if running else _start_unreal()
        running = bool(_unreal_proc and _unreal_proc.poll() is None)
        return {
            "service": APP_NAME,
            "gpu": SIM_GPU,
            "package_dir": str(_package_dir()),
            "start_script": str(_start_script()),
            "package_present": _find_launch_script() is not None,
            "unreal_running": running,
            "pid": _unreal_proc.pid if running and _unreal_proc else None,
            "returncode": _unreal_proc.poll() if _unreal_proc else None,
            "launch": current_launch,
            "log_tail": _unreal_logs()[-40:],
            "airsim": _airsim_ping() if running else {"ok": False, "error": "unreal_not_running"},
            "mission": _mission_snapshot(),
            "constraints": {"obstacles": SPATIAL_OBSTACLES, "safety": SAFETY_CONFIG},
            "time": time.time(),
        }

    @api.get("/api/sim/logs")
    def logs():
        return {
            "unreal_running": bool(_unreal_proc and _unreal_proc.poll() is None),
            "returncode": _unreal_proc.poll() if _unreal_proc else None,
            "lines": _unreal_logs(),
        }

    @api.get("/api/sim/diagnostics")
    def diagnostics():
        def run(cmd: list[str], env_overrides: dict[str, str] | None = None) -> dict[str, Any]:
            try:
                run_env = os.environ.copy()
                if env_overrides:
                    run_env.update(env_overrides)
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=20, env=run_env)
                return {
                    "cmd": cmd,
                    "env_overrides": env_overrides or {},
                    "returncode": result.returncode,
                    "stdout": result.stdout[-4000:],
                    "stderr": result.stderr[-4000:],
                }
            except Exception as exc:
                return {"cmd": cmd, "error": f"{type(exc).__name__}: {exc}"}

        return {
            "env": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "NVIDIA_VISIBLE_DEVICES",
                    "NVIDIA_DRIVER_CAPABILITIES",
                    "VK_ICD_FILENAMES",
                    "LD_LIBRARY_PATH",
                )
            },
            "icd_files": [str(path) for path in Path("/usr/share/vulkan/icd.d").glob("*")],
            "nvidia_icd": Path(NVIDIA_VULKAN_ICD).read_text(errors="ignore")
            if Path(NVIDIA_VULKAN_ICD).exists()
            else None,
            "nvidia_proc": Path("/proc/driver/nvidia/version").read_text(errors="ignore")
            if Path("/proc/driver/nvidia/version").exists()
            else None,
            "nvidia_smi": run(["nvidia-smi"]),
            "vulkaninfo": run(["vulkaninfo", "--summary"]),
            "vulkaninfo_lavapipe": run(["vulkaninfo", "--summary"], {"VK_ICD_FILENAMES": LAVAPIPE_VULKAN_ICD}),
            "vulkaninfo_nvidia": run(["vulkaninfo", "--summary"], {"VK_ICD_FILENAMES": NVIDIA_VULKAN_ICD}),
            "nvidia_libs": run(
                [
                    "sh",
                    "-lc",
                    "ldconfig -p | grep -E 'libGLX_nvidia|libEGL_nvidia|libnvidia-vulkan|libvulkan' || true; "
                    "find /usr /lib -name 'libGLX_nvidia*' -o -name 'libEGL_nvidia*' -o -name 'libnvidia*.so*' 2>/dev/null | head -120",
                ]
            ),
        }

    @api.get("/api/sim/constraints")
    def constraints():
        return {"obstacles": SPATIAL_OBSTACLES, "safety": SAFETY_CONFIG}

    @api.post("/api/sim/takeoff-test")
    def takeoff_test():
        ping = _airsim_ping()
        if not ping["ok"]:
            raise HTTPException(status_code=409, detail=ping)

        client = _airsim_client(timeout_value=10)
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync(timeout_sec=10).join()
        state = client.getMultirotorState()
        return {
            "ok": True,
            "position": {
                "x": state.kinematics_estimated.position.x_val,
                "y": state.kinematics_estimated.position.y_val,
                "z": state.kinematics_estimated.position.z_val,
            },
        }

    @api.post("/api/sim/fly-mission")
    def fly_mission(req: MissionRequest):
        if not req.request.strip():
            raise HTTPException(status_code=400, detail="request is required")
        ping = _airsim_ping()
        if not ping["ok"]:
            raise HTTPException(status_code=409, detail=ping)

        current = _mission_snapshot()
        if current.get("status") in {"connecting", "takeoff", "flying"}:
            return {"ok": True, "already_running": True, "mission": current}

        flight_id = uuid.uuid4().hex[:12]
        resolved_goal, route = _build_route(req.request, req.goal)
        route_public = [
            {
                **waypoint,
                "geo": _state_to_geo(type("P", (), {"x_val": waypoint["x"], "y_val": waypoint["y"], "z_val": waypoint["z"]})()),
            }
            for waypoint in route
        ]
        _update_mission(
            flight_id=flight_id,
            status="queued",
            request=req.request,
            goal=resolved_goal["name"],
            resolved_goal=resolved_goal,
            route=route_public,
            samples=[],
            started_at=time.time(),
            completed_at=None,
            error=None,
            active_waypoint=0,
            active_label=None,
        )
        threading.Thread(target=_run_flight_worker, args=(flight_id, req.request, resolved_goal["name"], route), daemon=True).start()
        return {"ok": True, "mission": _mission_snapshot()}

    @api.post("/api/sim/stop")
    def stop():
        ping = _airsim_ping()
        if not ping["ok"]:
            raise HTTPException(status_code=409, detail=ping)
        client = _airsim_client(timeout_value=10)
        client.enableApiControl(True)
        client.hoverAsync().join()
        state = client.getMultirotorState()
        _record_sample(state.kinematics_estimated.position)
        _update_mission(status="stopped", completed_at=time.time(), error=None)
        return {"ok": True, "mission": _mission_snapshot()}

    @api.post("/api/sim/pause")
    def pause():
        ping = _airsim_ping()
        if not ping["ok"]:
            raise HTTPException(status_code=409, detail=ping)
        client = _airsim_client(timeout_value=10)
        client.enableApiControl(True)
        client.hoverAsync().join()
        state = client.getMultirotorState()
        _record_sample(state.kinematics_estimated.position)
        _update_mission(status="paused", completed_at=time.time(), error=None)
        return {"ok": True, "mission": _mission_snapshot()}

    @api.post("/api/sim/set-start")
    def set_start(req: StartPositionRequest):
        import airsim

        ping = _airsim_ping()
        if not ping["ok"]:
            raise HTTPException(status_code=409, detail=ping)
        client = _airsim_client(timeout_value=10)
        client.enableApiControl(True)
        client.armDisarm(True)
        x, y = _geo_to_sim(req.lat, req.lon)
        altitude = max(0.5, min(30.0, float(req.altitude_m)))
        pose = airsim.Pose(airsim.Vector3r(x, y, -altitude), airsim.to_quaternion(0.0, 0.0, 0.0))
        client.simSetVehiclePose(pose, False)
        client.hoverAsync().join()
        state = client.getMultirotorState()
        _record_sample(state.kinematics_estimated.position)
        _update_mission(
            status="stopped",
            request="Manual start position",
            goal="manual",
            resolved_goal={"name": "Manual start", "lat": req.lat, "lon": req.lon},
            route=[],
            samples=_mission_snapshot().get("samples", []),
            completed_at=time.time(),
            error=None,
        )
        return {"ok": True, "mission": _mission_snapshot(), "geo": _state_to_geo(state.kinematics_estimated.position)}

    @api.post("/api/sim/return-home")
    def return_home():
        ping = _airsim_ping()
        if not ping["ok"]:
            raise HTTPException(status_code=409, detail=ping)
        request = "Return home and hover"
        goal = "home"
        route = [{"x": 0.0, "y": 0.0, "z": -_mission_altitude_m(), "label": "home"}]
        flight_id = uuid.uuid4().hex[:12]
        _update_mission(
            flight_id=flight_id,
            status="queued",
            request=request,
            goal=goal,
            resolved_goal={"name": "Home", "lat": _sf_origin()[0], "lon": _sf_origin()[1]},
            route=[
                {
                    **route[0],
                    "geo": _state_to_geo(type("P", (), {"x_val": 0.0, "y_val": 0.0, "z_val": route[0]["z"]})()),
                }
            ],
            samples=[],
            started_at=time.time(),
            completed_at=None,
            error=None,
            active_waypoint=0,
            active_label=None,
        )
        threading.Thread(target=_run_flight_worker, args=(flight_id, request, goal, route), daemon=True).start()
        return {"ok": True, "mission": _mission_snapshot()}

    @api.get("/api/sim/telemetry")
    def telemetry():
        ping = _airsim_ping()
        if not ping["ok"]:
            raise HTTPException(status_code=409, detail=ping)

        client = _airsim_client(timeout_value=10)
        state = client.getMultirotorState()
        position = state.kinematics_estimated.position
        velocity = state.kinematics_estimated.linear_velocity
        geo = _state_to_geo(position)
        try:
            gps = client.getGpsData()
            gps_point = gps.gnss.geo_point
            origin_lat = float(os.getenv("SF_ORIGIN_LAT", "37.7955"))
            origin_lon = float(os.getenv("SF_ORIGIN_LON", "-122.3937"))
            use_gps = os.getenv("USE_AIRSIM_GPS", "0") == "1"
            near_origin = abs(gps_point.latitude - origin_lat) < 0.5 and abs(gps_point.longitude - origin_lon) < 0.5
            if abs(gps_point.latitude) > 0.001 and abs(gps_point.longitude) > 0.001 and (use_gps or near_origin):
                geo = {
                    "lat": gps_point.latitude,
                    "lon": gps_point.longitude,
                    "alt": gps_point.altitude,
                }
        except Exception:
            pass

        mission = _mission_snapshot()
        if mission.get("status") in {"queued", "connecting", "takeoff", "flying", "complete"}:
            _record_sample(position)
            mission = _mission_snapshot()

        return {
            "ok": True,
            "flying": int(state.landed_state) != 0,
            "landed_state": int(state.landed_state),
            "position": {
                "x": position.x_val,
                "y": position.y_val,
                "z": position.z_val,
            },
            "velocity": {
                "x": velocity.x_val,
                "y": velocity.y_val,
                "z": velocity.z_val,
            },
            "geo": geo,
            "mission": mission,
            "time": time.time(),
        }

    @api.get("/api/sim/image")
    def camera_image():
        ping = _airsim_ping()
        if not ping["ok"]:
            raise HTTPException(status_code=409, detail=ping)
        try:
            return Response(content=_scene_png(), media_type="image/png")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    @api.get("/api/sim/screen.png")
    def screen_image():
        running = bool(_unreal_proc and _unreal_proc.poll() is None)
        if not running:
            raise HTTPException(status_code=409, detail={"ok": False, "error": "unreal_not_running"})
        try:
            return Response(content=_screen_png(), media_type="image/png")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    @api.get("/api/sim/camera.mjpg")
    def camera_stream():
        ping = _airsim_ping()
        if not ping["ok"]:
            raise HTTPException(status_code=409, detail=ping)

        def frames():
            while True:
                try:
                    png = _scene_png()
                    yield b"--frame\r\nContent-Type: image/png\r\n\r\n" + png + b"\r\n"
                except Exception as exc:
                    yield (
                        b"--frame\r\nContent-Type: text/plain\r\n\r\n"
                        + f"{type(exc).__name__}: {exc}".encode("utf-8")
                        + b"\r\n"
                    )
                    break
                time.sleep(float(os.getenv("CAMERA_STREAM_INTERVAL_SECONDS", "0.2")))

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @api.get("/api/sim/screen.mjpg")
    def screen_stream():
        running = bool(_unreal_proc and _unreal_proc.poll() is None)
        if not running:
            raise HTTPException(status_code=409, detail={"ok": False, "error": "unreal_not_running"})

        def frames():
            while True:
                try:
                    png = _screen_png()
                    yield b"--frame\r\nContent-Type: image/png\r\n\r\n" + png + b"\r\n"
                except Exception as exc:
                    yield (
                        b"--frame\r\nContent-Type: text/plain\r\n\r\n"
                        + f"{type(exc).__name__}: {exc}".encode("utf-8")
                        + b"\r\n"
                    )
                    break
                time.sleep(float(os.getenv("SCREEN_STREAM_INTERVAL_SECONDS", "1.0")))

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    return api
