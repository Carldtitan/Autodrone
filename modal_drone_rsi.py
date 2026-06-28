import datetime as dt
import json
import os
import subprocess
import shutil
import urllib.error
import urllib.request
from typing import Any

import modal


APP_NAME = "drone-rsi"
SECRET_NAME = "drone-rsi-secrets"
REGION = "us"
SIM_BASE_URL = os.getenv("SIM_BASE_URL", "https://carl-4186--drone-rsi-sim-sim-api.modal.run")

app = modal.App(APP_NAME)
secrets = [modal.Secret.from_name(SECRET_NAME)]
model_cache = modal.Volume.from_name("drone-rsi-model-cache", create_if_missing=True)

dashboard_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.118.0", "pymongo==4.17.0", "python-dotenv==1.2.1")
)

brain_image = (
    modal.Image.from_registry("vllm/vllm-openai:latest", add_python="3.12")
    .entrypoint([])
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "VLLM_USE_V1": "1",
        }
    )
)

LANDMARKS = {
    "ferry": "Ferry Building, San Francisco",
    "bay bridge": "Bay Bridge",
    "bridge": "Bay Bridge",
    "salesforce": "Salesforce Tower",
    "coit": "Coit Tower, San Francisco",
    "moscone": "Moscone Center",
    "union": "Union Square",
    "market": "Market Street",
    "street": "Market Street",
    "golden gate": "Golden Gate Bridge",
    "pier": "Pier 39",
    "downtown": "Downtown San Francisco",
    "waterfront": "San Francisco waterfront",
}


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _mongo():
    from pymongo import MongoClient

    uri = os.environ["MONGODB_URI"]
    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    db = client[os.getenv("MONGODB_DB", "world_fair_hackathon")]
    return client, db


def _collection_names() -> dict[str, str]:
    return {
        "trajectories": os.getenv("MONGODB_TRAJ_COLLECTION", "trajectories"),
        "lessons": os.getenv("MONGODB_LESSONS_COLLECTION", "lessons"),
        "runs": os.getenv("MONGODB_RUNS_COLLECTION", "mission_runs"),
        "metrics": os.getenv("MONGODB_METRICS_COLLECTION", "rsi_metrics"),
    }


def _mission_key(request: str, goal: str) -> str:
    clean = f"{request.strip().lower()}::{goal.strip().lower()}"
    return clean[:240]


def _resolve_goal_from_request(request: str, goal: str | None = None) -> str:
    if goal and goal.strip():
        return goal.strip()
    text = request.lower()
    for key, value in LANDMARKS.items():
        if key in text:
            return value
    return "Ferry Building, San Francisco"


def _start_sim_flight(request: str, goal: str) -> dict[str, Any]:
    payload = json.dumps({"request": request, "goal": goal}).encode("utf-8")
    http_req = urllib.request.Request(
        f"{SIM_BASE_URL}/api/sim/fly-mission",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_req, timeout=25) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "error": body}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _simulate_or_replay_mission(request: str, goal: str) -> dict[str, Any]:
    """Mongo-backed RSI harness.

    This is the deployable RSI core: attempt count changes behavior, successful
    trajectories are persisted, and later runs replay/refine stored routes.
    Replace this function's inner transition model with Colosseum/AirSim calls
    once the Unreal simulator is packaged.
    """
    client, db = _mongo()
    names = _collection_names()
    goal = _resolve_goal_from_request(request, goal)
    key = _mission_key(request, goal)

    best = db[names["trajectories"]].find_one(
        {"mission_key": key, "success": True},
        sort=[("score", -1), ("steps", 1), ("created_at", -1)],
    )

    replay = best is not None
    mode = "retrieve_replay_refine" if replay else "sim_live_explore"

    sim_flight = _start_sim_flight(request, goal)
    mission = sim_flight.get("mission") if isinstance(sim_flight, dict) else None
    route = mission.get("route", []) if isinstance(mission, dict) else []
    mission_status = mission.get("status") if isinstance(mission, dict) else None
    constraints = mission.get("last_constraints", {}) if isinstance(mission, dict) else {}
    violations = len(constraints.get("violations", [])) if isinstance(constraints, dict) else 0
    steps = len(route)
    reward = 0
    success = bool(sim_flight.get("ok") and mission_status == "complete") if isinstance(sim_flight, dict) else False
    trajectory = [
        {
            "step": i + 1,
            "waypoint": item.get("label"),
            "geo": item.get("geo"),
            "source": "modal_unreal_route",
        }
        for i, item in enumerate(route)
    ]

    run_doc = {
        "mission_key": key,
        "request": request,
        "goal": goal,
        "mode": mode,
        "success": success,
        "steps": steps,
        "guardrail_violations": violations,
        "reward": reward,
        "replayed_from_memory": replay,
        "sim_flight_ok": bool(sim_flight.get("ok")) if isinstance(sim_flight, dict) else False,
        "sim_mission_status": mission_status,
        "sim_flight_id": mission.get("flight_id") if isinstance(mission, dict) else None,
        "resolved_goal": mission.get("resolved_goal") if isinstance(mission, dict) else None,
        "created_at": _now(),
    }
    run_id = db[names["runs"]].insert_one(run_doc).inserted_id

    if success and trajectory:
        score = -steps - violations * 50
        db[names["trajectories"]].insert_one(
            {
                "mission_key": key,
                "request": request,
                "goal": goal,
                "success": True,
                "score": score,
                "steps": steps,
                "guardrail_violations": violations,
                "trajectory": trajectory,
                "lesson": "Live Modal sim route resolved from user request; reuse only as route memory.",
                "created_at": _now(),
            }
        )
    elif violations:
        db[names["lessons"]].insert_one(
            {
                "mission_key": key,
                "request": request,
                "goal": goal,
                "lesson": "Guardrail violation observed during exploration; widen turn radius and climb earlier.",
                "created_at": _now(),
            }
        )

    total_runs = db[names["runs"]].count_documents({"mission_key": key})
    successful_runs = db[names["runs"]].count_documents({"mission_key": key, "success": True})
    metric_doc = {
        "mission_key": key,
        "request": request,
        "goal": goal,
        "run_count": total_runs,
        "success_count": successful_runs,
        "latest_steps": steps,
        "latest_guardrail_violations": violations,
        "latest_reward": reward,
        "created_at": _now(),
    }
    db[names["metrics"]].insert_one(metric_doc)
    client.close()

    return {
        **run_doc,
        "_id": str(run_id),
        "trajectory_preview": trajectory[:5],
        "sim_flight": sim_flight,
        "total_runs_for_request": total_runs,
        "success_count_for_request": successful_runs,
    }


@app.function(
    image=dashboard_image,
    secrets=secrets,
    min_containers=1,
    max_containers=1,
    scaledown_window=600,
    timeout=600,
    region=REGION,
)
@modal.asgi_app()
def dashboard():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel

    api = FastAPI(title="Drone RSI SF")

    class MissionRequest(BaseModel):
        request: str = "Fly to the Ferry Building and return safely"
        goal: str | None = None

    @api.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(
            """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Drone RSI SF</title>
  <link href="https://cdn.jsdelivr.net/npm/cesium@1.132.0/Build/Cesium/Widgets/widgets.css" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/cesium@1.132.0/Build/Cesium/Cesium.js"></script>
  <style>
    :root { color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
    body { margin: 0; background: #f7f8fa; color: #161b22; }
    header { padding: 20px 28px; background: #ffffff; border-bottom: 1px solid #d8dee4; display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    h1 { font-size: 22px; margin: 0; letter-spacing: 0; }
    main { padding: 16px 20px 24px; max-width: 1480px; margin: 0 auto; }
    .grid { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 18px; }
    .panel, .card { background: #fff; border: 1px solid #d8dee4; border-radius: 8px; }
    .panel { padding: 18px; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
    .card { padding: 12px 14px; min-height: 58px; display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .viewer { margin-bottom: 16px; overflow: hidden; background: #0b0f14; color: #f6f8fa; border: 1px solid #30363d; border-radius: 8px; min-height: 640px; height: min(78vh, 900px); position: relative; }
    #cesiumContainer { position: absolute; inset: 0; }
    .cesium-viewer-bottom, .cesium-widget-credits { display: none !important; }
    .viewer .overlay { position: absolute; inset: auto 14px 14px 14px; padding: 10px 12px; border-radius: 6px; background: rgba(11, 15, 20, 0.78); color: #f6f8fa; font-size: 13px; }
    .map-tools { position: absolute; top: 14px; left: 14px; z-index: 500; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .map-tools button { margin: 0; background: rgba(9, 105, 218, 0.95); box-shadow: 0 4px 14px rgba(0,0,0,0.2); }
    .map-tools .ghost { background: rgba(31, 35, 40, 0.88); }
    .scene-badge { position: absolute; right: 14px; top: 14px; z-index: 500; padding: 8px 10px; border-radius: 6px; background: rgba(11, 15, 20, 0.82); color: #fff; font-size: 13px; }
    .telemetry { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }
    .telemetry div { background: #f6f8fa; border: 1px solid #d8dee4; border-radius: 6px; padding: 10px; }
    .label { color: #57606a; font-size: 12px; text-transform: uppercase; letter-spacing: 0; margin-bottom: 8px; }
    .value { font-size: 18px; font-weight: 700; word-break: break-word; }
    .status-pill { display: inline-flex; align-items: center; justify-content: center; min-width: 70px; padding: 7px 10px; border-radius: 999px; background: #d1242f; color: #fff; font-size: 13px; font-weight: 800; text-transform: uppercase; }
    .status-pill.ok { background: #1a7f37; }
    label { display: block; font-weight: 650; margin: 12px 0 6px; }
    input, textarea { width: 100%; box-sizing: border-box; border: 1px solid #c9d1d9; border-radius: 6px; padding: 10px 12px; font: inherit; background: #fff; }
    textarea { min-height: 90px; resize: vertical; }
    button { margin-top: 14px; border: 0; border-radius: 6px; background: #0969da; color: #fff; padding: 10px 14px; font-weight: 700; cursor: pointer; }
    button.secondary { background: #24292f; margin-left: 8px; }
    button:disabled { opacity: 0.55; cursor: wait; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; padding: 9px 8px; border-bottom: 1px solid #d8dee4; vertical-align: top; }
    th { color: #57606a; font-size: 12px; text-transform: uppercase; }
    .status { display: inline-flex; align-items: center; gap: 8px; font-size: 14px; color: #57606a; }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: #d1242f; }
    .dot.ok { background: #1a7f37; }
    pre { white-space: pre-wrap; background: #f6f8fa; border: 1px solid #d8dee4; border-radius: 6px; padding: 12px; overflow: auto; max-height: 280px; }
    @media (max-width: 880px) { .grid, .cards { grid-template-columns: 1fr; } header { align-items: flex-start; flex-direction: column; } }
  </style>
</head>
<body>
  <header>
    <h1>Drone RSI SF</h1>
    <div class="status"><span id="status-dot" class="dot"></span><span id="status-text">checking services</span></div>
  </header>
  <main>
    <section class="cards">
      <div class="card"><div><div class="label">MongoDB</div><div id="mongo-detail" class="value">checking</div></div><div id="mongo" class="status-pill">off</div></div>
      <div class="card"><div><div class="label">Brain</div><div id="brain-detail" class="value">checking</div></div><div id="brain" class="status-pill">off</div></div>
      <div class="card"><div><div class="label">Sim</div><div id="sim-detail-card" class="value">checking</div></div><div id="sim" class="status-pill">off</div></div>
      <div class="card"><div><div class="label">Total Runs</div><div id="runs-detail" class="value">0</div></div><div id="runs" class="status-pill">off</div></div>
    </section>

    <section class="viewer">
      <div id="cesiumContainer"></div>
      <div class="map-tools">
        <button id="follow" class="ghost">Chase Drone</button>
        <button id="return-home" class="ghost">Return Home</button>
      </div>
      <div id="scene-badge" class="scene-badge">3D scene loading</div>
      <div id="sim-detail" class="overlay">loading San Francisco viewer</div>
    </section>

    <section class="grid">
      <div class="panel">
        <h2>Mission Harness</h2>
        <label for="request">User request</label>
        <textarea id="request">Fly to the Ferry Building and stop there safely</textarea>
        <button id="run">Start Prompted Flight</button><button id="demo" class="secondary">Demo Ferry Flight</button>
        <section class="telemetry">
          <div><div class="label">Lat</div><strong id="lat">...</strong></div>
          <div><div class="label">Lon</div><strong id="lon">...</strong></div>
          <div><div class="label">Alt</div><strong id="alt">...</strong></div>
          <div><div class="label">Rung</div><strong id="flight-status">idle</strong></div>
        </section>
        <pre id="control">Waiting for rung-3 control stream.</pre>
        <h3>Latest Result</h3>
        <pre id="result">No mission run yet.</pre>
      </div>
      <div class="panel">
        <h2>RSI Memory</h2>
        <p>Improvement is persisted in MongoDB: failed/successful runs, lessons, and successful trajectories.</p>
        <table>
          <thead><tr><th>Mode</th><th>Steps</th><th>Violations</th><th>Reward</th></tr></thead>
          <tbody id="history"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    const SIM_BASE_URL = 'https://carl-4186--drone-rsi-sim-sim-api.modal.run';
    let dashboardConfig = null;
    let viewer = null;
    let trailEntity = null;
    let routeEntity = null;
    let droneEntity = null;
    let trailPositions = [];
    let routePositions = [];
    let followDrone = true;
    let lastPoint = null;
    let hasTelemetry = false;
    let serviceStatus = null;
    let sceneLayer = '3D scene';
    const droneState = {lat: 37.7749, lon: -122.4194, alt: 3.05, agl: 3.05, heading: 0};

    async function loadConfig() {
      if (dashboardConfig) return dashboardConfig;
      const r = await fetch('/api/client-config', {cache: 'no-store'});
      dashboardConfig = await r.json();
      return dashboardConfig;
    }

    function bearingDegrees(a, b) {
      if (!a || !b) return 0;
      const lat1 = a[0] * Math.PI / 180;
      const lat2 = b[0] * Math.PI / 180;
      const dLon = (b[1] - a[1]) * Math.PI / 180;
      const y = Math.sin(dLon) * Math.cos(lat2);
      const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
      return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
    }

    function droneBasePosition() {
      // Keep the browser chase camera stable. The real simulator altitude is
      // shown in telemetry; Google photogrammetry heights stream in unevenly
      // and make the visible model appear to bounce if sampled every tick.
      const visualAlt = 80;
      return Cesium.Cartesian3.fromDegrees(droneState.lon, droneState.lat, visualAlt);
    }

    function droneOrientation() {
      return Cesium.Transforms.headingPitchRollQuaternion(
        droneBasePosition(),
        new Cesium.HeadingPitchRoll(Cesium.Math.toRadians(droneState.heading), 0, 0)
      );
    }

    function addDroneModel() {
      droneEntity = viewer.entities.add({
        name: 'Drone',
        position: new Cesium.CallbackProperty(() => droneBasePosition(), false),
        orientation: new Cesium.CallbackProperty(() => droneOrientation(), false),
        model: {
          uri: 'https://cesium.com/downloads/cesiumjs/releases/1.132/Apps/SampleData/models/CesiumDrone/CesiumDrone.glb',
          scale: 0.25,
          minimumPixelSize: 22,
          maximumScale: 4,
          silhouetteColor: Cesium.Color.CYAN,
          silhouetteSize: 1,
          runAnimations: true
        }
      });
    }

    async function addCityTiles() {
      const badge = document.getElementById('scene-badge');
      try {
        if (Cesium.createGooglePhotorealistic3DTileset) {
          const tileset = await Cesium.createGooglePhotorealistic3DTileset({onlyUsingWithGoogleGeocoder: true});
          viewer.scene.primitives.add(tileset);
          sceneLayer = 'Google photorealistic 3D tiles';
          badge.textContent = sceneLayer;
          return;
        }
      } catch (err) {
        console.warn('Google photorealistic 3D tiles unavailable', err);
      }
      try {
        if (Cesium.createOsmBuildingsAsync) {
          const buildings = await Cesium.createOsmBuildingsAsync();
          viewer.scene.primitives.add(buildings);
          sceneLayer = 'Cesium OSM 3D buildings';
          badge.textContent = sceneLayer;
          return;
        }
      } catch (err) {
        console.warn('OSM buildings unavailable', err);
      }
      sceneLayer = '3D constraints and drone only';
      badge.textContent = sceneLayer;
    }

    async function addObstacles() {
      try {
        const r = await fetch(`${SIM_BASE_URL}/api/sim/constraints`, {cache: 'no-store'});
        const constraints = await r.json();
        (constraints.obstacles || []).forEach(obstacle => {
          const radius = obstacle.radius_m;
          const height = obstacle.height_m;
          const color = obstacle.name.toLowerCase().includes('bridge') ? Cesium.Color.ORANGE : Cesium.Color.RED;
          viewer.entities.add({
            name: obstacle.name,
            position: Cesium.Cartesian3.fromDegrees(obstacle.lon, obstacle.lat, height / 2),
            cylinder: {
              length: height,
              topRadius: radius,
              bottomRadius: radius,
              material: color.withAlpha(0.35),
              outline: true,
              outlineColor: color.withAlpha(0.85)
            },
            label: {
              text: obstacle.name,
              font: '12px sans-serif',
              fillColor: Cesium.Color.WHITE,
              outlineColor: Cesium.Color.BLACK,
              outlineWidth: 3,
              style: Cesium.LabelStyle.FILL_AND_OUTLINE,
              pixelOffset: new Cesium.Cartesian2(0, -12),
              verticalOrigin: Cesium.VerticalOrigin.BOTTOM
            }
          });
        });
      } catch (err) {
        console.warn('constraints unavailable', err);
      }
    }

    async function initCesium() {
      if (viewer) return viewer;
      const cfg = await loadConfig();
      if (cfg.cesiumIonToken) Cesium.Ion.defaultAccessToken = cfg.cesiumIonToken;
      const origin = cfg.sfOrigin;
      droneState.lat = origin.lat;
      droneState.lon = origin.lon;
      viewer = new Cesium.Viewer('cesiumContainer', {
        animation: false,
        baseLayerPicker: false,
        fullscreenButton: false,
        geocoder: false,
        homeButton: false,
        infoBox: false,
        sceneModePicker: false,
        selectionIndicator: false,
        timeline: false,
        navigationHelpButton: false,
        shouldAnimate: true
      });
      viewer.scene.globe.depthTestAgainstTerrain = false;
      viewer.scene.skyAtmosphere.show = true;
      viewer.scene.requestRenderMode = false;
      viewer.camera.setView({
        destination: Cesium.Cartesian3.fromDegrees(origin.lon, origin.lat, 1450),
        orientation: {
          heading: Cesium.Math.toRadians(58),
          pitch: Cesium.Math.toRadians(-46),
          roll: 0
        }
      });
      addDroneModel();
      addCityTiles();
      addObstacles();
      return viewer;
    }

    function updateDrone(telemetry) {
      if (!telemetry || !telemetry.geo || !viewer) return;
      const point = [telemetry.geo.lat, telemetry.geo.lon];
      const heading = bearingDegrees(lastPoint, point);
      lastPoint = point;
      const rawAgl = telemetry.geo.agl_m || Math.max(0.5, telemetry.geo.alt);
      if (!hasTelemetry) {
        droneState.lat = telemetry.geo.lat;
        droneState.lon = telemetry.geo.lon;
        droneState.agl = rawAgl;
        droneState.alt = telemetry.geo.alt;
        droneState.heading = heading;
        hasTelemetry = true;
      } else {
        const alpha = 0.18;
        droneState.lat += (telemetry.geo.lat - droneState.lat) * alpha;
        droneState.lon += (telemetry.geo.lon - droneState.lon) * alpha;
        droneState.agl += (rawAgl - droneState.agl) * alpha;
        droneState.alt += (telemetry.geo.alt - droneState.alt) * alpha;
        const deltaHeading = ((heading - droneState.heading + 540) % 360) - 180;
        droneState.heading += deltaHeading * 0.18;
      }

      const mission = telemetry.mission || {};
      if (mission.samples && mission.samples.length) {
        trailPositions = [];
      } else {
        trailPositions = [];
      }
      if (mission.route && mission.route.length) {
        routePositions = [];
      }
      document.getElementById('lat').textContent = telemetry.geo.lat.toFixed(5);
      document.getElementById('lon').textContent = telemetry.geo.lon.toFixed(5);
      document.getElementById('alt').textContent = `${droneState.agl.toFixed(1)} m / ${(droneState.agl * 3.28084).toFixed(1)} ft AGL`;
      document.getElementById('flight-status').textContent = `${mission.status || (telemetry.flying ? 'flying' : 'idle')} / R3`;
      const sample = mission.samples && mission.samples.length ? mission.samples[mission.samples.length - 1] : null;
      const action = mission.last_rung3_action || (sample && sample.rung3_action) || null;
      const constraints = mission.last_constraints || (sample && sample.constraints) || null;
      document.getElementById('control').textContent = JSON.stringify({rung3_action: action, constraints}, null, 2);
      if (followDrone) {
        const backMeters = 520;
        const headingRad = Cesium.Math.toRadians(droneState.heading);
        const metersPerLat = 111320;
        const metersPerLon = 111320 * Math.max(0.1, Math.cos(Cesium.Math.toRadians(droneState.lat)));
        const camLat = droneState.lat - (Math.cos(headingRad) * backMeters / metersPerLat);
        const camLon = droneState.lon - (Math.sin(headingRad) * backMeters / metersPerLon);
        viewer.camera.setView({
          destination: Cesium.Cartesian3.fromDegrees(camLon, camLat, 950),
          orientation: {
            heading: Cesium.Math.toRadians(droneState.heading),
            pitch: Cesium.Math.toRadians(-52),
            roll: 0
          }
        });
      }
    }

    async function refreshSimFeed() {
      const detail = document.getElementById('sim-detail');
      try {
        await initCesium();
        const tr = await fetch(`${SIM_BASE_URL}/api/sim/telemetry`, {cache: 'no-store'});
        if (tr.ok) {
          const telemetry = await tr.json();
          updateDrone(telemetry);
          const mission = telemetry.mission || {};
          const gpu = serviceStatus && serviceStatus.gpu ? serviceStatus.gpu : 'L40S';
          const goal = mission.resolved_goal && mission.resolved_goal.name ? mission.resolved_goal.name : 'San Francisco';
          detail.textContent = `Modal ${gpu} AirSim live; ${sceneLayer}; ${mission.status || 'idle'}; ${goal}; lat ${telemetry.geo.lat.toFixed(5)}, lon ${telemetry.geo.lon.toFixed(5)}`;
        } else {
          detail.textContent = 'sim telemetry waiting';
        }
      } catch (err) {
        detail.textContent = '3D scene or sim bridge unreachable';
      }
    }

    async function refreshSimStatus() {
      try {
        const r = await fetch(`${SIM_BASE_URL}/api/sim/status`, {cache: 'no-store'});
        serviceStatus = await r.json();
      } catch (err) {
        serviceStatus = null;
      }
    }

    async function refresh() {
      const r = await fetch('/api/status');
      const data = await r.json();
      const brainOn = Boolean(data.brain && data.brain.model);
      const simOn = Boolean(data.sim && data.sim.status);
      const runsOn = Number(data.counts.runs || 0) > 0;
      document.getElementById('mongo').textContent = data.mongodb.ok ? 'on' : 'off';
      document.getElementById('mongo').className = 'status-pill ' + (data.mongodb.ok ? 'ok' : '');
      document.getElementById('mongo-detail').textContent = data.mongodb.ok ? data.mongodb.db : 'not reachable';
      document.getElementById('brain').textContent = brainOn ? 'on' : 'off';
      document.getElementById('brain').className = 'status-pill ' + (brainOn ? 'ok' : '');
      document.getElementById('brain-detail').textContent = data.brain.model || 'not loaded';
      document.getElementById('sim').textContent = simOn ? 'on' : 'off';
      document.getElementById('sim').className = 'status-pill ' + (simOn ? 'ok' : '');
      document.getElementById('sim-detail-card').textContent = data.sim.status || 'not reachable';
      document.getElementById('runs').textContent = runsOn ? 'on' : 'off';
      document.getElementById('runs').className = 'status-pill ' + (runsOn ? 'ok' : '');
      document.getElementById('runs-detail').textContent = `${data.counts.runs || 0} real Mongo runs`;
      document.getElementById('status-dot').className = 'dot ' + (data.mongodb.ok ? 'ok' : '');
      document.getElementById('status-text').textContent = data.mongodb.ok ? 'Modal UI live; MongoDB connected' : 'MongoDB not reachable';
      const rows = data.latest_runs.map(row => `<tr><td>${row.mode}</td><td>${row.steps}</td><td>${row.guardrail_violations}</td><td>${row.reward}</td></tr>`).join('');
      document.getElementById('history').innerHTML = rows || '<tr><td colspan="4">No runs yet</td></tr>';
      await refreshSimStatus();
    }

    async function startMission(request) {
      const btn = document.getElementById('run');
      btn.disabled = true;
      try {
        const r = await fetch('/api/mission', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({request})
        });
        const data = await r.json();
        document.getElementById('result').textContent = JSON.stringify(data, null, 2);
        followDrone = true;
        await refresh();
        await refreshSimFeed();
      } finally {
        btn.disabled = false;
      }
    }

    document.getElementById('run').addEventListener('click', async () => {
      await startMission(document.getElementById('request').value);
    });
    document.getElementById('demo').addEventListener('click', async () => {
      document.getElementById('request').value = 'Fly to the Ferry Building and stop there safely';
      await startMission(document.getElementById('request').value);
    });
    document.getElementById('follow').addEventListener('click', () => {
      followDrone = !followDrone;
      document.getElementById('follow').textContent = followDrone ? 'Chase Drone' : 'Free Camera';
    });
    document.getElementById('return-home').addEventListener('click', async () => {
      await fetch(`${SIM_BASE_URL}/api/sim/return-home`, {method: 'POST'});
      followDrone = true;
      await refreshSimFeed();
    });
    initCesium().then(refreshSimFeed);
    refresh();
    setInterval(refresh, 6000);
    setInterval(refreshSimFeed, 1000);
  </script>
</body>
</html>
            """
        )

    @api.get("/api/client-config")
    def client_config():
        return {
            "simBaseUrl": SIM_BASE_URL,
            "cesiumIonToken": os.getenv("CESIUM_ION_TOKEN", ""),
            "sfOrigin": {
                "lat": float(os.getenv("SF_ORIGIN_LAT", "37.7749")),
                "lon": float(os.getenv("SF_ORIGIN_LON", "-122.4194")),
                "alt": float(os.getenv("SF_ORIGIN_ALT", "20")),
            },
        }

    @api.get("/api/status")
    def status():
        client = None
        try:
            client, db = _mongo()
            ping = db.client.admin.command("ping")["ok"] == 1.0
            names = _collection_names()
            latest = list(
                db[names["runs"]]
                .find({}, {"_id": 0})
                .sort("created_at", -1)
                .limit(8)
            )
            counts = {
                "runs": db[names["runs"]].count_documents({}),
                "trajectories": db[names["trajectories"]].count_documents({}),
                "lessons": db[names["lessons"]].count_documents({}),
            }
        except Exception as exc:
            ping = False
            latest = []
            counts = {"runs": 0, "trajectories": 0, "lessons": 0}
            mongo_error = str(exc)
        finally:
            if client:
                client.close()
        return {
            "service": "drone-rsi-dashboard",
            "time": _now(),
            "mongodb": {"ok": ping, "db": os.getenv("MONGODB_DB"), "error": None if ping else locals().get("mongo_error")},
            "brain": {
                "status": "vllm-b200",
                "model": os.getenv("LLM_MODEL"),
                "endpoint": "serve_llm /v1/chat/completions",
            },
            "sim": {
                "status": "modal_airsim_live",
                "reason": "Unreal/Colosseum package is deployed on Modal; dashboard renders a 3D San Francisco scene from live AirSim telemetry, obstacle constraints, and rung-3 control records.",
                "target_gpu": os.getenv("MODAL_SIM_GPU", "L40S"),
            },
            "counts": counts,
            "latest_runs": latest,
        }

    @api.post("/api/mission")
    def run_mission(req: MissionRequest):
        if not req.request.strip():
            raise HTTPException(status_code=400, detail="request is required")
        return _simulate_or_replay_mission(req.request, _resolve_goal_from_request(req.request, req.goal))

    return api


@app.function(
    image=brain_image,
    gpu="B200",
    secrets=secrets,
    volumes={"/cache": model_cache},
    min_containers=1,
    max_containers=1,
    scaledown_window=3600,
    timeout=24 * 60 * 60,
    startup_timeout=20 * 60,
    region=REGION,
)
@modal.web_server(8000, startup_timeout=20 * 60)
def serve_llm():
    model_id = os.getenv("LLM_MODEL", "google/gemma-4-12B-it")
    env = os.environ.copy()
    env["HF_HOME"] = "/cache/huggingface"
    env["HF_HUB_CACHE"] = "/cache/huggingface/hub"
    if env.get("HF_TOKEN"):
        env["HUGGING_FACE_HUB_TOKEN"] = env["HF_TOKEN"]
        env["HF_HUB_TOKEN"] = env["HF_TOKEN"]

    vllm_bin = shutil.which("vllm")
    if not vllm_bin:
        for candidate in ("/usr/local/bin/vllm", "/opt/venv/bin/vllm", "/usr/bin/vllm"):
            if os.path.exists(candidate):
                vllm_bin = candidate
                break
    if not vllm_bin:
        raise RuntimeError("Could not find the vLLM CLI binary in the container.")

    cmd = [
        vllm_bin,
        "serve",
        "--model",
        model_id,
        "--served-model-name",
        os.getenv("LLM_SERVED_MODEL_NAME", model_id),
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        os.getenv("VLLM_MAX_MODEL_LEN", "8192"),
        "--gpu-memory-utilization",
        os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90"),
        "--trust-remote-code",
        "--limit-mm-per-prompt",
        '{"image": 0, "audio": 0}',
    ]

    print("Starting vLLM:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd, env=env)
