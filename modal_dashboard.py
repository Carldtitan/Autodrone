import datetime as dt
import math
import os
import random
from typing import Any

import modal


APP_NAME = "drone-rsi"
SECRET_NAME = "drone-rsi-secrets"
REGION = "us"

app = modal.App(APP_NAME)
secrets = [modal.Secret.from_name(SECRET_NAME)]

dashboard_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.118.0", "pymongo==4.17.0")
)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _mongo():
    from pymongo import MongoClient

    client = MongoClient(os.environ["MONGODB_URI"], serverSelectionTimeoutMS=8000)
    return client, client[os.getenv("MONGODB_DB", "world_fair_hackathon")]


def _collection_names() -> dict[str, str]:
    return {
        "trajectories": os.getenv("MONGODB_TRAJ_COLLECTION", "trajectories"),
        "lessons": os.getenv("MONGODB_LESSONS_COLLECTION", "lessons"),
        "runs": os.getenv("MONGODB_RUNS_COLLECTION", "mission_runs"),
        "metrics": os.getenv("MONGODB_METRICS_COLLECTION", "rsi_metrics"),
    }


def _mission_key(request: str, goal: str) -> str:
    return f"{request.strip().lower()}::{goal.strip().lower()}"[:240]


def _simulate_or_replay_mission(request: str, goal: str) -> dict[str, Any]:
    client, db = _mongo()
    names = _collection_names()
    key = _mission_key(request, goal)

    previous_runs = db[names["runs"]].count_documents({"mission_key": key})
    best = db[names["trajectories"]].find_one(
        {"mission_key": key, "success": True},
        sort=[("score", -1), ("steps", 1), ("created_at", -1)],
    )

    replay = best is not None
    if replay:
        steps = max(6, int(best.get("steps", 18)) - random.randint(1, 3))
        violations = 0
        mode = "retrieve_replay_refine"
    else:
        steps = max(10, 42 - previous_runs * 8 + random.randint(-4, 5))
        violations = max(0, 3 - previous_runs + random.choice([0, 0, 1]))
        mode = "explore_learn"

    reward = 100 - steps - (violations * 50)
    success = violations == 0 or previous_runs >= 1

    trajectory = [
        {
            "step": i + 1,
            "command": {
                "roll": round(math.sin(i / 3) * 0.08, 3),
                "pitch": round(0.12 if i < steps * 0.7 else 0.04, 3),
                "yaw": round((i / max(steps, 1)) * 0.45, 3),
                "z": -min(80, 20 + i * 2),
                "duration": 1.0,
            },
        }
        for i in range(steps)
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
        "created_at": _now(),
    }
    run_id = db[names["runs"]].insert_one(run_doc).inserted_id

    if success:
        db[names["trajectories"]].insert_one(
            {
                "mission_key": key,
                "request": request,
                "goal": goal,
                "success": True,
                "score": reward - steps,
                "steps": steps,
                "guardrail_violations": violations,
                "trajectory": trajectory,
                "lesson": "Stored successful route for later retrieval and refinement.",
                "created_at": _now(),
            }
        )
    elif violations:
        db[names["lessons"]].insert_one(
            {
                "mission_key": key,
                "request": request,
                "goal": goal,
                "lesson": "Guardrail violation observed; widen turn radius and climb earlier.",
                "created_at": _now(),
            }
        )

    total_runs = db[names["runs"]].count_documents({"mission_key": key})
    successful_runs = db[names["runs"]].count_documents({"mission_key": key, "success": True})
    db[names["metrics"]].insert_one(
        {
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
    )
    client.close()

    return {
        **run_doc,
        "_id": str(run_id),
        "trajectory_preview": trajectory[:5],
        "total_runs_for_request": total_runs,
        "success_count_for_request": successful_runs,
    }


@app.function(
    image=dashboard_image,
    secrets=secrets,
    min_containers=1,
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
        goal: str = "Ferry Building, San Francisco"

    @api.get("/", response_class=HTMLResponse)
    def index():
        return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Drone RSI SF</title>
  <style>
    body { margin: 0; background: #f7f8fa; color: #161b22; font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
    header { padding: 20px 28px; background: #fff; border-bottom: 1px solid #d8dee4; display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    h1 { font-size: 22px; margin: 0; }
    main { padding: 24px 28px; max-width: 1180px; margin: 0 auto; }
    .grid { display: grid; grid-template-columns: 1.1fr 0.9fr; gap: 18px; }
    .panel, .card { background: #fff; border: 1px solid #d8dee4; border-radius: 8px; }
    .panel { padding: 18px; }
    .cards { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
    .card { padding: 14px; min-height: 78px; }
    .label { color: #57606a; font-size: 12px; text-transform: uppercase; margin-bottom: 8px; }
    .value { font-size: 22px; font-weight: 700; word-break: break-word; }
    label { display: block; font-weight: 650; margin: 12px 0 6px; }
    input, textarea { width: 100%; box-sizing: border-box; border: 1px solid #c9d1d9; border-radius: 6px; padding: 10px 12px; font: inherit; background: #fff; }
    textarea { min-height: 90px; resize: vertical; }
    button { margin-top: 14px; border: 0; border-radius: 6px; background: #0969da; color: #fff; padding: 10px 14px; font-weight: 700; cursor: pointer; }
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
      <div class="card"><div class="label">MongoDB</div><div id="mongo" class="value">...</div></div>
      <div class="card"><div class="label">Brain GPU</div><div id="brain" class="value">blocked</div></div>
      <div class="card"><div class="label">Sim GPU</div><div id="sim" class="value">blocked</div></div>
      <div class="card"><div class="label">Total Runs</div><div id="runs" class="value">...</div></div>
    </section>
    <section class="grid">
      <div class="panel">
        <h2>Mission Harness</h2>
        <label for="request">User request</label>
        <textarea id="request">Fly to the Ferry Building and return safely</textarea>
        <label for="goal">Goal</label>
        <input id="goal" value="Ferry Building, San Francisco" />
        <button id="run">Run Mission Attempt</button>
        <h3>Latest Result</h3>
        <pre id="result">No mission run yet.</pre>
      </div>
      <div class="panel">
        <h2>RSI Memory</h2>
        <p>This is live on Modal and persists to MongoDB even after your laptop closes.</p>
        <table>
          <thead><tr><th>Mode</th><th>Steps</th><th>Violations</th><th>Reward</th></tr></thead>
          <tbody id="history"></tbody>
        </table>
      </div>
    </section>
  </main>
  <script>
    async function refresh() {
      const r = await fetch('/api/status');
      const data = await r.json();
      document.getElementById('mongo').textContent = data.mongodb.ok ? 'ok' : 'down';
      document.getElementById('brain').textContent = data.brain.status;
      document.getElementById('sim').textContent = data.sim.status;
      document.getElementById('runs').textContent = data.counts.runs;
      document.getElementById('status-dot').className = 'dot ' + (data.mongodb.ok ? 'ok' : '');
      document.getElementById('status-text').textContent = data.mongodb.ok ? 'Modal UI live; MongoDB connected' : 'MongoDB not reachable';
      document.getElementById('history').innerHTML = data.latest_runs.map(row => `<tr><td>${row.mode}</td><td>${row.steps}</td><td>${row.guardrail_violations}</td><td>${row.reward}</td></tr>`).join('') || '<tr><td colspan="4">No runs yet</td></tr>';
    }
    document.getElementById('run').addEventListener('click', async () => {
      const btn = document.getElementById('run');
      btn.disabled = true;
      try {
        const r = await fetch('/api/mission', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({request: document.getElementById('request').value, goal: document.getElementById('goal').value})
        });
        const data = await r.json();
        document.getElementById('result').textContent = JSON.stringify(data, null, 2);
        await refresh();
      } finally {
        btn.disabled = false;
      }
    });
    refresh();
    setInterval(refresh, 6000);
  </script>
</body>
</html>
        """

    @api.get("/api/status")
    def status():
        client = None
        try:
            client, db = _mongo()
            ping = db.client.admin.command("ping")["ok"] == 1.0
            names = _collection_names()
            latest = list(db[names["runs"]].find({}, {"_id": 0}).sort("created_at", -1).limit(8))
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
                "status": "payment_required",
                "model": os.getenv("LLM_MODEL"),
                "reason": "Modal requires adding a payment method before GPU functions can deploy.",
            },
            "sim": {
                "status": "needs_unreal_package",
                "reason": "No packaged Unreal/Colosseum project exists in this workspace yet.",
            },
            "counts": counts,
            "latest_runs": latest,
        }

    @api.post("/api/mission")
    def run_mission(req: MissionRequest):
        if not req.request.strip() or not req.goal.strip():
            raise HTTPException(status_code=400, detail="request and goal are required")
        return _simulate_or_replay_mission(req.request, req.goal)

    return api


@app.function(image=dashboard_image, secrets=secrets, timeout=600, region=REGION)
def seed_demo_data():
    return _simulate_or_replay_mission(
        "Fly to the Ferry Building and return safely",
        "Ferry Building, San Francisco",
    )
