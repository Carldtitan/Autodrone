from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parent
PROMPT_PATH = ROOT / "prompts" / "drone_rsi_system_prompt.md"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def normalized_key(*parts: str) -> str:
    joined = "::".join(p.strip().lower() for p in parts if p and p.strip())
    joined = re.sub(r"[^a-z0-9:._ -]+", "", joined)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined[:240]


STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "back",
    "fly",
    "go",
    "in",
    "of",
    "return",
    "safe",
    "safely",
    "sf",
    "san",
    "the",
    "to",
}


def mission_tags(request: str, goal: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", f"{request} {goal}".lower())
    tags = []
    for word in words:
        if len(word) < 3 or word in STOP_WORDS:
            continue
        if word not in tags:
            tags.append(word)
    return tags[:16]


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if obj.__class__.__name__ == "ObjectId":
        return str(obj)
    return obj


def read_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return "You are a drone policy. Output only rung-3 action JSON."


@dataclass
class DroneAction:
    roll: float
    pitch: float
    yaw: float
    z: float
    duration: float = 1.0

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "DroneAction":
        return cls(
            roll=clamp(float(obj.get("roll", 0.0)), -0.15, 0.15),
            pitch=clamp(float(obj.get("pitch", 0.0)), -0.15, 0.15),
            yaw=clamp(float(obj.get("yaw", 0.0)), -0.75, 0.75),
            z=clamp(float(obj.get("z", -40.0)), -120.0, -10.0),
            duration=clamp(float(obj.get("duration", 1.0)), 0.5, 2.0),
        )


@dataclass
class DroneState:
    x: float
    y: float
    z: float
    yaw: float
    velocity: float
    distance_to_goal: float
    collision: bool = False
    geofence_violation: bool = False
    altitude_violation: bool = False
    goal_reached: bool = False
    image_ref: str | None = None
    image_summary: str | None = None


@dataclass
class StepRecord:
    step_index: int
    prompt: str
    state: dict[str, Any]
    image_ref: str | None
    model_raw: str
    action: dict[str, Any]
    reward: float
    done: bool
    observation: dict[str, Any]
    context_k: int
    created_at: str = field(default_factory=now_iso)


@dataclass
class MissionSpec:
    request: str
    goal: str
    mission_key: str
    tags: list[str]

    @classmethod
    def create(cls, request: str, goal: str) -> "MissionSpec":
        return cls(
            request=request.strip(),
            goal=goal.strip(),
            mission_key=normalized_key(request, goal),
            tags=mission_tags(request, goal),
        )


@dataclass
class PolicyConfig:
    policy_id: str = "gemma4-12b-base"
    controller_ids: list[str] = field(default_factory=list)
    context_k: int = 5
    min_context_k: int = 3
    max_context_k: int = 12
    failure_streak_to_grow_k: int = 3
    prefer_exact_replay: bool = True
    max_steps: int = 40
    exploration_temperature: float = 0.7


@dataclass
class EpisodeResult:
    run_id: str
    mission: MissionSpec
    policy: PolicyConfig
    success: bool
    final_reward: float
    steps: int
    collisions: int
    guardrail_violations: int
    mode: str
    group_id: str | None
    replayed_from_memory: bool
    created_training_examples: int


class DroneEnvironment(Protocol):
    def reset(self, mission: MissionSpec) -> DroneState:
        ...

    def observe(self) -> DroneState:
        ...

    def execute(self, action: DroneAction) -> tuple[DroneState, float, bool, dict[str, Any]]:
        ...

    def clone(self) -> "DroneEnvironment":
        ...


class SimulatedDroneEnvironment:
    """Small deterministic stand-in for Colosseum/AirSim.

    The harness logic should work unchanged once this class is replaced with a
    real adapter. The state uses a simple local coordinate system; z is negative
    altitude to match AirSim conventions.
    """

    def __init__(self, seed: int = 7) -> None:
        self.rng = random.Random(seed)
        self.goal_xy = (120.0, 80.0)
        self.home_xy = (0.0, 0.0)
        self.state = DroneState(0.0, 0.0, -30.0, 0.0, 0.0, 1e9)
        self.previous_distance = 1e9
        self.step_count = 0
        self.last_action = DroneAction(0.0, 0.0, 0.0, -30.0, 1.0)

    def clone(self) -> "SimulatedDroneEnvironment":
        return copy.deepcopy(self)

    def reset(self, mission: MissionSpec) -> DroneState:
        self.step_count = 0
        self.last_action = DroneAction(0.0, 0.0, 0.0, -30.0, 1.0)
        # Stable pseudo goal per mission so repeated runs are comparable.
        h = abs(hash(mission.mission_key))
        self.goal_xy = (90.0 + (h % 70), 55.0 + ((h // 13) % 90))
        self.state = DroneState(0.0, 0.0, -30.0, 0.0, 0.0, 0.0)
        self._refresh_distance()
        self.previous_distance = self.state.distance_to_goal
        return self.observe()

    def observe(self) -> DroneState:
        self._refresh_distance()
        self.state.image_ref = f"sim://frame/{self.step_count:04d}"
        self.state.image_summary = (
            f"synthetic forward view; goal bearing {self._goal_bearing():.2f} rad; "
            f"distance {self.state.distance_to_goal:.1f}m"
        )
        return copy.deepcopy(self.state)

    def execute(self, action: DroneAction) -> tuple[DroneState, float, bool, dict[str, Any]]:
        self.step_count += 1
        action = DroneAction.from_obj(asdict(action))
        yaw = self.state.yaw + action.yaw * action.duration
        speed = max(0.0, 8.0 + action.pitch * 80.0)
        dx = math.cos(yaw) * speed * action.duration
        dy = math.sin(yaw) * speed * action.duration
        drift = action.roll * 20.0 * action.duration
        dx += -math.sin(yaw) * drift
        dy += math.cos(yaw) * drift
        self.state.x += dx
        self.state.y += dy
        self.state.z += (action.z - self.state.z) * 0.35
        self.state.yaw = yaw
        self.state.velocity = speed

        self._refresh_distance()
        collision = self._inside_building_zone(self.state.x, self.state.y, self.state.z)
        geofence = abs(self.state.x) > 220 or abs(self.state.y) > 220
        altitude = not (-120.0 <= self.state.z <= -10.0)
        reached = self.state.distance_to_goal < 12.0

        self.state.collision = collision
        self.state.geofence_violation = geofence
        self.state.altitude_violation = altitude
        self.state.goal_reached = reached

        progress = self.previous_distance - self.state.distance_to_goal
        smooth_penalty = (
            abs(action.roll - self.last_action.roll)
            + abs(action.pitch - self.last_action.pitch)
            + 0.2 * abs(action.yaw - self.last_action.yaw)
        )
        reward = progress * 0.08 - 0.05 - smooth_penalty
        if collision:
            reward -= 8.0
        if geofence:
            reward -= 5.0
        if altitude:
            reward -= 3.0
        if reached:
            reward += 20.0

        done = reached or collision or geofence or self.step_count >= 80
        obs = {
            "progress_m": progress,
            "distance_to_goal": self.state.distance_to_goal,
            "collision": collision,
            "geofence_violation": geofence,
            "altitude_violation": altitude,
            "goal_reached": reached,
        }
        self.previous_distance = self.state.distance_to_goal
        self.last_action = action
        return self.observe(), float(reward), bool(done), obs

    def _refresh_distance(self) -> None:
        gx, gy = self.goal_xy
        self.state.distance_to_goal = math.hypot(gx - self.state.x, gy - self.state.y)

    def _goal_bearing(self) -> float:
        gx, gy = self.goal_xy
        return math.atan2(gy - self.state.y, gx - self.state.x) - self.state.yaw

    @staticmethod
    def _inside_building_zone(x: float, y: float, z: float) -> bool:
        # A fake downtown obstacle: safe if high enough.
        in_zone = 45.0 < x < 85.0 and 25.0 < y < 95.0
        too_low = z > -55.0
        return bool(in_zone and too_low)


class ColosseumAirSimEnvironment:
    """Adapter skeleton for the real Unreal/Colosseum process."""

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self.host = host or os.getenv("AIRSIM_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("AIRSIM_PORT", "41451"))
        self.client = None
        self.goal_xy = (0.0, 0.0)

    def clone(self) -> "ColosseumAirSimEnvironment":
        raise NotImplementedError("real AirSim clone/reset snapshots must be implemented per simulator setup")

    def _connect(self):
        if self.client is not None:
            return self.client
        try:
            import airsim  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install the Colosseum/AirSim Python client to use the real adapter") from exc
        self.client = airsim.MultirotorClient(ip=self.host, port=self.port)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        return self.client

    def reset(self, mission: MissionSpec) -> DroneState:
        client = self._connect()
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        return self.observe()

    def observe(self) -> DroneState:
        client = self._connect()
        kin = client.getMultirotorState().kinematics_estimated
        pos = kin.position
        vel = kin.linear_velocity
        # TODO: transform Unreal/AirSim coordinates into goal distance once SF
        # georeference is finalized.
        speed = math.sqrt(vel.x_val**2 + vel.y_val**2 + vel.z_val**2)
        return DroneState(
            x=float(pos.x_val),
            y=float(pos.y_val),
            z=float(pos.z_val),
            yaw=0.0,
            velocity=float(speed),
            distance_to_goal=0.0,
            image_ref=None,
            image_summary="real AirSim frame capture not wired yet",
        )

    def execute(self, action: DroneAction) -> tuple[DroneState, float, bool, dict[str, Any]]:
        client = self._connect()
        client.moveByRollPitchYawZAsync(
            action.roll,
            action.pitch,
            action.yaw,
            action.z,
            action.duration,
        ).join()
        state = self.observe()
        # TODO: replace with real verifier based on goal, geofence, altitude and
        # collision info from AirSim/Cesium physics.
        reward = -0.05
        done = False
        obs = {"real_adapter": True, "verifier": "pending"}
        return state, reward, done, obs


class MongoStore:
    def __init__(self) -> None:
        from pymongo import MongoClient

        uri = os.environ["MONGODB_URI"]
        self.client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        self.db = self.client[os.getenv("MONGODB_DB", "world_fair_hackathon")]
        self.names = {
            "runs": os.getenv("MONGODB_RUNS_COLLECTION", "mission_runs"),
            "trajectories": os.getenv("MONGODB_TRAJ_COLLECTION", "trajectories"),
            "lessons": os.getenv("MONGODB_LESSONS_COLLECTION", "lessons"),
            "metrics": os.getenv("MONGODB_METRICS_COLLECTION", "rsi_metrics"),
            "rollouts": os.getenv("MONGODB_ROLLOUTS_COLLECTION", "rsi_rollouts"),
            "training_examples": os.getenv("MONGODB_TRAINING_COLLECTION", "rsi_training_examples"),
            "controllers": os.getenv("MONGODB_CONTROLLERS_COLLECTION", "rsi_controllers"),
            "policy_versions": os.getenv("MONGODB_POLICY_COLLECTION", "rsi_policy_versions"),
        }
        self.ensure_indexes()

    def col(self, name: str):
        return self.db[self.names[name]]

    def ensure_indexes(self) -> None:
        self.col("runs").create_index([("mission_key", 1), ("created_at", -1)])
        self.col("trajectories").create_index([("mission_key", 1), ("success", 1), ("score", -1)])
        self.col("trajectories").create_index([("tags", 1), ("score", -1)])
        self.col("lessons").create_index([("tags", 1), ("created_at", -1)])
        self.col("rollouts").create_index([("group_id", 1), ("final_reward", -1)])
        self.col("training_examples").create_index([("source_run_id", 1), ("reward", -1)])
        self.col("controllers").create_index([("tags", 1), ("active", 1), ("eval.avg_reward", -1)])

    def exact_trajectory(self, mission: MissionSpec) -> dict[str, Any] | None:
        doc = self.col("trajectories").find_one(
            {"mission_key": mission.mission_key, "success": True},
            sort=[("score", -1), ("steps", 1), ("created_at", -1)],
        )
        return json_safe(doc) if doc else None

    def similar_trajectories(self, mission: MissionSpec, limit: int = 3) -> list[dict[str, Any]]:
        rows = list(
            self.col("trajectories")
            .find({"tags": {"$in": mission.tags}, "success": True}, {"trajectory": {"$slice": 6}})
            .sort([("score", -1), ("steps", 1)])
            .limit(limit)
        )
        return [json_safe(r) for r in rows]

    def lessons(self, mission: MissionSpec, limit: int = 5) -> list[dict[str, Any]]:
        rows = list(
            self.col("lessons")
            .find({"tags": {"$in": mission.tags}})
            .sort("created_at", -1)
            .limit(limit)
        )
        return [json_safe(r) for r in rows]

    def controllers(self, mission: MissionSpec, limit: int = 3) -> list[dict[str, Any]]:
        rows = list(
            self.col("controllers")
            .find({"tags": {"$in": mission.tags}, "active": True})
            .sort([("eval.avg_reward", -1), ("created_at", -1)])
            .limit(limit)
        )
        return [json_safe(r) for r in rows]

    def insert_run(self, doc: dict[str, Any]) -> str:
        return str(self.col("runs").insert_one(json_safe(doc)).inserted_id)

    def insert_rollout(self, doc: dict[str, Any]) -> str:
        return str(self.col("rollouts").insert_one(json_safe(doc)).inserted_id)

    def insert_trajectory(self, doc: dict[str, Any]) -> str:
        return str(self.col("trajectories").insert_one(json_safe(doc)).inserted_id)

    def insert_lesson(self, doc: dict[str, Any]) -> str:
        return str(self.col("lessons").insert_one(json_safe(doc)).inserted_id)

    def insert_training_examples(self, docs: list[dict[str, Any]]) -> int:
        if not docs:
            return 0
        self.col("training_examples").insert_many([json_safe(d) for d in docs])
        return len(docs)

    def register_controller(self, doc: dict[str, Any]) -> str:
        self.col("controllers").update_many(
            {"controller_id": doc["controller_id"], "active": True},
            {"$set": {"active": False, "retired_at": now_iso()}},
        )
        return str(self.col("controllers").insert_one(json_safe(doc)).inserted_id)

    def export_training_examples(self, out: Path, min_reward: float = 0.0, limit: int = 2000) -> int:
        rows = list(
            self.col("training_examples")
            .find({"reward": {"$gte": float(min_reward)}})
            .sort([("reward", -1), ("created_at", -1)])
            .limit(int(limit))
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({"prompt": row["prompt"], "completion": row["completion"]}, ensure_ascii=False) + "\n")
        return len(rows)


class GemmaClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = os.getenv("LLM_SERVED_MODEL_NAME") or os.getenv("LLM_MODEL", "google/gemma-4-12B-it")
        self.api_key = os.getenv("LLM_API_KEY", "not-needed")

    def chat(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2, image_path: str | None = None) -> str:
        content: str | list[dict[str, Any]]
        if image_path and Path(image_path).exists():
            b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
            content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        else:
            content = user_prompt
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": 320,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return str(data["choices"][0]["message"]["content"])
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            return self._fallback_action(user_prompt, exc)

    @staticmethod
    def _fallback_action(user_prompt: str, exc: Exception) -> str:
        # Deterministic fallback keeps the harness usable while the LLM endpoint
        # is cold or being worked on by another agent.
        m = re.search(r"goal bearing ([\-0-9.]+)", user_prompt)
        bearing = float(m.group(1)) if m else 0.0
        yaw = clamp(bearing * 0.35, -0.4, 0.4)
        action = {
            "tool": "execute_rung3_action",
            "action": {"roll": 0.0, "pitch": 0.09, "yaw": yaw, "z": -65.0, "duration": 1.0},
            "request_context_k_change": None,
            "finish_mission": False,
            "rationale": f"fallback heuristic because LLM call failed: {type(exc).__name__}",
        }
        return json.dumps(action)


class PromptBuilder:
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def build(
        self,
        mission: MissionSpec,
        policy: PolicyConfig,
        state: DroneState,
        recent_steps: list[StepRecord],
        exact_route: dict[str, Any] | None,
        similar_routes: list[dict[str, Any]],
        lessons: list[dict[str, Any]],
        controllers: list[dict[str, Any]],
    ) -> str:
        recent = [
            {
                "step": s.step_index,
                "state": s.state,
                "action": s.action,
                "reward": round(s.reward, 4),
                "observation": s.observation,
            }
            for s in recent_steps[-policy.context_k :]
        ]
        exact_preview = None
        if exact_route:
            actions = exact_route.get("trajectory") or exact_route.get("actions") or []
            exact_preview = {
                "trajectory_id": str(exact_route.get("_id") or exact_route.get("trajectory_id")),
                "score": exact_route.get("score"),
                "steps": exact_route.get("steps"),
                "first_actions": actions[: min(8, len(actions))],
            }
        ctx = {
            "mission": asdict(mission),
            "policy": asdict(policy),
            "current_state": asdict(state),
            "exact_success_route": exact_preview,
            "similar_success_routes": similar_routes[:3],
            "failure_lessons": [
                {"lesson": x.get("lesson"), "tags": x.get("tags"), "failure_type": x.get("failure_type")}
                for x in lessons[:5]
            ],
            "active_controllers": [
                {"controller_id": x.get("controller_id"), "tags": x.get("tags"), "eval": x.get("eval")}
                for x in controllers[:3]
            ],
            "recent_steps": recent,
        }
        return (
            "Use the following JSON context to choose the next safe rung-3 action.\n"
            "Return only the action JSON required by the system prompt.\n\n"
            + json.dumps(ctx, indent=2, ensure_ascii=False)
        )


def parse_model_action(raw: str) -> tuple[DroneAction, dict[str, Any] | None, bool, str]:
    text = raw.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = {}
    action_obj = obj.get("action") if isinstance(obj.get("action"), dict) else obj
    action = DroneAction.from_obj(action_obj if isinstance(action_obj, dict) else {})
    k_change = obj.get("request_context_k_change") if isinstance(obj.get("request_context_k_change"), dict) else None
    finish = bool(obj.get("finish_mission", False))
    rationale = str(obj.get("rationale", ""))
    return action, k_change, finish, rationale


class DroneRsiHarness:
    def __init__(
        self,
        store: MongoStore,
        llm: GemmaClient,
        env: DroneEnvironment,
        *,
        policy: PolicyConfig | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.env = env
        self.policy = policy or PolicyConfig()
        self.prompt_builder = PromptBuilder(read_system_prompt())

    def run_episode(self, mission: MissionSpec, *, group_id: str | None = None) -> EpisodeResult:
        policy = copy.deepcopy(self.policy)
        exact = self.store.exact_trajectory(mission)
        similar = self.store.similar_trajectories(mission)
        lessons = self.store.lessons(mission)
        controllers = self.store.controllers(mission)
        policy.controller_ids = [str(c.get("controller_id")) for c in controllers if c.get("controller_id")]

        mode = "mongo_exact_replay" if exact and policy.prefer_exact_replay else "agent_rollout"
        replay_actions = (exact or {}).get("trajectory") or (exact or {}).get("actions") or []

        state = self.env.reset(mission)
        steps: list[StepRecord] = []
        total_reward = 0.0
        collisions = 0
        guardrails = 0
        done = False
        no_progress_streak = 0
        previous_distance = state.distance_to_goal

        for idx in range(policy.max_steps):
            if mode == "mongo_exact_replay" and idx < len(replay_actions):
                raw_action = replay_actions[idx].get("command") or replay_actions[idx].get("action") or replay_actions[idx]
                action = DroneAction.from_obj(raw_action)
                raw = json.dumps(
                    {
                        "tool": "execute_rung3_action",
                        "action": asdict(action),
                        "request_context_k_change": None,
                        "finish_mission": False,
                        "rationale": "replaying verified MongoDB trajectory",
                    }
                )
                prompt = "Mongo exact route replay"
            else:
                prompt = self.prompt_builder.build(mission, policy, state, steps, exact, similar, lessons, controllers)
                raw = self.llm.chat(
                    self.prompt_builder.system_prompt,
                    prompt,
                    temperature=policy.exploration_temperature,
                )
                action, k_change, finish, _rationale = parse_model_action(raw)
                if k_change:
                    self._maybe_apply_k_change(policy, k_change, no_progress_streak)
                if finish:
                    done = True
            if not (mode == "mongo_exact_replay" and idx < len(replay_actions)):
                action, _k_change, _finish, _rationale = parse_model_action(raw)

            next_state, reward, env_done, obs = self.env.execute(action)
            total_reward += reward
            collisions += 1 if obs.get("collision") else 0
            guardrails += 1 if obs.get("geofence_violation") or obs.get("altitude_violation") else 0
            if next_state.distance_to_goal >= previous_distance - 0.25:
                no_progress_streak += 1
            else:
                no_progress_streak = 0
            previous_distance = next_state.distance_to_goal
            if no_progress_streak >= policy.failure_streak_to_grow_k:
                policy.context_k = min(policy.max_context_k, policy.context_k + 2)
                no_progress_streak = 0

            record = StepRecord(
                step_index=idx + 1,
                prompt=prompt,
                state=asdict(state),
                image_ref=state.image_ref,
                model_raw=raw,
                action=asdict(action),
                reward=reward,
                done=bool(env_done or done),
                observation=obs,
                context_k=policy.context_k,
            )
            steps.append(record)
            state = next_state
            if env_done or done:
                done = True
                break

        success = bool(state.goal_reached) and collisions == 0 and guardrails == 0
        run_doc = {
            "mission_key": mission.mission_key,
            "request": mission.request,
            "goal": mission.goal,
            "tags": mission.tags,
            "mode": mode,
            "policy_id": policy.policy_id,
            "controller_ids": policy.controller_ids,
            "context_k": policy.context_k,
            "success": success,
            "reward": total_reward,
            "steps": len(steps),
            "collisions": collisions,
            "guardrail_violations": guardrails,
            "replayed_from_memory": mode == "mongo_exact_replay",
            "group_id": group_id,
            "created_at": now_iso(),
        }
        run_id = self.store.insert_run(run_doc)
        rollout_doc = {
            **run_doc,
            "run_id": run_id,
            "structured_steps": [asdict(s) for s in steps],
            "final_state": asdict(state),
        }
        self.store.insert_rollout(rollout_doc)

        if success:
            self.store.insert_trajectory(
                {
                    "mission_key": mission.mission_key,
                    "request": mission.request,
                    "goal": mission.goal,
                    "tags": mission.tags,
                    "source_run_id": run_id,
                    "success": True,
                    "score": total_reward,
                    "steps": len(steps),
                    "collisions": collisions,
                    "guardrail_violations": guardrails,
                    "trajectory": [
                        {"step": s.step_index, "command": s.action, "reward": s.reward, "image_ref": s.image_ref}
                        for s in steps
                    ],
                    "created_at": now_iso(),
                }
            )
        elif collisions or guardrails:
            self.store.insert_lesson(
                {
                    "mission_key": mission.mission_key,
                    "request": mission.request,
                    "goal": mission.goal,
                    "tags": mission.tags,
                    "source_run_id": run_id,
                    "failure_type": "collision" if collisions else "guardrail",
                    "lesson": self._lesson_from_failure(collisions, guardrails, state),
                    "created_at": now_iso(),
                }
            )

        examples = self._training_examples_from_steps(run_id, mission, steps, success)
        created_examples = self.store.insert_training_examples(examples)
        return EpisodeResult(
            run_id=run_id,
            mission=mission,
            policy=policy,
            success=success,
            final_reward=total_reward,
            steps=len(steps),
            collisions=collisions,
            guardrail_violations=guardrails,
            mode=mode,
            group_id=group_id,
            replayed_from_memory=mode == "mongo_exact_replay",
            created_training_examples=created_examples,
        )

    def run_cycle(self, missions: list[MissionSpec], *, rollouts_per_mission: int) -> list[EpisodeResult]:
        results: list[EpisodeResult] = []
        for mission in missions:
            group_id = f"group-{mission.mission_key}-{int(time.time())}-{random.randint(1000, 9999)}"
            group: list[EpisodeResult] = []
            for _ in range(rollouts_per_mission):
                result = self.run_episode(mission, group_id=group_id)
                group.append(result)
                results.append(result)
            self._write_group_advantages(group)
        return results

    def _write_group_advantages(self, group: list[EpisodeResult]) -> None:
        if not group:
            return
        mean_reward = sum(r.final_reward for r in group) / len(group)
        for r in group:
            self.store.col("runs").update_one(
                {"_id": self._object_id_or_none(r.run_id)},
                {"$set": {"group_mean_reward": mean_reward, "grpo_advantage": r.final_reward - mean_reward}},
            )
            self.store.col("rollouts").update_one(
                {"run_id": r.run_id},
                {"$set": {"group_mean_reward": mean_reward, "grpo_advantage": r.final_reward - mean_reward}},
            )

    @staticmethod
    def _object_id_or_none(value: str):
        try:
            from bson import ObjectId

            return ObjectId(value)
        except Exception:
            return value

    @staticmethod
    def _maybe_apply_k_change(policy: PolicyConfig, request: dict[str, Any], no_progress_streak: int) -> None:
        try:
            requested = int(request.get("new_k", policy.context_k))
        except (TypeError, ValueError):
            return
        if no_progress_streak >= 2 or requested < policy.context_k:
            policy.context_k = int(clamp(requested, policy.min_context_k, policy.max_context_k))

    @staticmethod
    def _lesson_from_failure(collisions: int, guardrails: int, state: DroneState) -> str:
        if collisions:
            return "Collision during rollout; increase altitude margin and reduce pitch before dense obstacle regions."
        if guardrails:
            return "Guardrail violation during rollout; reduce yaw/roll aggression and return toward mission corridor."
        return f"Mission failed near distance {state.distance_to_goal:.1f}m; prefer prior high-progress actions."

    @staticmethod
    def _training_examples_from_steps(
        run_id: str,
        mission: MissionSpec,
        steps: list[StepRecord],
        success: bool,
    ) -> list[dict[str, Any]]:
        if not steps:
            return []
        # Distill successful runs and positive-reward steps from failed runs.
        rows: list[dict[str, Any]] = []
        for step in steps:
            if not success and step.reward < 0.2:
                continue
            completion = json.dumps(
                {
                    "tool": "execute_rung3_action",
                    "action": step.action,
                    "request_context_k_change": None,
                    "finish_mission": step.done,
                    "rationale": "high-reward simulator-verified action",
                },
                separators=(",", ":"),
            )
            rows.append(
                {
                    "source_run_id": run_id,
                    "mission_key": mission.mission_key,
                    "tags": mission.tags,
                    "prompt": step.prompt,
                    "completion": completion,
                    "reward": step.reward,
                    "success_run": success,
                    "context_k": step.context_k,
                    "created_at": now_iso(),
                }
            )
        return rows


DEFAULT_MISSIONS = [
    ("Fly to the Ferry Building and return safely", "Ferry Building, San Francisco"),
    ("Fly toward Coit Tower while avoiding downtown buildings", "Coit Tower, San Francisco"),
    ("Cross the waterfront corridor without geofence violations", "San Francisco waterfront"),
]


def build_harness(args: argparse.Namespace) -> DroneRsiHarness:
    load_dotenv()
    store = MongoStore()
    llm = GemmaClient()
    env: DroneEnvironment
    if getattr(args, "env", "sim") == "airsim":
        env = ColosseumAirSimEnvironment()
    else:
        env = SimulatedDroneEnvironment(seed=int(getattr(args, "seed", 7)))
    policy = PolicyConfig(
        policy_id=getattr(args, "policy_id", "gemma4-12b-base"),
        context_k=int(getattr(args, "context_k", int(os.getenv("CONTEXT_WINDOW_PAST", "5")))),
        max_steps=int(getattr(args, "max_steps", 40)),
        prefer_exact_replay=not getattr(args, "no_exact_replay", False),
        exploration_temperature=float(getattr(args, "temperature", 0.7)),
    )
    return DroneRsiHarness(store, llm, env, policy=policy)


def cmd_episode(args: argparse.Namespace) -> None:
    harness = build_harness(args)
    mission = MissionSpec.create(args.request, args.goal)
    result = harness.run_episode(mission)
    print(json.dumps(json_safe(asdict(result)), indent=2))


def cmd_cycle(args: argparse.Namespace) -> None:
    harness = build_harness(args)
    missions = [MissionSpec.create(req, goal) for req, goal in DEFAULT_MISSIONS]
    results = harness.run_cycle(missions, rollouts_per_mission=args.rollouts_per_mission)
    out = {
        "episodes": len(results),
        "successes": sum(1 for r in results if r.success),
        "avg_reward": sum(r.final_reward for r in results) / max(1, len(results)),
        "runs": [asdict(r) for r in results],
    }
    print(json.dumps(json_safe(out), indent=2))
    if args.train_controller:
        train_path = Path(args.train_jsonl)
        count = harness.store.export_training_examples(train_path, min_reward=args.min_reward)
        print(f"exported {count} training examples to {train_path}")
        if count:
            train_ntk_controller(harness.store, train_path, Path(args.controller_out), args)


def cmd_export_ntk(args: argparse.Namespace) -> None:
    load_dotenv()
    store = MongoStore()
    count = store.export_training_examples(Path(args.out), min_reward=args.min_reward, limit=args.limit)
    print(json.dumps({"out": args.out, "examples": count}, indent=2))


def train_ntk_controller(store: MongoStore, train_path: Path, out_path: Path, args: argparse.Namespace) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = getattr(args, "ntk_model", None) or os.getenv("NTKMIRROR_MODEL") or os.getenv("LLM_MODEL", "")
    if not model:
        print("skipping NTK-Mirror: no model configured", file=sys.stderr)
        return
    cmd = [
        "ntkmirror",
        "fit",
        "--model",
        model,
        "--train",
        str(train_path),
        "--out",
        str(out_path),
        "--gates",
        str(getattr(args, "ntk_gates", 5000)),
        "--steps",
        str(getattr(args, "ntk_steps", 120)),
        "--batch-size",
        str(getattr(args, "ntk_batch_size", 2)),
        "--max-length",
        str(getattr(args, "ntk_max_length", 1024)),
    ]
    print("running:", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        print("ntkmirror command not found; install leochlon/ntkmirror to train controllers", file=sys.stderr)
        return
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return
    artifact_bytes = out_path.stat().st_size if out_path.exists() else 0
    controller_id = out_path.stem
    store.register_controller(
        {
            "controller_id": controller_id,
            "artifact_path": str(out_path),
            "model": model,
            "tags": ["drone", "rung3", "behavior"],
            "active": True,
            "method": "ntkmirror",
            "artifact_bytes": artifact_bytes,
            "trained_from": str(train_path),
            "eval": {"status": "needs_eval", "avg_reward": 0.0},
            "created_at": now_iso(),
        }
    )
    print(f"registered controller {controller_id} ({artifact_bytes} bytes)")


def cmd_train_ntk(args: argparse.Namespace) -> None:
    load_dotenv()
    store = MongoStore()
    train_ntk_controller(store, Path(args.train), Path(args.out), args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Autonomous RSI harness for the drone project")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--env", choices=["sim", "airsim"], default="sim")
        sp.add_argument("--seed", type=int, default=7)
        sp.add_argument("--policy-id", default="gemma4-12b-base")
        sp.add_argument("--context-k", type=int, default=int(os.getenv("CONTEXT_WINDOW_PAST", "5")))
        sp.add_argument("--max-steps", type=int, default=40)
        sp.add_argument("--temperature", type=float, default=0.7)
        sp.add_argument("--no-exact-replay", action="store_true")

    ep = sub.add_parser("episode")
    add_common(ep)
    ep.add_argument("--request", required=True)
    ep.add_argument("--goal", required=True)
    ep.set_defaults(func=cmd_episode)

    cy = sub.add_parser("cycle")
    add_common(cy)
    cy.add_argument("--rollouts-per-mission", type=int, default=3)
    cy.add_argument("--train-controller", action="store_true")
    cy.add_argument("--train-jsonl", default="runs/ntk/drone_train.jsonl")
    cy.add_argument("--controller-out", default="runs/ntk/drone_behavior_controller.pt")
    cy.add_argument("--min-reward", type=float, default=0.2)
    cy.add_argument("--ntk-model", default=None)
    cy.add_argument("--ntk-gates", type=int, default=5000)
    cy.add_argument("--ntk-steps", type=int, default=120)
    cy.add_argument("--ntk-batch-size", type=int, default=2)
    cy.add_argument("--ntk-max-length", type=int, default=1024)
    cy.set_defaults(func=cmd_cycle)

    ex = sub.add_parser("export-ntk")
    ex.add_argument("--out", default="runs/ntk/drone_train.jsonl")
    ex.add_argument("--min-reward", type=float, default=0.2)
    ex.add_argument("--limit", type=int, default=2000)
    ex.set_defaults(func=cmd_export_ntk)

    tr = sub.add_parser("train-ntk")
    tr.add_argument("--train", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--ntk-model", default=None)
    tr.add_argument("--ntk-gates", type=int, default=5000)
    tr.add_argument("--ntk-steps", type=int, default=120)
    tr.add_argument("--ntk-batch-size", type=int, default=2)
    tr.add_argument("--ntk-max-length", type=int, default=1024)
    tr.set_defaults(func=cmd_train_ntk)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
