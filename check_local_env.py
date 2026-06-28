from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"


def load_dotenv() -> dict[str, str]:
    values: dict[str, str] = {}
    if not ENV_PATH.exists():
        return values
    for raw in ENV_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def exists(path: str | None) -> bool:
    return bool(path and Path(path).exists())


def run_version(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=20)
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"
    out = (proc.stdout or proc.stderr).strip()
    return out.splitlines()[0] if out else f"exit {proc.returncode}"


def main() -> None:
    env = load_dotenv()
    unreal_root = env.get("UNREAL_ENGINE_ROOT") or r"C:\Program Files\Epic Games\UE_5.3"
    unreal_editor = env.get("UNREAL_EDITOR_EXE") or str(Path(unreal_root) / "Engine" / "Binaries" / "Win64" / "UnrealEditor.exe")
    run_uat = env.get("UNREAL_RUNUAT_BAT") or str(Path(unreal_root) / "Engine" / "Build" / "BatchFiles" / "RunUAT.bat")
    cesium_token = env.get("CESIUM_ION_TOKEN", "")
    report = {
        "workspace": str(ROOT),
        "python": run_version(["python", "--version"]),
        "modal": run_version(["python", "-m", "modal", "--version"]),
        "vscode_cli": shutil.which("code"),
        "unreal": {
            "root": unreal_root,
            "root_exists": exists(unreal_root),
            "editor": unreal_editor,
            "editor_exists": exists(unreal_editor),
            "run_uat": run_uat,
            "run_uat_exists": exists(run_uat),
        },
        "visual_studio": {
            "cl_on_path": shutil.which("cl"),
            "msbuild_on_path": shutil.which("msbuild"),
            "note": "cl/msbuild only appear automatically inside a VS Developer Command Prompt.",
        },
        "cesium": {
            "ion_token_set": bool(cesium_token),
            "google_maps_key_set": bool(env.get("GOOGLE_MAPS_API_KEY")),
        },
        "modal_app": {
            "app_name": env.get("MODAL_APP_NAME"),
            "secret_name": env.get("MODAL_SECRET_NAME"),
            "brain_gpu": env.get("MODAL_BRAIN_GPU"),
            "sim_gpu": env.get("MODAL_SIM_GPU"),
        },
        "rsi": {
            "mode": env.get("RSI_MODE"),
            "llm_model": env.get("LLM_MODEL"),
            "mongodb_db": env.get("MONGODB_DB"),
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
