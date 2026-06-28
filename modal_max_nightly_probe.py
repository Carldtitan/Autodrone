import os
import shutil
import subprocess

import modal


app = modal.App("max-gemma4-nightly-probe")
secret = modal.Secret.from_name(os.getenv("MODAL_SECRET_NAME", "drone-rsi-secrets"))
cache = modal.Volume.from_name("drone-rsi-model-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("modular/max-nvidia-full:latest", add_python="3.12")
    .entrypoint([])
    .run_commands(
        "python -m pip install --upgrade pip",
        "python -m pip install --pre --upgrade modular --extra-index-url https://whl.modular.com/nightly/simple/",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "MAX_CACHE_DIR": "/cache/max",
            "MODULAR_MAX_CACHE_DIR": "/cache/max",
            "MAX_SERVE_LOGS_CONSOLE_LEVEL": "INFO",
            "MODULAR_STRUCTURED_LOGGING": "0",
        }
    )
)


@app.function(
    image=image,
    gpu="B200",
    secrets=[secret],
    volumes={"/cache": cache},
    timeout=45 * 60,
    startup_timeout=20 * 60,
)
def probe():
    env = os.environ.copy()
    if env.get("HF_TOKEN"):
        env["HUGGING_FACE_HUB_TOKEN"] = env["HF_TOKEN"]
        env["HF_HUB_TOKEN"] = env["HF_TOKEN"]

    max_bin = shutil.which("max") or "/opt/venv/bin/max"
    print("max binary:", max_bin, flush=True)
    subprocess.run([max_bin, "--version"], check=False, env=env)

    cmd = [
        max_bin,
        "serve",
        "--model",
        os.getenv("LLM_MODEL", "google/gemma-4-12B-it"),
        "--served-model-name",
        os.getenv("LLM_SERVED_MODEL_NAME", "google/gemma-4-12B-it"),
        "--devices",
        "gpu:0",
        "--max-batch-size",
        "1",
        "--max-length",
        os.getenv("MODULAR_MAX_LENGTH", "8192"),
        "--device-memory-utilization",
        "0.8",
        "--no-device-graph-capture",
        "--trust-remote-code",
    ]
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False, env=env, text=True, timeout=25 * 60)
    print("max serve return code:", proc.returncode, flush=True)
    return {"returncode": proc.returncode}


@app.local_entrypoint()
def main():
    print(probe.remote())
