import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

import modal


app = modal.App("vllm-gemma4-probe")
secret = modal.Secret.from_name(os.getenv("MODAL_SECRET_NAME", "drone-rsi-secrets"))
cache = modal.Volume.from_name("drone-rsi-model-cache", create_if_missing=True)

image = (
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


def wait_json(url: str, timeout_s: int) -> dict:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(5)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def post_json(url: str, payload: dict, timeout_s: int = 180) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


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

    model = os.getenv("LLM_MODEL", "google/gemma-4-12B-it")
    vllm_bin = shutil.which("vllm")
    python_bin = shutil.which("python") or "python"
    print("python:", python_bin, flush=True)
    print("vllm:", vllm_bin, flush=True)
    if not vllm_bin:
        raise RuntimeError("The vLLM executable is not available in the container.")
    subprocess.run(["head", "-1", vllm_bin], check=False, env=env)
    subprocess.run([vllm_bin, "--help"], check=False, env=env)

    cmd = [
        vllm_bin,
        "serve",
        "--model",
        model,
        "--served-model-name",
        os.getenv("LLM_SERVED_MODEL_NAME", model),
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
    print("starting:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env)
    try:
        models = wait_json("http://127.0.0.1:8000/v1/models", timeout_s=25 * 60)
        print("models:", json.dumps(models)[:1000], flush=True)
        response = post_json(
            "http://127.0.0.1:8000/v1/chat/completions",
            {
                "model": os.getenv("LLM_SERVED_MODEL_NAME", model),
                "messages": [
                    {
                        "role": "user",
                        "content": "Return one JSON object with keys status and message. status must be ok.",
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 96,
            },
        )
        text = response["choices"][0]["message"]["content"]
        print("inference:", text, flush=True)
        return {"ok": True, "text": text}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


@app.local_entrypoint()
def main():
    print(probe.remote())
