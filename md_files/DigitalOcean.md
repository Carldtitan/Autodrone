# Build with DigitalOcean AI

> Everything you need to build with DigitalOcean's AI offerings. DigitalOcean
> Inference Engine gives you every model and every modality behind one
> OpenAI-compatible API key. This reference is built for both humans and LLMs.
> Three ways to put it to work in your coding agent: paste this URL, paste this
> markdown, or install the agent skill (`npx skills add ajot/digitalocean-ai`,
> https://github.com/ajot/digitalocean-ai). Any of them gives your agent enough
> context to unblock you on anything DigitalOcean.

What you can do with DigitalOcean: run inference across 55+ models, route requests
by cost or latency, evaluate outputs (LLM-as-a-Judge), build agents with knowledge
bases, then deploy your app on App Platform with a managed PostgreSQL + pgvector
database and Spaces object storage.

---

## What you can do with DigitalOcean

One key, one OpenAI-compatible endpoint. Here's what's behind it:

- **Run any model** — 55+ chat and reasoning models (Anthropic, OpenAI, Meta, DeepSeek, Qwen, and more) on one endpoint. Swap the model name, nothing else changes.
- **Generate images, audio & video** — multimodal generation through the same key: images sync, audio and video via async jobs.
- **Build agents with RAG** — managed agents backed by knowledge bases that retrieve from your own docs at inference time.
- **Route & evaluate** — auto-route each request to the cheapest or fastest model, then score outputs with LLM-as-a-Judge.
- **Scale & save** — batch jobs up to ~50% cheaper, or dedicated GPU endpoints with bring-your-own-model.
- **Deploy & store** — ship to App Platform from GitHub, with managed PostgreSQL + pgvector and Spaces object storage.

---

## Get started

The fastest path: create one model access key and make your first inference call
in under a minute. Everything below runs against a single OpenAI-compatible
endpoint, so most existing code works by changing only the `base_url`.

---

## Quick start: your first inference call in 60 seconds

### 1. Create a model access key

1. Sign up or log in at the [DigitalOcean console](https://cloud.digitalocean.com).
2. In the left nav, open **Inference > Serverless Inference**, then on the **Get Started** tab click **Create a Model Access Key**.
3. Export it:

```bash
export DIGITAL_OCEAN_MODEL_ACCESS_KEY="your-key-here"
```

One key unlocks every model and every modality (text, image, audio, video).

### 2. Try it with zero code: the Model Playground

The fastest "first 60 seconds" is the **Model Playground** — test and compare
text, image, audio, and video models side by side in the browser, tune
parameters, then export a ready-to-run cURL or SDK snippet. No code required.

Docs: https://docs.digitalocean.com/products/inference/how-to/use-model-playground/

### 3. First inference call (~60 seconds)

Base URL: `https://inference.do-ai.run/v1/`
Auth: `Authorization: Bearer $DIGITAL_OCEAN_MODEL_ACCESS_KEY`

#### cURL

```bash
curl -s -X POST https://inference.do-ai.run/v1/chat/completions \
  -H "Authorization: Bearer $DIGITAL_OCEAN_MODEL_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.3-70b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_completion_tokens": 256
  }'
```

#### Python (OpenAI SDK — the recommended path)

```bash
pip install openai python-dotenv
```

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://inference.do-ai.run/v1/",
    api_key=os.getenv("DIGITAL_OCEAN_MODEL_ACCESS_KEY"),
)

response = client.chat.completions.create(
    model="llama3.3-70b-instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    max_completion_tokens=256,
)

print(response.choices[0].message.content)
```

The endpoints are **OpenAI- and Anthropic-compatible**, so the OpenAI SDK,
LangChain, and LlamaIndex work by pointing `base_url` at DigitalOcean — no other
code changes. (A native Gradient SDK exists but is being deprecated; lead with
the OpenAI SDK.)

---

## AI offerings: inference, routing, evals & agents

DigitalOcean Inference Engine: every model, every modality, on one platform.

### Serverless Inference

Real-time responses across 55+ models with auto-scaling, pay per token. One API
key unlocks every model and modality. Best for getting started and variable
traffic.

Docs: https://docs.digitalocean.com/products/inference/how-to/use-serverless-inference/

### Batch Inference

Async jobs with a 24-hour SLA, up to ~50% cheaper than real-time. Good for large
offline workloads (bulk classification, summarization, embeddings backfills).

Docs: https://docs.digitalocean.com/products/inference/

### Dedicated Inference

Dedicated GPU endpoints for sustained, high-throughput workloads. Bring Your Own
Model via Hugging Face imports. Public or private endpoints.

Docs: https://docs.digitalocean.com/products/inference/

### Model catalog

70+ open-source models plus Day 0 access to select OpenAI and Anthropic releases
(Anthropic, OpenAI, Meta, DeepSeek, Mistral, Qwen, NVIDIA, and more). Curated
highlights:

**Text & reasoning** — call via `/v1/chat/completions`:

| Model ID | Provider |
|---|---|
| `anthropic-claude-opus-4.8` | Anthropic |
| `anthropic-claude-haiku-4.5` | Anthropic |
| `anthropic-claude-4.6-sonnet` | Anthropic |
| `openai-gpt-5.5` | OpenAI |
| `openai-gpt-5.4-mini` | OpenAI |
| `openai-o3` (reasoning) | OpenAI |
| `llama3.3-70b-instruct` | Meta |
| `deepseek-3.2` | DeepSeek |
| `alibaba-qwen3-32b` | Alibaba |
| `kimi-k2.6` | Moonshot AI |

**Image** — call via `/v1/images/generations` (synchronous):

| Model ID | Provider |
|---|---|
| `openai-gpt-image-2` | OpenAI |
| `stable-diffusion-3.5-large` | Stability AI |

**Image, audio & speech (fal models)** — call via `/v1/async-invoke` (submit-poll-retrieve):

| Model ID | Provider | Type |
|---|---|---|
| `fal-ai/flux/schnell` | fal | Image |
| `fal-ai/elevenlabs/tts/multilingual-v2` | fal | Text-to-speech |
| `fal-ai/stable-audio-25/text-to-audio` | fal | Audio |

`client.models.list()` returns the synchronous models only; the async fal models
(image, audio, speech) run through `/v1/async-invoke` and are not in that list:

```python
for m in client.models.list().data:
    print(m.id)
```

Full catalog: https://docs.digitalocean.com/products/inference/details/models/

### Image, audio & video generation

Multimodal generation through the same single API key. OpenAI image models run
synchronously; fal models (image, audio, TTS) use an async submit-poll-retrieve
workflow.

A few image models to start with: `openai-gpt-image-2` and
`stable-diffusion-3.5-large` (sync), or `fal-ai/flux/schnell` for fast async
generation.

#### Sync image (OpenAI)

```bash
curl -s -X POST https://inference.do-ai.run/v1/images/generations \
  -H "Authorization: Bearer $DIGITAL_OCEAN_MODEL_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai-gpt-image-2",
    "prompt": "A cute baby sea otter wearing a hat",
    "n": 1,
    "size": "1024x1024"
  }' | jq -r '.data[0].b64_json' | base64 --decode > image.png
```

Or with the OpenAI SDK (same `client` as above):

```python
import base64

result = client.images.generate(
    model="openai-gpt-image-2",
    prompt="A cute baby sea otter wearing a hat",
    n=1,
    size="1024x1024",
)

with open("image.png", "wb") as f:
    f.write(base64.b64decode(result.data[0].b64_json))

print("Saved image.png")
```

#### Async (fal models): submit, poll, retrieve

```python
import os, time, requests

BASE_URL = "https://inference.do-ai.run/v1"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('DIGITAL_OCEAN_MODEL_ACCESS_KEY')}",
    "Content-Type": "application/json",
}

def async_generate(model_id, input_data, poll_interval=2):
    resp = requests.post(f"{BASE_URL}/async-invoke", headers=HEADERS, json={
        "model_id": model_id,
        "input": input_data,
    })
    request_id = resp.json()["request_id"]
    while True:
        status = requests.get(
            f"{BASE_URL}/async-invoke/{request_id}/status", headers=HEADERS
        ).json()["status"]
        if status == "COMPLETED":
            break
        time.sleep(poll_interval)
    return requests.get(f"{BASE_URL}/async-invoke/{request_id}", headers=HEADERS).json()

# Image
async_generate("fal-ai/flux/schnell", {"prompt": "A futuristic cityscape at sunset"})
# Text-to-speech
async_generate("fal-ai/elevenlabs/tts/multilingual-v2",
               {"text": "Hello from DigitalOcean Inference!", "voice_id": "JBFqnCBsd6RMkjVDRZzb"})
```

### Agent Platform + Knowledge Bases

Persistent AI agents with attached knowledge bases. Unlike stateless serverless
inference, agents keep context and can reference your uploaded documents (RAG).
Agents use a separate endpoint and access key (create them in the console).

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("AGENT_ENDPOINT"),   # https://agents.do-ai.run/v1/<agent-id>
    api_key=os.getenv("AGENT_ACCESS_KEY"),
)

response = client.chat.completions.create(
    model="n/a",  # required by SDK, ignored by the agent (uses its configured model)
    messages=[{"role": "user", "content": "What can you help me with?"}],
)
print(response.choices[0].message.content)
```

Docs: https://docs.digitalocean.com/products/inference/

### Inference Router (public preview)

Routes each request to the best-fit model by cost or latency, with automatic
fallback if a model is unavailable. One router endpoint, many models behind it.

Docs: https://docs.digitalocean.com/products/inference/how-to/use-inference-router/

### Evaluations (public preview)

LLM-as-a-Judge scoring of models, routers, and dedicated deployments against your
own datasets. Configure candidate + judge, pick metrics, run from the console or
API. Agent-level evals are also available.

Docs: https://docs.digitalocean.com/products/inference/how-to/evaluate-models/

### Compatibility & observability

Endpoints are OpenAI- and Anthropic-compatible, so existing code migrates by
changing the `base_url`. Built-in observability covers tokens, latency, errors,
and spend.

---

## Deploy and store what you build

Deploy what you build, with the data layer to back it.

### App Platform (deploy from GitHub)

Push to GitHub, point App Platform at the repo, get continuous deployment.
Requirements: listen on port **8080**, and for Flask include `gunicorn` with
`--timeout 120` (inference calls, especially image generation, exceed the default
30s).

```python
# app.py
import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(
    base_url="https://inference.do-ai.run/v1/",
    api_key=os.getenv("DIGITAL_OCEAN_MODEL_ACCESS_KEY"),
)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    r = client.chat.completions.create(
        model=data.get("model", "llama3.3-70b-instruct"),
        messages=data["messages"],
        max_completion_tokens=data.get("max_tokens", 256),
    )
    return jsonify({"response": r.choices[0].message.content})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
```

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "app:app"]
```

Docs: https://docs.digitalocean.com/products/app-platform/

### Managed PostgreSQL + pgvector

Managed Postgres with the `pgvector` extension for embeddings and semantic
search — the storage layer for RAG. Docs:
https://docs.digitalocean.com/products/databases/postgresql/

### Spaces object storage

S3-compatible object storage with a built-in CDN for generated images, audio,
and uploads. Docs: https://docs.digitalocean.com/products/spaces/

---

## Point your coding agent at the skill

An agent skill packages this whole guide so your LLM or coding agent can use it
directly. Install it:

```bash
npx skills add ajot/digitalocean-ai
```

It covers serverless inference, dedicated inference, the model catalog,
image/audio generation, agents, databases, Spaces, droplets, and App Platform
deploys. When you're stuck, this is the first thing to point your agent at.

---

## Get help

1. Point your LLM/coding agent at the skill above (`npx skills add ajot/digitalocean-ai`).
2. See the docs at https://docs.digitalocean.com/products/inference/.

---

## Quick reference

- Base URL: `https://inference.do-ai.run/v1/`
- Auth: `Authorization: Bearer $DIGITAL_OCEAN_MODEL_ACCESS_KEY`
- Key endpoints:
  - `GET  /v1/models` — list models
  - `POST /v1/chat/completions` — chat
  - `POST /v1/responses` — text/multimodal responses with reasoning
  - `POST /v1/embeddings` — embeddings
  - `POST /v1/images/generations` — image generation (sync, OpenAI)
  - `POST /v1/audio/speech` — text-to-speech
  - `POST /v1/async-invoke` — async image/audio/video (fal models)
- SDK: OpenAI SDK is the primary path (point `base_url` at the endpoint). Gradient SDK is legacy / being deprecated.
- Docs home: https://docs.digitalocean.com/products/inference/
