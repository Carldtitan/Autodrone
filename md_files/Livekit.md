LiveKit at the AI Engineer World's Fair Hackathon
LiveKit at the AI Engineer World's Fair Hackathon
Build a voice AI agent with LiveKit Agents. Hosted with Cerebral Valley at the AI Engineer World's Fair, San Francisco, on June 27–28, 2026. The theme is recursive self-improvement — agents that learn from their own outputs and get more capable as they iterate. Everything below is what you need to ship in a weekend.

Pick a starter, wire in your keys, and you have a real-time voice (and multimodal) agent running in minutes — so you can spend the hackathon on your idea, not the plumbing.

1. Install the LiveKit Docs MCP server
The fastest way to keep your coding agent on the rails. It pulls live docs, code examples, and changelogs straight into Cursor / Claude Code / VS Code so you stop fighting hallucinations.

Claude Code

claude mcp add --transport http livekit-docs https://docs.livekit.io/mcp
Cursor — one-click install

VS Code

code --add-mcp '{"name":"livekit-docs","type":"http","url":"https://docs.livekit.io/mcp"}'
Codex

codex mcp add --url https://docs.livekit.io/mcp livekit-docs
Full reference and other clients (Gemini CLI, Antigravity, Copilot CLI): docs.livekit.io/mcp.

Or use the LiveKit Agents skill
If you'd rather use a Claude Code skill instead of (or alongside) the MCP server:

npx skills add https://github.com/livekit/agent-skills --skill livekit-agents
Activates automatically for relevant tasks, or invoke with /livekit-agents.

2. Pick your starter
Two LiveKit starter kits, both with a Python agent and a Next.js frontend. Clone whichever fits your idea — or use them as a reference for the patterns.

Gemini multimodal starter
git clone https://github.com/livekit-examples/gemini-hacker-starter
A multimodal agent built on Google's models and LiveKit. Demonstrates:

Real-time voice + video — native audio and video understanding with Gemini 3.1 Flash Audio
Image generation — a function tool that generates images from text prompts with NanoBanana 2
Live music — streams generative music into the LiveKit room as a live audio track with Lyria RealTime
Voice-activity-driven video sampling — frames are sampled based on when the user speaks
You'll need two free accounts:

LiveKit Cloud — sign up at cloud.livekit.io for your LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET (grab your $50 in credits in step 3)
Google AI — a GOOGLE_API_KEY with access to Gemini 3.1, NanoBanana 2, and Lyria
Then follow the README:

cd agent && uv sync && cp .env.example .env.local       # add your LiveKit + Google keys
cd ../frontend && pnpm install && cp .env.example .env.local
cd ../agent && uv run agent.py dev                       # run the agent
# in a new terminal:
cd frontend && pnpm dev                                  # run the frontend
Open http://localhost:3000, click Start hacking, and talk to your agent.

MongoDB Atlas starter
git clone https://github.com/livekit-examples/mongodb-hacker-starter
A LiveKit voice agent (Python or Node) wired into MongoDB Atlas. Demonstrates five integration patterns out of the box:

RAG with $vectorSearch
Agentic memory with $rankFusion
Pre-loaded user context before the LLM's first turn
Function-tool CRUD against your domain collections
Session report persistence on disconnect
It uses LiveKit Inference, so STT/LLM/TTS are selected by model name and the only secrets you wire are LiveKit + MongoDB. Follow the README for pnpm setup → pnpm db:init → pnpm db:seed → pnpm dev:py (or pnpm dev:ts).

3. Redeem your $50 in inference credits
All attendees get a 7-day free trial of the LiveKit Scale plan — $50 in inference credits, no credit card required.

Create a project at cloud.livekit.io
Redeem at cloud.livekit.io/projects/p_/redeem
Code: HACK-AIEWF-2026
4. Ship it
Install the LiveKit CLI (one-time):

brew install livekit-cli
# or
curl -sSL https://get.livekit.io/cli | bash
CLI setup and reference: docs.livekit.io/intro/basics/cli.

When your agent works locally, deploy from inside the agent directory:

lk agent create
Then test against the frontend you cloned.

Prize
The top winning team takes home Keychron Q3 Max mechanical keyboards.

Judging favors:

Uniqueness — agents we haven't seen before
Technical depth — non-trivial use of LiveKit's framework and Cloud
Polish — the LiveKit integration feels seamless and is core to the product
Reference
LiveKit Agents docs
Python framework · Python examples
TypeScript framework
Voice AI quickstart
LiveKit Hackathon Starter Kits

Find relevant docs, resources, and mo