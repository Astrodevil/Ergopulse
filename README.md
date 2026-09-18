# ErgoPulse

AI-assisted webcam posture and blink checks for developers who spend long sessions at a desk.

> Want to build with open models on Nebius? Register for the [Nebius Builder Program](https://dub.sh/AIStudio) and get up to **$4,400 in AI credits**. You can also discover 100+ AI projects built using open models on [awesome-ai-apps](https://github.com/Arindam200/awesome-ai-apps).

ErgoPulse is a lightweight web app that runs a 1-minute or 5-minute desk check using your webcam. It tracks face, blink, and upper-body posture signals locally in the browser, then uses an LLM vision summary to turn the session into a practical ergonomics report.

## Run In Under A Minute

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your Nebius Token Factory key to `.env`:

```bash
NEBIUS_API_KEY=your_nebius_token_factory_key_here
```

Start ErgoPulse:

```bash
uvicorn app:app --host 127.0.0.1 --port 8010
```

Open `http://127.0.0.1:8010/`.

## What It Does

- Tracks blink rate, long eye-open streaks, head tilt/turn/drift, and shoulder/lean signals.
- Uses local MediaPipe models for real-time face and pose landmark detection.
- Sends selected session metrics and key frames to Nebius Token Factory for the final report.
- Stores session history locally in SQLite so users can compare scores, graphs, and key frames over time.
- Shows hoverable graphs for blink rate, posture load, and score history.

## LLM Powered By Nebius Token Factory

The final report is generated through [Nebius Token Factory](https://tokenfactory.nebius.com/) using:

- Model endpoint: [deepseek-ai/DeepSeek-V4.1-Flash](https://tokenfactory.nebius.com/?modals=endpoint-details&model-id=deepseek-ai/DeepSeek-V4.1-Flash)
- Model card: [DeepSeek-V4.1-Flash](https://www.deepseek.com/en/news/deepseek-v4-1-flash/)

DeepSeek describes V4.1-Flash as a faster, more efficient model in the V4.1 family with native visual understanding and multimodal support. ErgoPulse uses that vision capability only for the end-of-session report, while live tracking stays local in the browser.

## Privacy Shape

- Live face and pose tracking runs locally with MediaPipe in your browser.
- Nebius receives the session metrics, recent timeline samples, and a small set of labeled key frames only when a report is generated.
- Session history is stored in a local SQLite database: `ergopulse_history.sqlite3`.
- API keys stay in your local `.env` file and should never be committed.

This is wellness and ergonomics feedback, not medical diagnosis.

## How It Works

1. Browser loads local MediaPipe assets from `vendor/`.
2. MediaPipe Face Landmarker estimates blink and head signals.
3. MediaPipe Pose Landmarker Lite estimates shoulder and upper-body posture signals.
4. ErgoPulse samples the session and builds graphs locally.
5. At the end, selected key frames plus metrics are sent to Nebius Token Factory.
6. DeepSeek-V4.1-Flash returns a structured JSON report.
7. The app saves the session locally for history and comparison.

## For Contributors

Good next areas to build on:

- Add a short neutral calibration step for shoulders and head position.
- Improve posture scoring with landmark visibility/confidence checks.
- Add export for session reports as PDF or Markdown.
- Add optional BYOK settings in the UI instead of requiring `.env`.
- Add tests for history persistence, prompt normalization, and report parsing.
- Package the app with Docker for one-command local runs.

Please keep the app privacy-first: local analysis by default, explicit BYOK, and no committed session data or API keys.

## Tech Stack

- FastAPI
- OpenAI-compatible Python SDK
- Nebius Token Factory
- DeepSeek-V4.1-Flash
- MediaPipe Face Landmarker
- MediaPipe Pose Landmarker Lite
- SQLite
- Vanilla HTML/CSS/JavaScript
