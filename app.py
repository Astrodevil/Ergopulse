from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
BASE_URL = "https://api.tokenfactory.us-north1.nebius.com/v1/"
DB_PATH = BASE_DIR / "ergopulse_history.sqlite3"

SYSTEM_PROMPT = """
You are ErgoPulse, a practical ergonomics and eye-comfort coach for software developers.
You review a short webcam-based posture test using numeric session metrics plus labeled key frames.

Important boundaries:
- This is wellness and ergonomics feedback, not a medical diagnosis.
- Do not claim the user has a disease or injury.
- Use careful wording such as "may contribute to", "can be associated with", or "worth adjusting".
- Be specific, practical, and concise.
- If image quality is poor, say confidence is low rather than inventing visual details.
- Ground posture feedback in mainstream ergonomics guidance: neutral supported sitting, head stacked over shoulders, relaxed shoulders, feet supported, monitor roughly arm's length away, top of screen at or slightly below eye level, and regular posture variation or short movement breaks.
- Ground eye feedback in computer-vision comfort guidance: blink quality/frequency, the 20-20-20 break habit, and avoiding long unbroken eye-open stretches.

Return strict JSON only. No markdown, no comments, no prose outside JSON.

JSON shape:
{
  "score": 0-100,
  "headline": "one concise sentence",
  "good": ["specific thing that looked good"],
  "needs_work": ["specific thing to improve"],
  "key_moment_review": ["what the labeled key frames visually suggest"],
  "before_after": ["how the baseline/early frame compares with the final or worst drift frame"],
  "risks": ["what this pattern may contribute to if repeated during long work sessions"],
  "fixes": ["concrete adjustment the user can do in the next minute"],
  "next_test": ["what to watch during the next 5-minute test"],
  "confidence": "low | medium | high"
}
"""


class PostureSummaryRequest(BaseModel):
    metrics: dict[str, Any] = Field(default_factory=dict)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    frames: list[Any] = Field(default_factory=list)


class HistorySession(BaseModel):
    id: str
    createdAt: str
    source: str = ""
    report: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    frames: list[dict[str, Any]] = Field(default_factory=list)
    trend: str = ""


app = FastAPI(title="ErgoPulse")
app.mount("/vendor", StaticFiles(directory=BASE_DIR / "vendor"), name="vendor")


@app.on_event("startup")
def startup() -> None:
    _init_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        BASE_DIR / "index.html",
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/sessions")
def list_sessions() -> dict[str, list[dict[str, Any]]]:
    _init_db()
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, source, report_json, metrics_json, timeline_json, events_json, frames_json, trend
            FROM sessions
            ORDER BY created_at DESC
            LIMIT 100
            """
        ).fetchall()
    return {"sessions": [_session_from_row(row) for row in rows]}


@app.post("/api/sessions")
def save_session(session: HistorySession) -> dict[str, bool]:
    _init_db()
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, created_at, source, report_json, metrics_json, timeline_json, events_json, frames_json, trend)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              created_at = excluded.created_at,
              source = excluded.source,
              report_json = excluded.report_json,
              metrics_json = excluded.metrics_json,
              timeline_json = excluded.timeline_json,
              events_json = excluded.events_json,
              frames_json = excluded.frames_json,
              trend = excluded.trend
            """,
            (
                session.id,
                session.createdAt,
                session.source,
                json.dumps(session.report),
                json.dumps(session.metrics),
                json.dumps(session.timeline),
                json.dumps(session.events),
                json.dumps(session.frames),
                session.trend,
            ),
        )
        conn.execute(
            """
            DELETE FROM sessions
            WHERE id NOT IN (
              SELECT id FROM sessions ORDER BY created_at DESC LIMIT 100
            )
            """
        )
    return {"ok": True}


@app.delete("/api/sessions")
def clear_sessions() -> dict[str, bool]:
    _init_db()
    with _db() as conn:
        conn.execute("DELETE FROM sessions")
    return {"ok": True}


@app.post("/api/posture-summary")
def posture_summary(request: PostureSummaryRequest) -> dict[str, Any]:
    api_key = _load_api_key()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Missing NEBIUS_API_KEY. Add it to the local .env file and restart the server.",
        )

    frames = _normalize_frames(request.frames)
    client = OpenAI(base_url=BASE_URL, api_key=api_key, timeout=75, max_retries=1)
    prompt = {
        "task": "Create a final report for a developer posture and eye-comfort test.",
        "metrics": request.metrics,
        "timeline_sample": request.timeline[-80:],
        "key_frame_manifest": [
            {
                "index": index + 1,
                "label": frame["label"],
                "time_seconds": frame["time_seconds"],
                "reason": frame["reason"],
            }
            for index, frame in enumerate(frames)
        ],
        "instructions": [
            "Base the report on metrics and the labeled key frames; do not describe body details that are not visible.",
            "Use the baseline/early frame and final or worst-drift frame for a before/after comparison when both are present.",
            "Mention blink rate compared with a typical adult range of about 15-20 blinks per minute and suggest 20-20-20 breaks when eye strain risk appears high.",
            "For posture, connect forward head, tilt, shoulder unevenness, or static sitting to possible neck, shoulder, and back discomfort risk without diagnosing.",
            "Do not diagnose. Keep the tone useful, direct, and encouraging.",
            "Return the exact JSON shape from the system prompt.",
        ],
    }

    content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(prompt, indent=2)}]
    content.extend({"type": "image_url", "image_url": {"url": frame["image"]}} for frame in frames)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.35,
            max_tokens=1800,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Nebius request failed: {str(exc)[:500]}") from exc

    raw = response.choices[0].message.content or "{}"
    report = _parse_json_object(raw)
    if not isinstance(report, dict):
        raise HTTPException(status_code=502, detail="Nebius returned a non-JSON summary.")

    return {"model": MODEL, "report": _normalize_report(report)}


def _normalize_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": _score(report.get("score")),
        "headline": _string(report.get("headline")),
        "good": _string_list(report.get("good")),
        "needs_work": _string_list(report.get("needs_work")),
        "key_moment_review": _string_list(report.get("key_moment_review")),
        "before_after": _string_list(report.get("before_after")),
        "risks": _string_list(report.get("risks")),
        "fixes": _string_list(report.get("fixes")),
        "next_test": _string_list(report.get("next_test")),
        "confidence": _string(report.get("confidence")),
    }


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              source TEXT NOT NULL DEFAULT '',
              report_json TEXT NOT NULL DEFAULT '{}',
              metrics_json TEXT NOT NULL DEFAULT '{}',
              timeline_json TEXT NOT NULL DEFAULT '[]',
              events_json TEXT NOT NULL DEFAULT '[]',
              frames_json TEXT NOT NULL DEFAULT '[]',
              trend TEXT NOT NULL DEFAULT ''
            )
            """
        )
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if "timeline_json" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN timeline_json TEXT NOT NULL DEFAULT '[]'")
        if "events_json" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN events_json TEXT NOT NULL DEFAULT '[]'")
        if "frames_json" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN frames_json TEXT NOT NULL DEFAULT '[]'")


def _session_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "createdAt": row["created_at"],
        "source": row["source"],
        "report": _json_object(row["report_json"]),
        "metrics": _json_object(row["metrics_json"]),
        "timeline": _json_list(row["timeline_json"]),
        "events": _json_list(row["events_json"]),
        "frames": _json_list(row["frames_json"]),
        "trend": row["trend"],
    }


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _json_list(value: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        return []


def _normalize_frames(values: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        if len(normalized) >= 8:
            break
        if isinstance(value, str):
            image = value
            label = f"Frame {index + 1}"
            reason = "Unlabeled session frame"
            time_seconds = None
        elif isinstance(value, dict):
            image = _string(value.get("image"))
            label = _string(value.get("label")) or f"Frame {index + 1}"
            reason = _string(value.get("reason")) or "Selected key frame"
            time_seconds = value.get("time_seconds")
        else:
            continue
        if not image.startswith("data:image/"):
            continue
        normalized.append(
            {
                "image": image,
                "label": label[:80],
                "reason": reason[:180],
                "time_seconds": _number_or_none(time_seconds),
            }
        )
    return normalized


def _number_or_none(value: Any) -> float | int | None:
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None


def _score(value: Any) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def _string(value: Any) -> str:
    if value is None or isinstance(value, (dict, list)):
        return ""
    return str(value).strip()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_string(item) for item in value if _string(item)][:8]


def _parse_json_object(content: str) -> Any:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise HTTPException(status_code=502, detail="Nebius returned text that was not valid JSON.")


def _load_api_key() -> str | None:
    api_key = os.environ.get("NEBIUS_API_KEY")
    if api_key:
        return api_key

    env_path = BASE_DIR / ".env"
    if not env_path.is_file():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or "=" not in clean:
            continue
        name, value = clean.split("=", 1)
        if name.strip() == "NEBIUS_API_KEY":
            return value.strip().strip('"').strip("'") or None
    return None
