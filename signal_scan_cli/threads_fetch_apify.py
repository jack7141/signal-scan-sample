#!/usr/bin/env python3
"""
Fetch Threads posts via Apify and normalize to issue_radar JSONL format.

Env:
- APIFY_TOKEN (required)
- APIFY_TASK_THREADS (optional, preferred)
- APIFY_ACTOR_THREADS (optional fallback)
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def _pick(d: dict[str, Any], keys: list[str], default=None):
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            return v
    return default


def normalize(item: dict[str, Any]) -> dict[str, Any]:
    text = _pick(item, ["text", "caption", "content", "title"], "")
    url = _pick(item, ["url", "postUrl", "link", "permalink"], "")
    created = _pick(item, ["timestamp", "createdAt", "publishedAt", "date"], None)

    likes = _pick(item, ["likesCount", "likes", "likeCount"], 0) or 0
    comments = _pick(item, ["commentsCount", "comments", "replyCount"], 0) or 0

    return {
        "source": "threads",
        "title": (text or "")[:140],
        "text": text,
        "url": url,
        "created_at": created,
        "likes": likes,
        "comments": comments,
    }


def run_task(apify_token: str, task_id: str, timeout_secs: int = 120) -> list[dict[str, Any]]:
    u = f"https://api.apify.com/v2/actor-tasks/{task_id}/run-sync-get-dataset-items"
    r = requests.post(u, params={"token": apify_token, "timeout": timeout_secs})
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return data
    return []


def run_actor(apify_token: str, actor_id: str, run_input: dict[str, Any], timeout_secs: int = 180) -> list[dict[str, Any]]:
    u = f"https://api.apify.com/v2/acts/{actor_id}/run-sync-get-dataset-items"
    r = requests.post(
        u,
        params={"token": apify_token, "timeout": timeout_secs},
        json=run_input,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        return data
    return []


def main():
    ap = argparse.ArgumentParser(description="Fetch Threads via Apify and write normalized JSONL")
    ap.add_argument("--keywords", required=True, help="comma-separated keywords")
    ap.add_argument("--out", required=True, help="output jsonl path")
    ap.add_argument("--max-items", type=int, default=100)
    ap.add_argument("--task-id", default=os.getenv("APIFY_TASK_THREADS", ""))
    ap.add_argument("--actor-id", default=os.getenv("APIFY_ACTOR_THREADS", ""))
    args = ap.parse_args()

    token = os.getenv("APIFY_TOKEN", "").strip()
    if not token:
        raise SystemExit("APIFY_TOKEN is required")

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]

    raw_items: list[dict[str, Any]] = []
    if args.task_id:
        raw_items = run_task(token, args.task_id)
    elif args.actor_id:
        run_input = {
            "searchTerms": keywords,
            "maxItems": args.max_items,
        }
        raw_items = run_actor(token, args.actor_id, run_input)
    else:
        raise SystemExit("Provide APIFY_TASK_THREADS or APIFY_ACTOR_THREADS")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for it in raw_items[: args.max_items]:
            n = normalize(it)
            if not n.get("text"):
                continue
            if not n.get("created_at"):
                n["created_at"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
            count += 1

    print(f"[ok] wrote {count} rows -> {out_path}")


if __name__ == "__main__":
    main()
