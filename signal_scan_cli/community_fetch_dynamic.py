#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
No-API / No-Apify community fetcher.
- Reddit: subreddit RSS + DDGS fallback
- Threads: DDGS search (site:threads.net)

Outputs normalized JSONL for issue_radar_cli.py
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import requests
from ddgs import DDGS

TAB_TO_REDDIT_T = {"today": "day", "week": "week", "month": "month"}
TAB_TO_DDGS_TIMELIMIT = {"today": "d", "week": "w", "month": "m"}

FALLBACK_SUBREDDITS = {
    "마케팅": [
        "marketing",
        "Entrepreneur",
        "solopreneur",
        "SideProject",
        "startups",
        "digitalnomad",
        "webdev",
        "SaaS",
        "indiehackers",
    ]
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.lower().strip())


def keyword_match_score(text: str, keywords: list[str]) -> int:
    t = normalize_text(text)
    score = 0
    for kw in keywords:
        k1 = normalize_text(kw)
        k2 = k1.replace(" ", "")
        if k1 and k1 in t:
            score += 1
        elif k2 and k2 in t.replace(" ", ""):
            score += 1
    return score


def parse_reddit_rss(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    ch = root.find("channel")
    if ch is None:
        return []
    items = []
    for item in ch.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        items.append({"title": title, "url": link, "published": pub_date})
    return items


def extract_subreddit_from_url(url: str) -> str | None:
    m = re.search(r"reddit\.com/r/([A-Za-z0-9_]+)/?", url)
    return m.group(1) if m else None


def ddgs_search(query: str, timelimit: str, max_results: int = 10) -> list[dict[str, str]]:
    out = []
    with DDGS(timeout=10) as ddgs:
        for r in ddgs.text(query, timelimit=timelimit, max_results=max_results):
            out.append({
                "title": r.get("title") or "",
                "url": r.get("href") or "",
                "snippet": r.get("body") or "",
            })
    return out


def recommend_subreddits(category: str, keywords: list[str], max_subreddits: int = 10) -> list[str]:
    keyword_en = {
        "솔로 창업": ["solopreneur", "solo founder", "indie business"],
        "솔로 개발": ["solo developer", "indie hacker", "build in public"],
        "노마드": ["digital nomad", "remote work"],
    }
    tokens: list[str] = []
    for k in keywords:
        tokens.append(k)
        tokens.extend(keyword_en.get(k, []))
    tokens = tokens[:7]

    query = f"reddit subreddits for {' '.join(tokens)}"
    try:
        rows = ddgs_search(query, timelimit="m", max_results=20)
    except Exception:
        rows = []

    subs = []
    seen = set()
    for r in rows:
        u = r.get("url", "")
        s = extract_subreddit_from_url(u)
        if s and s.lower() not in seen:
            seen.add(s.lower())
            subs.append(s)
        if len(subs) >= max_subreddits:
            break

    if not subs:
        subs = FALLBACK_SUBREDDITS.get(category, FALLBACK_SUBREDDITS["마케팅"])
    return subs[:max_subreddits]


def fetch_reddit_rss(subreddit: str, tab: str, limit: int = 15) -> list[dict[str, Any]]:
    t = TAB_TO_REDDIT_T.get(tab, "week")
    # reddit search RSS query
    url = f"https://www.reddit.com/r/{subreddit}/search.rss?q=solopreneur+OR+%22digital+nomad%22+OR+%22indie+hacker%22&restrict_sr=1&sort=top&t={t}"
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return parse_reddit_rss(resp.text)[:limit]
    except Exception:
        return []


def to_row(source: str, title: str, text: str, url: str, published: str | None, subreddit: str | None, via: str, keywords: list[str]) -> dict[str, Any]:
    combo = f"{title} {text}".strip()
    return {
        "source": source,
        "title": title,
        "text": combo,
        "url": url,
        "created_at": now_utc_iso(),
        "published": published,
        "subreddit": subreddit,
        "via": via,
        "match_score": keyword_match_score(combo, keywords),
        "engagement": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", default="마케팅")
    ap.add_argument("--keywords", required=True)
    ap.add_argument("--sources", default="reddit,threads")
    ap.add_argument("--reddit-subreddits", default="")
    ap.add_argument("--tab", default="week", choices=["today", "week", "month"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]

    rows: list[dict[str, Any]] = []
    seen = set()

    if "reddit" in sources:
        if args.reddit_subreddits.strip():
            subreddits = [s.strip() for s in args.reddit_subreddits.split(",") if s.strip()]
        else:
            subreddits = recommend_subreddits(args.category, keywords)

        for sub in subreddits:
            for item in fetch_reddit_rss(sub, args.tab, limit=15):
                url = item.get("url", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                title = item.get("title", "")
                rows.append(to_row("reddit", title, "", url, item.get("published"), sub, "rss", keywords))

    if "threads" in sources:
        tl = TAB_TO_DDGS_TIMELIMIT.get(args.tab, "w")
        for kw in keywords:
            q = f'site:threads.net "{kw}"'
            try:
                res = ddgs_search(q, timelimit=tl, max_results=10)
            except Exception:
                res = []
            for r in res:
                url = r.get("url", "")
                if not url or url in seen:
                    continue
                seen.add(url)
                rows.append(to_row("threads", r.get("title", ""), r.get("snippet", ""), url, None, None, "ddgs", keywords))

    # keep only somewhat relevant rows
    rows = [r for r in rows if r.get("match_score", 0) >= 1]

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
