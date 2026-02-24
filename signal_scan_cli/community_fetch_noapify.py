#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ddgs import DDGS


def search_site(keyword: str, site: str, max_results: int = 20):
    q = f'site:{site} "{keyword}"'
    out = []
    with DDGS(timeout=10) as ddgs:
        for r in ddgs.text(q, max_results=max_results):
            out.append(r)
    return out


def normalize(result: dict, source: str, keyword: str):
    now = datetime.now(timezone.utc).isoformat()
    title = (result.get("title") or "").strip()
    snippet = (result.get("body") or "").strip()
    url = (result.get("href") or "").strip()
    text = f"{title} | {snippet}".strip(" |")
    return {
        "source": source,
        "keyword": keyword,
        "title": title,
        "text": text,
        "url": url,
        "created_at": now,
        "engagement": 0,
    }


def main():
    ap = argparse.ArgumentParser(description="No-Apify community fetcher (Reddit/Threads via search)")
    ap.add_argument("--keywords", required=True, help="comma-separated keywords")
    ap.add_argument("--sources", default="reddit,threads", help="reddit,threads")
    ap.add_argument("--max-per-keyword", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]

    site_map = {
        "reddit": "reddit.com",
        "threads": "threads.net",
    }

    rows = []
    seen = set()

    for s in sources:
        site = site_map.get(s)
        if not site:
            continue
        for kw in keywords:
            try:
                results = search_site(kw, site, max_results=args.max_per_keyword)
            except Exception:
                results = []
            for r in results:
                url = (r.get("href") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                rows.append(normalize(r, s, kw))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[ok] wrote {len(rows)} rows -> {outp}")


if __name__ == "__main__":
    main()
