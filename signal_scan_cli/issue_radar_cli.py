#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


TIME_KEYS = ["created_at", "published_at", "timestamp", "time"]
TEXT_KEYS = ["title", "text", "content", "body", "snippet"]
URL_KEYS = ["url", "link", "permalink"]
SOURCE_KEYS = ["source", "platform"]
ENGAGEMENT_KEYS = ["engagement", "score", "upvotes", "likes", "comments"]


@dataclass
class Item:
    source: str
    text: str
    url: str
    dt: datetime
    engagement: float


def parse_dt(obj: dict[str, Any]) -> datetime | None:
    for k in TIME_KEYS:
        v = obj.get(k)
        if not v:
            continue
        if isinstance(v, (int, float)):
            if v > 10_000_000_000:
                return datetime.fromtimestamp(v / 1000, tz=timezone.utc)
            return datetime.fromtimestamp(v, tz=timezone.utc)
        if isinstance(v, str):
            vv = v.replace("Z", "+00:00")
            try:
                d = datetime.fromisoformat(vv)
                if d.tzinfo is None:
                    d = d.replace(tzinfo=timezone.utc)
                return d
            except Exception:
                pass
    return None


def pick_text(obj: dict[str, Any]) -> str:
    chunks = []
    for k in TEXT_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            chunks.append(v.strip())
    return " | ".join(chunks)[:500]


def pick_url(obj: dict[str, Any]) -> str:
    for k in URL_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.startswith("http"):
            return v
    return ""


def pick_source(obj: dict[str, Any]) -> str:
    for k in SOURCE_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.lower().strip()
    return "unknown"


def pick_engagement(obj: dict[str, Any]) -> float:
    val = 0.0
    for k in ENGAGEMENT_KEYS:
        v = obj.get(k)
        if isinstance(v, (int, float)):
            val += float(v)
    return val


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def to_items(rows: list[dict[str, Any]], allowed_sources: set[str]) -> list[Item]:
    out = []
    for r in rows:
        dt = parse_dt(r)
        if not dt:
            continue
        src = pick_source(r)
        if allowed_sources and src not in allowed_sources:
            continue
        text = pick_text(r)
        if not text:
            continue
        out.append(
            Item(
                source=src,
                text=text,
                url=pick_url(r),
                dt=dt,
                engagement=pick_engagement(r),
            )
        )
    return out


def keyword_bucket(items: list[Item], keywords: list[str]) -> dict[str, list[Item]]:
    buckets = defaultdict(list)
    for it in items:
        low = it.text.lower()
        for kw in keywords:
            if kw.lower() in low:
                buckets[kw].append(it)
    return dict(buckets)


def issue_score(items: list[Item], now: datetime) -> float:
    if not items:
        return 0.0
    score = 0.0
    for i in items:
        age_h = max((now - i.dt).total_seconds() / 3600, 1)
        recency = 1.0 / (1.0 + age_h / 24)
        score += (1.0 + i.engagement / 100.0) * recency
    return round(score, 3)


def top_links(items: list[Item], limit=3):
    scored = sorted(items, key=lambda x: x.engagement, reverse=True)
    links = []
    seen = set()
    for i in scored:
        if not i.url or i.url in seen:
            continue
        links.append({"url": i.url, "source": i.source, "title": i.text.split("|")[0][:90]})
        seen.add(i.url)
        if len(links) >= limit:
            break
    return links


def action_pack(keyword: str) -> dict[str, str]:
    return {
        "content": f"'{keyword}' 관련해서 최근 반응 사례 3개를 비교한 짧은 인사이트 포스트 작성",
        "product": f"'{keyword}' 문제를 10분 내 해결하는 마이크로 기능/템플릿 가설 1개 실험",
        "comment": f"'{keyword}' 관련 인기 글에 공감 + 경험 + 간단한 해결 팁 3줄 댓글 작성",
    }


def build_cards(items: list[Item], keywords: list[str], now: datetime, window_days: int) -> list[dict[str, Any]]:
    since = now - timedelta(days=window_days)
    sliced = [i for i in items if i.dt >= since]
    buckets = keyword_bucket(sliced, keywords)

    cards = []
    for kw, bucket in buckets.items():
        sc = issue_score(bucket, now)
        if sc <= 0:
            continue
        cards.append(
            {
                "title": f"{kw}: 지금 반응이 빠르게 올라오는 이슈",
                "why_trending": [
                    f"최근 {window_days}일 내 언급/반응이 반복적으로 관찰됨",
                    f"관련 게시물의 참여(좋아요/댓글/업보트) 합계가 높음",
                ],
                "top_links": top_links(bucket, 3),
                "what_to_do": action_pack(kw),
                "score": sc,
                "evidence_count": len(bucket),
            }
        )

    cards = sorted(cards, key=lambda c: c["score"], reverse=True)[:5]
    return cards


def main():
    ap = argparse.ArgumentParser(description="Issue Radar (today/week/month) JSON generator")
    ap.add_argument("--category", required=True)
    ap.add_argument("--keywords", required=True, help="comma-separated, e.g. 솔로 창업,솔로 개발,노마드")
    ap.add_argument("--sources", default="reddit,threads", help="comma-separated")
    ap.add_argument("--in", dest="input_jsonl", required=True, help="input jsonl path")
    ap.add_argument("--out", dest="output_json", required=True, help="output json path")
    args = ap.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]

    rows = load_jsonl(Path(args.input_jsonl))
    items = to_items(rows, set(sources))
    now = datetime.now(timezone.utc)

    out = {
        "input": {
            "category": args.category,
            "keywords": keywords,
            "sources": sources,
        },
        "generated_at": now.isoformat(),
        "tabs": {
            "today": build_cards(items, keywords, now, window_days=1),
            "week": build_cards(items, keywords, now, window_days=7),
            "month": build_cards(items, keywords, now, window_days=30),
        },
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote {args.output_json}")


if __name__ == "__main__":
    main()
