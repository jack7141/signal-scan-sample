#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Set
import xml.etree.ElementTree as ET

import requests
from ddgs import DDGS

TAB_TO_REDDIT_T = {"today": "day", "week": "week", "month": "month"}
TAB_TO_DDGS_TIMELIMIT = {"today": "d", "week": "w", "month": "m"}

FALLBACK_SUBREDDITS = {
    "마케팅": ["marketing", "Entrepreneur", "solopreneur", "SideProject", "startups", "digitalnomad", "webdev", "SaaS"]
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def keyword_match_score(text: str, keywords: List[str]) -> int:
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


def parse_reddit_rss(xml_text: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_text)
    channel = root.find("channel")
    if channel is None:
        return []

    items = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        items.append({"title": title, "url": link, "published": pub_date})
    return items


@dataclass
class IssueCard:
    source: str
    time_window: str
    title: str
    url: str
    subreddit: Optional[str] = None
    match_score: int = 0
    published: Optional[str] = None
    keywords_hit: Optional[List[str]] = None
    via: Optional[str] = None


def load_input_from_stdin() -> Dict[str, Any]:
    try:
        raw = input().strip()
    except EOFError:
        raw = ""
    if raw:
        return json.loads(raw)
    return {
        "category": "마케팅",
        "keywords": ["솔로 창업", "솔로 개발", "노마드"],
        "sources": ["reddit", "threads"],
    }


def extract_subreddit_from_url(url: str) -> Optional[str]:
    m = re.search(r"reddit\.com/r/([A-Za-z0-9_]+)/?", url)
    return m.group(1) if m else None


def ddgs_search(query: str, timelimit: str, max_results: int = 10) -> List[Dict[str, str]]:
    out = []
    with DDGS(timeout=10) as ddgs:
        for r in ddgs.text(query, timelimit=timelimit, max_results=max_results):
            out.append({
                "title": r.get("title") or "",
                "url": r.get("href") or "",
                "snippet": r.get("body") or "",
            })
    return out


def recommend_subreddits(category: str, keywords: List[str], max_subreddits: int = 10) -> List[str]:
    keyword_en = {
        "솔로 창업": ["solopreneur", "solo founder", "indie business"],
        "솔로 개발": ["solo developer", "indie hacker", "build in public"],
        "노마드": ["digital nomad", "remote work"],
    }
    tokens = []
    for k in keywords:
        tokens.append(k)
        tokens.extend(keyword_en.get(k, []))
    tokens = tokens[:7]
    token_q = " OR ".join([f'"{t}"' for t in tokens if t])

    queries = [
        f"site:reddit.com/r {token_q} best subreddits",
        f"site:reddit.com/r {token_q} community",
        f"relevant subreddits {token_q}",
    ]

    found: List[str] = []
    seen: Set[str] = set()
    for q in queries:
        try:
            results = ddgs_search(q, timelimit="m", max_results=15)
        except Exception:
            continue
        for r in results:
            sr = extract_subreddit_from_url(r["url"])
            if sr and sr.lower() not in seen:
                seen.add(sr.lower())
                found.append(sr)
            if len(found) >= max_subreddits:
                return found

    return FALLBACK_SUBREDDITS.get(category, [])[:max_subreddits]


def resolve_subreddits(user_input: Dict[str, Any]) -> List[str]:
    category = user_input.get("category", "")
    keywords = user_input.get("keywords", [])
    reddit_cfg = user_input.get("reddit", {}) if isinstance(user_input.get("reddit"), dict) else {}

    subs = reddit_cfg.get("subreddits")
    if isinstance(subs, list) and subs:
        return [s.strip() for s in subs if isinstance(s, str) and s.strip()]

    max_subs = reddit_cfg.get("max_subreddits", 10)
    try:
        max_subs = int(max_subs)
    except Exception:
        max_subs = 10

    rec = recommend_subreddits(category, keywords, max_subreddits=max_subs)
    if rec:
        return rec

    return FALLBACK_SUBREDDITS.get(category, [])[:max_subs]


def fetch_reddit_top_rss(subreddit: str, t: str, timeout: int = 15) -> List[Dict[str, Any]]:
    url = f"https://www.reddit.com/r/{subreddit}/top.rss?t={t}"
    headers = {"User-Agent": "IssuePicker/0.3 (dynamic input/subreddits)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return parse_reddit_rss(r.text)


def pick_reddit_rss_cards(subreddits: List[str], keywords: List[str], tab: str, per_sub_limit: int = 10) -> List[IssueCard]:
    t = TAB_TO_REDDIT_T[tab]
    cards: List[IssueCard] = []
    for sr in subreddits:
        try:
            items = fetch_reddit_top_rss(sr, t=t)
        except Exception:
            continue
        for it in items[:per_sub_limit]:
            title = it.get("title", "")
            url = it.get("url", "")
            score = keyword_match_score(title, keywords)
            if score <= 0:
                continue
            hits = [kw for kw in keywords if keyword_match_score(title, [kw]) > 0]
            cards.append(IssueCard(
                source="reddit",
                time_window=tab,
                title=title,
                url=url,
                subreddit=sr,
                match_score=score,
                published=it.get("published"),
                keywords_hit=hits,
                via="rss",
            ))
    return cards


def pick_threads_cards_ddgs(keywords: List[str], tab: str, max_results: int = 10) -> List[IssueCard]:
    timelimit = TAB_TO_DDGS_TIMELIMIT[tab]
    kw_query = " OR ".join([f'"{k}"' for k in keywords])
    q = f"site:threads.net ({kw_query})"

    cards: List[IssueCard] = []
    try:
        results = ddgs_search(q, timelimit=timelimit, max_results=max_results)
    except Exception:
        return []

    for r in results:
        title = r["title"]
        url = r["url"]
        text = f"{title} {r.get('snippet','')}"
        score = keyword_match_score(text, keywords)
        if score <= 0:
            continue
        hits = [kw for kw in keywords if keyword_match_score(text, [kw]) > 0]
        cards.append(IssueCard(
            source="threads",
            time_window=tab,
            title=title,
            url=url,
            match_score=score,
            keywords_hit=hits,
            via="ddgs",
        ))
    return cards


def dedupe_and_rank(cards: List[IssueCard], limit: int = 10) -> List[IssueCard]:
    seen = set()
    uniq = []
    for c in cards:
        if not c.url or c.url in seen:
            continue
        seen.add(c.url)
        uniq.append(c)
    uniq.sort(key=lambda x: (x.match_score, 1 if x.via == "rss" else 0), reverse=True)
    return uniq[:limit]


def generate_payload(user_input: Dict[str, Any]) -> Dict[str, Any]:
    category = user_input.get("category", "")
    keywords = user_input.get("keywords", [])
    sources = user_input.get("sources", [])
    subreddits = resolve_subreddits(user_input)

    payload: Dict[str, Any] = {
        "input": {
            "category": category,
            "keywords": keywords,
            "sources": sources,
            "reddit": {"subreddits": subreddits},
        },
        "generated_at": now_utc_iso(),
        "tabs": {"today": [], "week": [], "month": []},
    }

    for tab in ["today", "week", "month"]:
        cards: List[IssueCard] = []
        if "reddit" in sources:
            cards.extend(pick_reddit_rss_cards(subreddits, keywords, tab))
        if "threads" in sources:
            cards.extend(pick_threads_cards_ddgs(keywords, tab))
        payload["tabs"][tab] = [asdict(c) for c in dedupe_and_rank(cards, limit=10)]

    return payload


if __name__ == "__main__":
    ui = load_input_from_stdin()
    result = generate_payload(ui)
    print(json.dumps(result, ensure_ascii=False, indent=2))
