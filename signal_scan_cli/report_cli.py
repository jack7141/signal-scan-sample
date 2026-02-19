#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

import requests
try:
    from ddgs import DDGS
except Exception:
    from duckduckgo_search import DDGS

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from google_play_scraper import search as gp_search, reviews as gp_reviews
    from google_play_scraper.constants import Sort as GPSort
except Exception:
    gp_search = None
    gp_reviews = None
    GPSort = None

try:
    from app_store_scraper import AppStore
except Exception:
    AppStore = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# Models
# -----------------------------

@dataclass
class EvidenceCard:
    source: str
    source_url: str
    title: str
    quote: str
    pain_tags: List[str]
    wtp: int  # 0/1/2
    workaround_tags: List[str]
    meta: Dict[str, Any]


# -----------------------------
# Heuristics
# -----------------------------

PAIN_KEYWORDS = {
    "time": ["시간", "느리", "오래", "delay", "slow", "time-consuming"],
    "money": ["비싸", "돈", "cost", "expensive", "budget"],
    "stress": ["스트레스", "짜증", "frustrat", "pain", "annoy"],
    "risk": ["리스크", "위험", "불안", "risk", "uncertain"],
    "repeat": ["반복", "매번", "수작업", "manual", "again and again"],
}

WORKAROUND_KEYWORDS = {
    "excel": ["excel", "엑셀", "spreadsheet"],
    "notion": ["notion", "노션"],
    "outsource": ["외주", "agency", "freelancer"],
    "manual": ["수작업", "manually", "by hand"],
    "chatgpt": ["chatgpt", "gpt", "llm"],
    "other_tool": ["tool", "saas", "software", "app"],
}

WTP_LEVEL_2 = [
    "돈 내", "pay for", "paid for", "already paying", "구독 중", "결제", "buy this", "지불할 의향", "낼 의향", "구매 의향",
]
WTP_LEVEL_1 = [
    "가격", "price", "pricing", "얼마", "worth it", "유료", "cost", "의향", "pay", "결제할",
]

RELEVANCE_TERMS = [
    "매매일지", "매매 일지", "트레이딩일지", "트레이딩 일지", "트레이더 저널",
    "trading journal", "trade journal", "trade log", "trading log", "journaling",
    "매매기록", "매매 기록", "trading notes", "거래기록",
    "trading performance", "trading review", "pnl tracker", "rule violation",
    "review my trades", "journal app", "trading mistakes",
]

PROBLEM_TERMS = [
    "노션", "notion", "엑셀", "excel", "spreadsheet", "수작업", "manual",
    "동기화", "sync", "대시보드", "dashboard", "복기", "매매복기",
]

BAD_URL_KEYWORDS = [
    "lotto", "arclink", "games-cn", "meme", "horror", "/r/ssss", "clickbait",
]

REDDIT_ALLOWED_SUBS = [
    "trading", "daytrading", "stocks", "stockmarket", "investing", "options", "forex", "crypto", "cryptocurrency", "algotrading",
]

PREFERRED_DOMAIN_HINTS = [
    "naver.com", "tistory.com", "brunch.co.kr", "velog.io", "youtube.com", "reddit.com", "threads.net"
]


def _contains_any(text: str, terms: List[str]) -> bool:
    t = text.lower()
    return any(term.lower() in t for term in terms)


def infer_pain_tags(text: str) -> List[str]:
    tags = []
    for k, terms in PAIN_KEYWORDS.items():
        if _contains_any(text, terms):
            tags.append(k)
    return tags


def infer_workarounds(text: str) -> List[str]:
    tags = []
    for k, terms in WORKAROUND_KEYWORDS.items():
        if _contains_any(text, terms):
            tags.append(k)
    return tags


def infer_wtp(text: str) -> int:
    if _contains_any(text, WTP_LEVEL_2):
        return 2
    if _contains_any(text, WTP_LEVEL_1):
        return 1
    return 0


def compact_text(s: str, max_len: int = 240) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len] + ("…" if len(s) > max_len else "")


def relevance_score(text: str, extra_terms: Optional[List[str]] = None) -> int:
    low = (text or "").lower()
    s = 0
    for t in RELEVANCE_TERMS:
        if t.lower() in low:
            s += 2
    for t in PROBLEM_TERMS:
        if t.lower() in low:
            s += 1
    for t in (extra_terms or []):
        tt = (t or "").strip().lower()
        if len(tt) >= 2 and tt in low:
            s += 2
    return s


def build_dynamic_terms(intake: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    for x in [intake.get("target", ""), intake.get("problem", ""), intake.get("current_alternative", "")]:
        if x:
            terms.extend(re.split(r"[\s,/]+", str(x)))
    for k in (intake.get("keywords") or []):
        terms.extend(re.split(r"[\s,/]+", str(k)))
    # dedup + sane length
    out = []
    seen = set()
    for t in terms:
        t = t.strip()
        if len(t) < 2:
            continue
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return out[:30]


KR_TO_EN_HINTS = {
    "매매일지": "trading journal",
    "트레이딩": "trading",
    "복기": "trade review",
    "규칙 위반": "rule violation",
    "손익": "profit and loss",
    "대시보드": "dashboard",
    "자동 기록": "auto logging",
    "트레이더": "trader",
    "엑셀": "excel",
    "노션": "notion",
}


def build_reddit_english_queries(intake: Dict[str, Any], plan: Dict[str, Any]) -> List[str]:
    seed = []
    seed.extend((plan.get("reddit_queries") or [])[:4])
    seed.extend((intake.get("keywords") or [])[:8])
    seed.extend([str(intake.get("target") or ""), str(intake.get("problem") or "")])

    english = [
        "trading journal app",
        "trade log app",
        "trading performance tracker",
        "trading review workflow",
        "willingness to pay trading app",
        "how traders track mistakes",
        "best trading journal app",
    ]

    for s in seed:
        ss = (s or "").strip()
        if not ss:
            continue
        low = ss.lower()
        # keep english-only query as-is (drop mixed KR+EN noise)
        if re.search(r"[a-z]", low):
            ascii_ratio = sum(1 for ch in low if ord(ch) < 128) / max(1, len(low))
            if ascii_ratio >= 0.9:
                english.append(low)
        # add mapped hints for korean terms
        for kr, en in KR_TO_EN_HINTS.items():
            if kr in ss:
                english.append(en)
                english.append(f"{en} app")
                english.append(f"{en} review")
                english.append(f"best {en} app")
                english.append(f"{en} alternatives")
                english.append(f"{en} willingness to pay")

    out = []
    seen = set()
    for q in english:
        qq = re.sub(r"\s+", " ", q).strip()
        if len(qq) < 3:
            continue
        # avoid overly generic one-word queries
        if len(qq.split()) == 1:
            continue
        if not any(k in qq for k in ["trading", "trade", "journal", "log", "review", "tracker"]):
            continue
        if qq in seen:
            continue
        seen.add(qq)
        out.append(qq)
    return out[:10]


def is_relevant_text(text: str, threshold: int = 2, extra_terms: Optional[List[str]] = None) -> bool:
    return relevance_score(text, extra_terms=extra_terms) >= threshold


# -----------------------------
# Data sources
# -----------------------------

def build_query_plan(intake: Dict[str, Any]) -> Dict[str, Any]:
    kws = intake.get("keywords", []) or []
    target = intake.get("target", "")
    problem = intake.get("problem", "")
    alt = intake.get("current_alternative", "")
    base_terms = [target, problem, alt] + kws
    base_terms = [x.strip() for x in base_terms if x and x.strip()]

    reddit_queries = []
    ddg_queries = []
    naver_queries = []
    ph_queries = []

    for term in base_terms[:8]:
        reddit_queries.extend([
            f"{term} validation",
            f"{term} pain point",
            f"{term} willing to pay",
        ])

        # generic web queries (not tied to reddit/x only)
        ddg_queries.extend([
            f"{term} 불편",
            f"{term} 문제점",
            f"{term} 후기",
            f"{term} alternatives",
            f"{term} workflow",
            f"{term} template",
            f"site:threads.net {term}",
            f"site:reddit.com {term}",
        ])

        # naver-friendly KR intent queries
        naver_queries.extend([
            f"{term} 매매일지",
            f"{term} 엑셀 노션",
            f"{term} 자동화",
            f"{term} 유료",
            f"{term} 가격",
            f"{term} 후기",
        ])
        ph_queries.append(term)

    subreddit_candidates = [
        "startups", "Entrepreneur", "SaaS", "nocode", "smallbusiness", "sideproject"
    ]

    return {
        "reddit_queries": list(dict.fromkeys(reddit_queries))[:12],
        "ddg_queries": list(dict.fromkeys(ddg_queries))[:18],
        "naver_queries": list(dict.fromkeys(naver_queries))[:18],
        "producthunt_queries": list(dict.fromkeys(ph_queries))[:8],
        "subreddit_candidates": subreddit_candidates,
    }


def fetch_producthunt(access_token: str, search_term: str, first: int = 5) -> List[EvidenceCard]:
    endpoint = "https://api.producthunt.com/v2/api/graphql"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    # 1) Find related topics by query
    topic_query = """
    query TopicSearch($term: String!, $first: Int!) {
      topics(query: $term, first: $first) {
        edges {
          node {
            id
            name
            slug
            followersCount
            url
          }
        }
      }
    }
    """
    topic_resp = requests.post(
        endpoint,
        headers=headers,
        json={"query": topic_query, "variables": {"term": search_term, "first": 3}},
        timeout=20,
    )
    topic_resp.raise_for_status()
    topic_data = topic_resp.json()

    topic_edges = (((topic_data or {}).get("data") or {}).get("topics") or {}).get("edges", [])
    topic_slugs = [((e or {}).get("node") or {}).get("slug") for e in topic_edges]
    topic_slugs = [s for s in topic_slugs if s]

    # 2) Pull posts for each topic slug
    posts_query = """
    query PostsByTopic($topic: String!, $first: Int!) {
      posts(topic: $topic, first: $first, order: RANKING) {
        edges {
          node {
            id
            name
            tagline
            description
            votesCount
            url
            createdAt
            topics(first: 5) {
              edges {
                node {
                  name
                  slug
                }
              }
            }
          }
        }
      }
    }
    """

    cards: List[EvidenceCard] = []

    # topic-based fetch
    for slug in topic_slugs[:2]:
        r = requests.post(
            endpoint,
            headers=headers,
            json={"query": posts_query, "variables": {"topic": slug, "first": first}},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("errors"):
            continue
        edges = (((data or {}).get("data") or {}).get("posts") or {}).get("edges", [])
        for e in edges:
            node = (e or {}).get("node") or {}
            text_blob = " ".join([
                str(node.get("name", "")),
                str(node.get("tagline", "")),
                str(node.get("description", "")),
            ])
            cards.append(EvidenceCard(
                source="producthunt",
                source_url=node.get("url", "https://www.producthunt.com"),
                title=node.get("name", "Product Hunt Post"),
                quote=compact_text(text_blob),
                pain_tags=infer_pain_tags(text_blob),
                wtp=infer_wtp(text_blob),
                workaround_tags=infer_workarounds(text_blob),
                meta={
                    "votesCount": node.get("votesCount"),
                    "topic": slug,
                    "createdAt": node.get("createdAt"),
                },
            ))

    # 3) fallback: recent ranked posts then soft keyword filter (when topic search is sparse)
    fallback_query = """
    query FallbackPosts($first: Int!) {
      posts(first: $first, order: RANKING) {
        edges {
          node {
            id
            name
            tagline
            description
            votesCount
            url
            createdAt
          }
        }
      }
    }
    """
    r = requests.post(endpoint, headers=headers, json={"query": fallback_query, "variables": {"first": 20}}, timeout=20)
    r.raise_for_status()
    data = r.json()
    edges = (((data or {}).get("data") or {}).get("posts") or {}).get("edges", [])
    term_l = search_term.lower().strip()
    tokens = [t for t in re.split(r"\s+", term_l) if len(t) >= 2]
    fallback_added = 0
    for e in edges:
        node = (e or {}).get("node") or {}
        text_blob = " ".join([
            str(node.get("name", "")),
            str(node.get("tagline", "")),
            str(node.get("description", "")),
        ])
        low = text_blob.lower()
        match = True
        if term_l:
            if tokens:
                match = any(t in low for t in tokens)
            else:
                match = term_l in low

        # soft filter: if no keyword match, still keep a few top posts as market baseline evidence
        if not match and fallback_added >= 3:
            continue

        cards.append(EvidenceCard(
            source="producthunt",
            source_url=node.get("url", "https://www.producthunt.com"),
            title=node.get("name", "Product Hunt Post"),
            quote=compact_text(text_blob),
            pain_tags=infer_pain_tags(text_blob),
            wtp=infer_wtp(text_blob),
            workaround_tags=infer_workarounds(text_blob),
            meta={"votesCount": node.get("votesCount"), "createdAt": node.get("createdAt"), "fallback": True, "matched": match},
        ))
        fallback_added += 1
        if fallback_added >= first:
            break

    return cards


def fetch_ddg(search_term: str, max_results: int = 8) -> List[EvidenceCard]:
    # DuckDuckGo Instant Answer API (lightweight fallback)
    # NOTE: This endpoint is not full web-search. It often returns sparse data.
    url = "https://api.duckduckgo.com/"
    params = {
        "q": search_term,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()

    cards: List[EvidenceCard] = []

    abstract = j.get("AbstractText") or ""
    if abstract:
        cards.append(EvidenceCard(
            source="duckduckgo",
            source_url=j.get("AbstractURL") or "https://duckduckgo.com",
            title=j.get("Heading") or search_term,
            quote=compact_text(abstract),
            pain_tags=infer_pain_tags(abstract),
            wtp=infer_wtp(abstract),
            workaround_tags=infer_workarounds(abstract),
            meta={"kind": "abstract"},
        ))

    related = j.get("RelatedTopics") or []
    for item in related[:max_results]:
        if isinstance(item, dict) and item.get("Text"):
            text = item.get("Text", "")
            cards.append(EvidenceCard(
                source="duckduckgo",
                source_url=item.get("FirstURL", "https://duckduckgo.com"),
                title=search_term,
                quote=compact_text(text),
                pain_tags=infer_pain_tags(text),
                wtp=infer_wtp(text),
                workaround_tags=infer_workarounds(text),
                meta={"kind": "related_topic"},
            ))
    return cards


def fetch_ddg_search(query: str, max_results: int = 10) -> List[EvidenceCard]:
    """
    Uses duckduckgo-search package for real web search results.
    This is the primary DDG source (better than Instant Answer API for evidence collection).
    """
    cards: List[EvidenceCard] = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        for item in results or []:
            title = item.get("title") or query
            body = item.get("body") or ""
            href = item.get("href") or "https://duckduckgo.com"
            text_blob = f"{title} {body}".strip()

            low_url = href.lower()
            # hard noise filter
            if any(k in low_url for k in BAD_URL_KEYWORDS):
                continue

            # if query is generic, prefer known content domains
            if not any(h in low_url for h in PREFERRED_DOMAIN_HINTS):
                if len(text_blob) < 30:
                    continue

            src = "duckduckgo"
            if "threads.net" in href:
                src = "threads"
            elif "reddit.com" in href:
                src = "reddit-web"

            cards.append(EvidenceCard(
                source=src,
                source_url=href,
                title=title,
                quote=compact_text(text_blob),
                pain_tags=infer_pain_tags(text_blob),
                wtp=infer_wtp(text_blob),
                workaround_tags=infer_workarounds(text_blob),
                meta={"provider": "duckduckgo-search", "query": query},
            ))
    return cards


def fetch_youtube_search_and_comments(api_key: str, query: str, max_videos: int = 4, max_comments: int = 10) -> List[EvidenceCard]:
    cards: List[EvidenceCard] = []

    s_url = "https://www.googleapis.com/youtube/v3/search"
    s_params = {
        "key": api_key,
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_videos,
        "relevanceLanguage": "ko",
    }
    sr = requests.get(s_url, params=s_params, timeout=20)
    sr.raise_for_status()
    items = sr.json().get("items", [])

    for it in items:
        vid = ((it.get("id") or {}).get("videoId") or "").strip()
        sn = it.get("snippet") or {}
        title = sn.get("title", "")
        desc = sn.get("description", "")
        if not vid:
            continue

        video_url = f"https://www.youtube.com/watch?v={vid}"
        v_blob = f"{title} {desc}".strip()
        cards.append(EvidenceCard(
            source="youtube-video",
            source_url=video_url,
            title=title or query,
            quote=compact_text(v_blob),
            pain_tags=infer_pain_tags(v_blob),
            wtp=infer_wtp(v_blob),
            workaround_tags=infer_workarounds(v_blob),
            meta={"provider": "youtube-data-api", "query": query, "videoId": vid},
        ))

        c_url = "https://www.googleapis.com/youtube/v3/commentThreads"
        c_params = {
            "key": api_key,
            "part": "snippet",
            "videoId": vid,
            "maxResults": max_comments,
            "order": "relevance",
            "textFormat": "plainText",
        }
        try:
            cr = requests.get(c_url, params=c_params, timeout=20)
            cr.raise_for_status()
            c_items = cr.json().get("items", [])
            for ci in c_items:
                top = (((ci.get("snippet") or {}).get("topLevelComment") or {}).get("snippet") or {})
                text = top.get("textDisplay", "") or top.get("textOriginal", "")
                if not text:
                    continue
                cards.append(EvidenceCard(
                    source="youtube-comment",
                    source_url=video_url,
                    title=title or "YouTube comment",
                    quote=compact_text(text),
                    pain_tags=infer_pain_tags(text),
                    wtp=infer_wtp(text),
                    workaround_tags=infer_workarounds(text),
                    meta={"provider": "youtube-data-api", "query": query, "videoId": vid},
                ))
        except Exception as e:
            print(f"[warn] youtube comments failed for video {vid}: {e}", file=sys.stderr)

    return cards


def fetch_hn_algolia_search(query: str, max_hits: int = 20) -> List[EvidenceCard]:
    """
    Hacker News search via Algolia public API (no key required).
    Useful until official HN API query endpoints are added.
    """
    cards: List[EvidenceCard] = []
    url = "https://hn.algolia.com/api/v1/search"
    params = {
        "query": query,
        "tags": "story,comment",
        "hitsPerPage": max_hits,
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    hits = r.json().get("hits", [])

    for h in hits:
        title = h.get("title") or h.get("story_title") or query
        text = h.get("comment_text") or h.get("story_text") or ""
        object_id = h.get("objectID")
        story_id = h.get("story_id") or object_id
        source_url = h.get("url") or (f"https://news.ycombinator.com/item?id={story_id}" if story_id else "https://news.ycombinator.com")
        text_blob = f"{title} {re.sub(r'<[^>]+>', ' ', text)}".strip()

        cards.append(EvidenceCard(
            source="hackernews",
            source_url=source_url,
            title=title,
            quote=compact_text(text_blob),
            pain_tags=infer_pain_tags(text_blob),
            wtp=infer_wtp(text_blob),
            workaround_tags=infer_workarounds(text_blob),
            meta={
                "provider": "hn-algolia",
                "query": query,
                "points": h.get("points"),
                "num_comments": h.get("num_comments"),
                "created_at": h.get("created_at"),
            },
        ))
    return cards


def fetch_naver_search(client_id: str, client_secret: str, query: str, display: int = 10) -> List[EvidenceCard]:
    """
    Naver OpenAPI Search (news + blog) collector.
    """
    cards: List[EvidenceCard] = []
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }

    for t, endpoint in [
        ("news", "https://openapi.naver.com/v1/search/news.json"),
        ("blog", "https://openapi.naver.com/v1/search/blog.json"),
    ]:
        try:
            r = requests.get(endpoint, headers=headers, params={"query": query, "display": display, "sort": "sim"}, timeout=20)
            r.raise_for_status()
            data = r.json()
            items = data.get("items", [])
            for it in items:
                title = re.sub(r"<[^>]+>", "", it.get("title", "")).strip()
                desc = re.sub(r"<[^>]+>", "", it.get("description", "")).strip()
                link = it.get("originallink") or it.get("link") or "https://search.naver.com"
                text_blob = f"{title} {desc}".strip()
                cards.append(EvidenceCard(
                    source=f"naver-{t}",
                    source_url=link,
                    title=title or query,
                    quote=compact_text(text_blob),
                    pain_tags=infer_pain_tags(text_blob),
                    wtp=infer_wtp(text_blob),
                    workaround_tags=infer_workarounds(text_blob),
                    meta={"provider": "naver-openapi", "query": query, "kind": t, "pubDate": it.get("pubDate")},
                ))
        except Exception as e:
            print(f"[warn] naver-{t} failed for '{query}': {e}", file=sys.stderr)

    return cards


def fetch_google_play_search_and_reviews(query: str, lang: str = "ko", country: str = "kr", max_apps: int = 5, max_reviews: int = 20) -> List[EvidenceCard]:
    cards: List[EvidenceCard] = []
    if gp_search is None or gp_reviews is None:
        return cards
    try:
        apps = gp_search(query, n_hits=max_apps, lang=lang, country=country) or []
        for app in apps[:max_apps]:
            title = str(app.get("title") or query)
            app_id = str(app.get("appId") or "")
            summary = str(app.get("summary") or "")
            score = app.get("score")
            url = app.get("url") or (f"https://play.google.com/store/apps/details?id={app_id}" if app_id else "https://play.google.com")
            blob = f"{title} {summary}"
            cards.append(EvidenceCard(
                source="googleplay-app",
                source_url=url,
                title=title,
                quote=compact_text(blob),
                pain_tags=infer_pain_tags(blob),
                wtp=infer_wtp(blob),
                workaround_tags=infer_workarounds(blob),
                meta={"provider": "google-play-scraper", "query": query, "score": score, "appId": app_id},
            ))
            if not app_id:
                continue
            try:
                rv, _ = gp_reviews(
                    app_id,
                    lang=lang,
                    country=country,
                    sort=GPSort.NEWEST if GPSort else None,
                    count=max_reviews,
                )
                for r in rv[:max_reviews]:
                    content = str(r.get("content") or "")
                    if not content:
                        continue
                    cards.append(EvidenceCard(
                        source="googleplay-review",
                        source_url=url,
                        title=title,
                        quote=compact_text(content, 320),
                        pain_tags=infer_pain_tags(content),
                        wtp=infer_wtp(content),
                        workaround_tags=infer_workarounds(content),
                        meta={
                            "provider": "google-play-scraper",
                            "query": query,
                            "appId": app_id,
                            "score": r.get("score"),
                            "at": str(r.get("at") or ""),
                        },
                    ))
            except Exception as e:
                print(f"[warn] googleplay reviews failed for {app_id}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[warn] googleplay search failed for '{query}': {e}", file=sys.stderr)
    return cards


def fetch_appstore_reviews(targets: List[Dict[str, Any]], review_count: int = 20) -> List[EvidenceCard]:
    cards: List[EvidenceCard] = []
    if AppStore is None:
        return cards
    for t in targets[:5]:
        try:
            country = str(t.get("country") or "kr")
            app_name = str(t.get("app_name") or "")
            app_id = t.get("app_id")
            if not app_name:
                continue
            app = AppStore(country=country, app_name=app_name, app_id=app_id)
            app.review(how_many=review_count)
            url = f"https://apps.apple.com/{country}/app/{app_name}/id{app.app_id}" if getattr(app, 'app_id', None) else "https://apps.apple.com"
            for r in (app.reviews or [])[:review_count]:
                text = str(r.get("review") or r.get("title") or "")
                if not text:
                    continue
                cards.append(EvidenceCard(
                    source="appstore-review",
                    source_url=url,
                    title=app_name,
                    quote=compact_text(text, 320),
                    pain_tags=infer_pain_tags(text),
                    wtp=infer_wtp(text),
                    workaround_tags=infer_workarounds(text),
                    meta={
                        "provider": "app-store-scraper",
                        "country": country,
                        "app_name": app_name,
                        "app_id": getattr(app, 'app_id', None),
                        "rating": r.get("rating"),
                        "date": r.get("date"),
                    },
                ))
        except Exception as e:
            print(f"[warn] appstore scrape failed for target={t}: {e}", file=sys.stderr)
    return cards


def fetch_reddit_oauth(token: str, query: str, limit: int = 10) -> List[EvidenceCard]:
    # Uses OAuth endpoint. Requires valid Reddit OAuth token with read scope.
    url = "https://oauth.reddit.com/search"
    headers = {
        "Authorization": f"bearer {token}",
        "User-Agent": "signalscan-cli/0.1 (by /u/yourusername)",
    }
    params = {
        "q": query,
        "limit": limit,
        "sort": "relevance",
        "type": "link",
    }
    r = requests.get(url, headers=headers, params=params, timeout=20)
    if r.status_code == 401:
        raise RuntimeError("Reddit token unauthorized (401). Check REDDIT_ACCESS_TOKEN and scope.")
    r.raise_for_status()
    data = r.json()

    cards: List[EvidenceCard] = []
    children = (((data or {}).get("data") or {}).get("children") or [])
    for c in children:
        d = (c or {}).get("data") or {}
        text = " ".join([d.get("title", ""), d.get("selftext", "")])
        permalink = d.get("permalink", "")
        full_url = f"https://reddit.com{permalink}" if permalink else d.get("url", "https://reddit.com")
        cards.append(EvidenceCard(
            source="reddit",
            source_url=full_url,
            title=d.get("title", "Reddit post"),
            quote=compact_text(text),
            pain_tags=infer_pain_tags(text),
            wtp=infer_wtp(text),
            workaround_tags=infer_workarounds(text),
            meta={"subreddit": d.get("subreddit"), "score": d.get("score")},
        ))
    return cards


def fetch_apify_task(token: str, task_id: str, run_input: Dict[str, Any], limit: int = 30) -> List[Dict[str, Any]]:
    base = "https://api.apify.com/v2"
    run_url = f"{base}/actor-tasks/{task_id}/runs"
    r = requests.post(run_url, params={"token": token}, json=run_input, timeout=60)
    r.raise_for_status()
    run = ((r.json() or {}).get("data") or {})
    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        return []
    items_url = f"{base}/datasets/{dataset_id}/items"
    ir = requests.get(items_url, params={"token": token, "limit": limit, "clean": 1}, timeout=60)
    ir.raise_for_status()
    data = ir.json()
    return data if isinstance(data, list) else []


def fetch_apify_actor(token: str, actor_id: str, run_input: Dict[str, Any], limit: int = 30) -> List[Dict[str, Any]]:
    base = "https://api.apify.com/v2"
    run_url = f"{base}/acts/{actor_id}/runs"
    r = requests.post(run_url, params={"token": token}, json=run_input, timeout=60)
    r.raise_for_status()
    run = ((r.json() or {}).get("data") or {})
    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        return []
    items_url = f"{base}/datasets/{dataset_id}/items"
    ir = requests.get(items_url, params={"token": token, "limit": limit, "clean": 1}, timeout=60)
    ir.raise_for_status()
    data = ir.json()
    return data if isinstance(data, list) else []


def normalize_apify_items(items: List[Dict[str, Any]], source: str, query: str) -> List[EvidenceCard]:
    cards: List[EvidenceCard] = []
    for it in items:
        title = str(it.get("title") or it.get("name") or query)
        text = str(it.get("text") or it.get("description") or it.get("body") or it.get("content") or "")
        url = str(it.get("url") or it.get("link") or "https://apify.com")
        blob = f"{title} {text}".strip()
        if not blob:
            continue
        cards.append(EvidenceCard(
            source=f"apify-{source}",
            source_url=url,
            title=title,
            quote=compact_text(blob, 320),
            pain_tags=infer_pain_tags(blob),
            wtp=infer_wtp(blob),
            workaround_tags=infer_workarounds(blob),
            meta={
                "provider": "apify",
                "query": query,
                "source": source,
                "author": it.get("author") or it.get("username"),
                "publishedAt": it.get("publishedAt") or it.get("date") or it.get("timestamp"),
            },
        ))
    return cards


# -----------------------------
# Scoring
# -----------------------------

def score_report(cards: List[EvidenceCard]) -> Dict[str, Any]:
    if not cards:
        return {
            "demand_score": 0,
            "confidence": "low",
            "decision": "iterate",
            "stats": {},
        }

    def source_weight(src: str) -> float:
        s = (src or "").lower()
        if s.startswith("naver") or s.startswith("apify-naver"):
            return 1.25
        if s.startswith("youtube") or s.startswith("apify-youtube"):
            return 1.2
        if s in {"reddit", "reddit-web"} or s.startswith("apify-reddit"):
            return 0.75
        if s == "hackernews":
            return 0.9
        if s == "duckduckgo":
            return 0.85
        return 1.0

    def recency_weight(meta: Dict[str, Any]) -> float:
        raw = str((meta or {}).get("publishedAt") or (meta or {}).get("pubDate") or (meta or {}).get("created_at") or "")
        m = re.search(r"(20\d{2})", raw)
        if not m:
            return 0.9
        y = int(m.group(1))
        if y >= 2025:
            return 1.05
        if y >= 2023:
            return 1.0
        if y >= 2020:
            return 0.9
        return 0.75

    weights = [max(0.3, source_weight(c.source) * recency_weight(c.meta)) for c in cards]
    w_total = sum(weights)

    pain_w = sum(w for c, w in zip(cards, weights) if c.pain_tags)
    wtp1_w = sum(w for c, w in zip(cards, weights) if c.wtp >= 1)
    wtp2_w = sum(w for c, w in zip(cards, weights) if c.wtp >= 2)
    workaround_w = sum(w for c, w in zip(cards, weights) if c.workaround_tags)
    sources = len(set(c.source for c in cards))

    pain_ratio = pain_w / max(1e-9, w_total)
    wtp_ratio = (wtp1_w + 2 * wtp2_w) / max(1e-9, (2 * w_total))
    workaround_ratio = workaround_w / max(1e-9, w_total)
    source_bonus = min(0.15, sources * 0.05)

    score = (
        45 * pain_ratio +
        35 * wtp_ratio +
        15 * workaround_ratio +
        100 * source_bonus
    )
    score = max(0, min(100, round(score, 1)))

    total = len(cards)
    if total >= 30 and sources >= 3:
        conf = "high"
    elif total >= 12 and sources >= 2:
        conf = "medium"
    else:
        conf = "low"

    if score >= 70:
        decision = "go"
    elif score >= 45:
        decision = "iterate"
    else:
        decision = "pivot"

    return {
        "demand_score": score,
        "confidence": conf,
        "decision": decision,
        "stats": {
            "evidence_count": total,
            "pain_hits": sum(1 for c in cards if c.pain_tags),
            "wtp_level1_plus": sum(1 for c in cards if c.wtp >= 1),
            "wtp_level2": sum(1 for c in cards if c.wtp >= 2),
            "workaround_hits": sum(1 for c in cards if c.workaround_tags),
            "source_count": sources,
            "weighted_total": round(w_total, 2),
            "weighted_pain": round(pain_w, 2),
            "weighted_wtp1_plus": round(wtp1_w, 2),
            "weighted_wtp2": round(wtp2_w, 2),
            "weighted_workaround": round(workaround_w, 2),
        },
    }


def recommend_next_experiment(decision: str) -> str:
    if decision == "go":
        return "결제 의향 검증용 스모크 테스트 랜딩(가격 3안) + 72시간 트래픽 테스트 실행"
    if decision == "iterate":
        return "타겟/문제정의 문구 2개 버전으로 커뮤니티 포스트 A/B 후 반응 비교"
    return "핵심 문제 가설 재정의 후 다른 세그먼트(타겟)로 쿼리 재수집"


# -----------------------------
# Reporting
# -----------------------------

def top_tags(cards: List[EvidenceCard], attr: str, k: int = 3) -> List[Tuple[str, int]]:
    freq: Dict[str, int] = {}
    for c in cards:
        vals = getattr(c, attr)
        for v in vals:
            freq[v] = freq.get(v, 0) + 1
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]


def llm_upgrade_summary(intake: Dict[str, Any], score: Dict[str, Any], cards: List[EvidenceCard]) -> Dict[str, str]:
    api_key = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # low-cost default

    if not api_key or OpenAI is None:
        return {}

    try:
        client = OpenAI(api_key=api_key)
        evidence_lines = []
        for c in cards[:12]:
            evidence_lines.append(f"- [{c.source}] {c.quote} ({c.source_url})")

        prompt = f"""
당신은 시장성 검증 리포트 에디터다.
아래 입력과 증거를 보고 한국어로 짧고 명확하게 작성:
1) executive_one_liner (1문장)
2) why_this_score (2~3문장)
3) actionable_next_step (1문장)

입력:
- target: {intake.get('target')}
- problem: {intake.get('problem')}
- alternatives: {intake.get('current_alternative')}
- outcome: {intake.get('promised_outcome')}

점수:
- demand_score: {score.get('demand_score')}
- confidence: {score.get('confidence')}
- decision: {score.get('decision')}

증거:
{chr(10).join(evidence_lines)}

JSON으로만 출력:
{{"executive_one_liner":"...","why_this_score":"...","actionable_next_step":"..."}}
"""
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "정확하고 간결한 시장성 분석 어시스턴트"},
                {"role": "user", "content": prompt},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)
        return {
            "executive_one_liner": data.get("executive_one_liner", ""),
            "why_this_score": data.get("why_this_score", ""),
            "actionable_next_step": data.get("actionable_next_step", ""),
        }
    except Exception as e:
        print(f"[warn] openai summary upgrade failed: {e}", file=sys.stderr)
        return {}


LLM_USAGE = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}


def _cap_payload(payload: Dict[str, Any], max_chars: int = 12000) -> Dict[str, Any]:
    """Best-effort payload cap for cost control."""
    try:
        s = json.dumps(payload, ensure_ascii=False)
        if len(s) <= max_chars:
            return payload
        # hard cap evidence length first
        p = dict(payload)
        if isinstance(p.get("evidence"), list):
            p["evidence"] = p["evidence"][:12]
        s2 = json.dumps(p, ensure_ascii=False)
        if len(s2) <= max_chars:
            return p
        # final cap by string truncation
        return {"truncated_payload": s2[:max_chars]}
    except Exception:
        return payload


def llm_json_call(task: str, schema_hint: str, payload: Dict[str, Any], temperature: float = 0.2) -> Dict[str, Any]:
    api_key = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_calls = int(os.getenv("LLM_MAX_CALLS", "4"))
    if not api_key or OpenAI is None:
        return {}
    if LLM_USAGE.get("calls", 0) >= max_calls:
        return {}
    try:
        client = OpenAI(api_key=api_key)
        common_rules = (
            "You are generating an evidence-based report. "
            "Never invent facts. Every claim should rely on provided evidence. "
            "If evidence is insufficient, output conservative conclusions. "
            "Return valid JSON only. Output language: Korean."
        )
        capped = _cap_payload(payload, max_chars=int(os.getenv("LLM_PAYLOAD_MAX_CHARS", "12000")))
        prompt = f"Task: {task}\nSchema: {schema_hint}\nPayload(JSON):\n{json.dumps(capped, ensure_ascii=False)}"
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "700")),
            messages=[
                {"role": "system", "content": common_rules},
                {"role": "user", "content": prompt},
            ],
        )

        usage = getattr(resp, "usage", None)
        if usage:
            LLM_USAGE["calls"] += 1
            LLM_USAGE["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            LLM_USAGE["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            LLM_USAGE["total_tokens"] += int(getattr(usage, "total_tokens", 0) or 0)
        content = (resp.choices[0].message.content or "").strip()
        try:
            return json.loads(content)
        except Exception:
            s = content.find("{")
            e = content.rfind("}")
            if s != -1 and e != -1 and e > s:
                return json.loads(content[s:e+1])
            raise
    except Exception as e:
        print(f"[warn] llm_json_call failed: {e}", file=sys.stderr)
        return {}


def module_replacement_map(intake: Dict[str, Any], cards: List[EvidenceCard]) -> Dict[str, Any]:
    seed = {}
    for c in cards:
        for w in c.workaround_tags:
            seed[w] = seed.get(w, 0) + 1
    seed_list = [{"alternative_name": k, "count": v} for k, v in sorted(seed.items(), key=lambda x: x[1], reverse=True)[:8]]
    payload = {
        "project_brief": intake,
        "seed_workarounds": seed_list,
        "evidence": [asdict(c) for c in cards[:50]],
    }
    schema = '{"confidence":"low|medium|high","replacement_map":[{"alternative_name":"","type":"tool|manual|outsourcing|existing_app|template|other","pros":[""],"cons":[""],"opportunity_gap":"","priority":1,"evidence_card_ids":[""]}],"more_evidence_needed":[""]}'
    out = llm_json_call("Build replacement map", schema, payload)
    return out or {"confidence": "low", "replacement_map": [], "more_evidence_needed": ["대안 근거 부족"]}


def module_offer_generator(intake: Dict[str, Any], replacement_map: Dict[str, Any], cards: List[EvidenceCard]) -> Dict[str, Any]:
    payload = {
        "project_brief": intake,
        "replacement_map": replacement_map,
        "evidence": [asdict(c) for c in cards[:40]],
    }
    schema = '{"confidence":"low|medium|high","offers":[{"tier":"low|mid|high","offer_name":"","core_promise":"","deliverables":[""],"price_anchor":{"currency":"KRW","range":""},"proof_citations":[""]}],"risk_reversal":[{"type":"trial|refund|pilot","copy":""}],"pricing_experiments":[{"type":"paid_beta|deposit|subscription","setup_steps":[""],"success_metric":""}]}'
    out = llm_json_call("Generate 3 offers, risk reversal, pricing experiments", schema, payload)
    return out or {"confidence": "low", "offers": [], "risk_reversal": [], "pricing_experiments": []}


def module_7day_plan(intake: Dict[str, Any], offers: Dict[str, Any], decision: str) -> Dict[str, Any]:
    payload = {"project_brief": intake, "offers": offers, "decision": decision}
    schema = '{"confidence":"low|medium|high","experiment_goal":"","kpi_cutlines":[{"metric":"","target":"","why":""}],"day_by_day_plan":[{"day":1,"action":"","deliverable":"","time_box_minutes":60}],"pivot_rules":[{"if":"","then":""}],"templates":{"threads_post":[""],"reddit_post":[""],"landing_copy":[""]}}'
    out = llm_json_call("Create 7-day fail-fast plan with KPI cutlines", schema, payload)
    return out or {"confidence": "low", "experiment_goal": "", "kpi_cutlines": [], "day_by_day_plan": [], "pivot_rules": [], "templates": {}}


def module_quality_explainer(intake: Dict[str, Any], score: Dict[str, Any], cards: List[EvidenceCard]) -> Dict[str, Any]:
    total = max(1, len(cards))
    distinct_sources = len(set(c.source for c in cards))
    unique_urls = len(set(c.source_url for c in cards))
    quality_score = round(min(1.0, 0.5 * (distinct_sources / 4) + 0.5 * (unique_urls / total)) * 100, 1)
    payload = {
        "project_brief": intake,
        "quality_metrics": {
            "quality_score": quality_score,
            "distinct_sources": distinct_sources,
            "total_evidence": len(cards),
            "decision": score.get("decision"),
        },
        "evidence": [asdict(c) for c in cards[:20]],
    }
    schema = '{"confidence":"low|medium|high","quality_score":0,"breakdown":[{"metric":"source_diversity","value":0,"why_it_matters":""}],"conclusion_guardrail":{"allow_strong_conclusion":false,"reason":""},"more_evidence_needed":[{"goal":"","how_to_collect":""}]}'
    out = llm_json_call("Explain evidence quality score and guardrail", schema, payload)
    if not out:
        out = {
            "confidence": "medium" if quality_score >= 50 else "low",
            "quality_score": quality_score,
            "breakdown": [{"metric": "source_diversity", "value": distinct_sources, "why_it_matters": "소스 다양성이 높을수록 결론 신뢰도 상승"}],
            "conclusion_guardrail": {"allow_strong_conclusion": quality_score >= 60, "reason": "품질점수 기반"},
            "more_evidence_needed": [{"goal": "WTP 신호 보강", "how_to_collect": "가격/유료 언급 키워드 중심 추가 수집"}],
        }
    return out


def module_template_pack(intake: Dict[str, Any], offers: Dict[str, Any], cards: List[EvidenceCard]) -> Dict[str, Any]:
    payload = {"project_brief": intake, "offers": offers, "evidence": [asdict(c) for c in cards[:30]]}
    schema = '{"confidence":"low|medium|high","threads_posts":[{"type":"save|recruit|repost","text":""}],"reddit_posts":[{"type":"question|feedback","title":"","body":""}],"landing_copy":[{"variant":"short|medium|strong","headline":"","subheadline":"","bullets":[""],"cta":""}],"faqs":[{"q":"","a":"","citations":[""]}]}'
    out = llm_json_call("Create copy-paste template pack", schema, payload) or {"confidence": "low", "threads_posts": [], "reddit_posts": [], "landing_copy": [], "faqs": []}

    # enforce minimum deliverables for paid UX
    if len(out.get("threads_posts", [])) < 3:
        out["threads_posts"] = out.get("threads_posts", []) + [
            {"type": "save", "text": "트레이더분들 매매일지 엑셀/노션으로 관리할 때 가장 불편한 점 1가지만 알려주세요."},
            {"type": "repost", "text": "매매복기 자동화가 되면 실제로 어떤 지표를 가장 먼저 보고 싶으신가요?"},
            {"type": "recruit", "text": "매매일지 웹서비스 유료베타 10명 모집합니다. 가장 필요한 기능 댓글로 주세요."},
        ]
        out["threads_posts"] = out["threads_posts"][:3]
    if len(out.get("reddit_posts", [])) < 2:
        out["reddit_posts"] = out.get("reddit_posts", []) + [
            {"type": "question", "title": "트레이딩 저널을 엑셀/노션 대신 웹으로 쓰면 어떤 점이 가장 중요할까요?", "body": "동기화/복기/대시보드/리스크관리 중 우선순위를 알고 싶습니다."},
            {"type": "feedback", "title": "매매일지 유료베타 오퍼 피드백 부탁드립니다", "body": "48시간 리포트 + 규칙위반 체크 오퍼가 실제로 결제할 가치가 있는지 의견 부탁드립니다."},
        ]
        out["reddit_posts"] = out["reddit_posts"][:2]
    if len(out.get("landing_copy", [])) < 3:
        out["landing_copy"] = out.get("landing_copy", []) + [
            {"variant": "short", "headline": "엑셀 대신 자동 동기화 매매일지", "subheadline": "복기와 대시보드를 한 화면에서", "bullets": ["자동 기록", "주간 리포트", "리스크 신호"], "cta": "유료베타 신청"},
            {"variant": "medium", "headline": "흩어진 매매기록을 하나의 웹서비스로", "subheadline": "노션/엑셀보다 빠른 복기", "bullets": ["검색 가능한 일지", "대시보드", "실수 패턴 추적"], "cta": "지금 검증하기"},
            {"variant": "strong", "headline": "트레이더를 위한 실행형 매매일지", "subheadline": "기록이 아니라 의사결정을 돕는 저널", "bullets": ["손실원인 분석", "규칙 위반 감지", "다음 액션 추천"], "cta": "48시간 리포트 받기"},
        ]
        out["landing_copy"] = out["landing_copy"][:3]
    if len(out.get("faqs", [])) < 10:
        defaults = [
            {"q": "엑셀과 뭐가 다른가요?", "a": "자동 동기화와 대시보드로 복기 시간을 줄입니다.", "citations": []},
            {"q": "누구에게 맞나요?", "a": "반복 매매복기와 성과 추적이 필요한 트레이더에게 적합합니다.", "citations": []},
            {"q": "유료 전환 기준은요?", "a": "시간 절감/리스크 감소 체감이 명확할 때 전환 가능성이 높습니다.", "citations": []},
            {"q": "데이터는 안전한가요?", "a": "민감정보 최소 수집과 접근제어를 적용합니다.", "citations": []},
            {"q": "모바일도 되나요?", "a": "핵심 기능은 모바일에서도 확인 가능하도록 설계합니다.", "citations": []},
            {"q": "초보자도 쓸 수 있나요?", "a": "템플릿 기반 입력으로 시작 장벽을 낮춥니다.", "citations": []},
            {"q": "기존 노션 템플릿과 병행 가능한가요?", "a": "초기에는 병행 사용 후 점진 전환을 권장합니다.", "citations": []},
            {"q": "리포트는 얼마나 자주 받나요?", "a": "주간 리포트 기본, 필요시 일간 요약 확장 가능합니다.", "citations": []},
            {"q": "환불/보장 정책은?", "a": "유료베타는 재분석 1회 제공 같은 리스크 완화 정책을 권장합니다.", "citations": []},
            {"q": "지금 바로 뭘 해야 하나요?", "a": "타겟 1개 세그먼트로 랜딩/게시글 실험을 7일 실행하세요.", "citations": []},
        ]
        out["faqs"] = (out.get("faqs", []) + defaults)[:10]
    return out


def module_pivot_map(intake: Dict[str, Any], score: Dict[str, Any], replacement_map: Dict[str, Any], cards: List[EvidenceCard]) -> Dict[str, Any]:
    payload = {
        "project_brief": intake,
        "scoring_summary": score,
        "replacement_map": replacement_map,
        "evidence": [asdict(c) for c in cards[:40]],
    }
    schema = '{"confidence":"low|medium|high","why_not_now":[{"reason":"","citations":[""]}],"pivot_candidates":[{"pivot_title":"","what_changes":"problem|target|offer","new_angle":"","why_this_might_work":"","supporting_evidence":[""],"next_test":""}],"keep_doing":[""],"stop_doing":[""]}'
    out = llm_json_call("Generate 3 adjacent pivot directions", schema, payload)
    if not out:
        out = {
            "confidence": "medium",
            "why_not_now": [{"reason": "WTP 강신호 부족", "citations": []}],
            "pivot_candidates": [
                {"pivot_title": "문제 피봇", "what_changes": "problem", "new_angle": "기록 도구 -> 규칙 위반 경고", "why_this_might_work": "리스크 pain 강함", "supporting_evidence": [], "next_test": "규칙위반 알림 오퍼 게시"},
                {"pivot_title": "타겟 피봇", "what_changes": "target", "new_angle": "전체 트레이더 -> 겸업 초보 트레이더", "why_this_might_work": "복기 루틴 니즈 집중", "supporting_evidence": [], "next_test": "초보 세그먼트 전용 랜딩"},
                {"pivot_title": "오퍼 피봇", "what_changes": "offer", "new_angle": "일지 SaaS -> 월말 리포트 자동생성", "why_this_might_work": "돈/시간 절감 가치 명확", "supporting_evidence": [], "next_test": "1회성 리포트 결제 실험"},
            ],
            "keep_doing": ["증거 기반 의사결정"],
            "stop_doing": ["근거 없는 강한 결론"],
        }
    return out


def estimate_llm_cost_usd() -> Dict[str, Any]:
    in_rate = float(os.getenv("OPENAI_RATE_INPUT_PER_1M", "0"))
    out_rate = float(os.getenv("OPENAI_RATE_OUTPUT_PER_1M", "0"))

    prompt_toks = LLM_USAGE.get("prompt_tokens", 0)
    completion_toks = LLM_USAGE.get("completion_tokens", 0)
    input_cost = (prompt_toks / 1_000_000.0) * in_rate
    output_cost = (completion_toks / 1_000_000.0) * out_rate
    total_cost = input_cost + output_cost

    return {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "calls": LLM_USAGE.get("calls", 0),
        "prompt_tokens": prompt_toks,
        "completion_tokens": completion_toks,
        "total_tokens": LLM_USAGE.get("total_tokens", 0),
        "rates": {
            "input_per_1m": in_rate,
            "output_per_1m": out_rate,
        },
        "cost_usd": {
            "input": round(input_cost, 6),
            "output": round(output_cost, 6),
            "total": round(total_cost, 6),
        },
    }


def compute_insight_metrics(cards: List[EvidenceCard], score: Dict[str, Any], score_breakdown: Dict[str, Any]) -> Dict[str, Any]:
    total = max(1, len(cards))
    unique_urls = len(set(c.source_url for c in cards))
    unique_sources = len(set(c.source for c in cards))

    wtp_direct_count = sum(1 for c in cards if c.wtp >= 1)
    wtp_strong_count = sum(1 for c in cards if c.wtp >= 2)
    wtp_ratio = round(wtp_direct_count / total, 4)

    duplicate_ratio = round(1.0 - (unique_urls / total), 4)

    raw_score = round(
        float(score_breakdown.get("pain", 0))
        + float(score_breakdown.get("wallet", 0))
        + float(score_breakdown.get("workaround", 0))
        + float(score_breakdown.get("momentum", 0)),
        1,
    )
    adjusted_score = float(score.get("demand_score", 0))

    return {
        "evidence_count": len(cards),
        "unique_sources": unique_sources,
        "unique_urls": unique_urls,
        "duplicate_ratio": duplicate_ratio,
        "wtp_direct_count": wtp_direct_count,
        "wtp_strong_count": wtp_strong_count,
        "wtp_direct_ratio": wtp_ratio,
        "raw_score": raw_score,
        "adjusted_score": adjusted_score,
        "score_gap": round(adjusted_score - raw_score, 1),
    }


def build_self_checklist(insight: Dict[str, Any]) -> Dict[str, Any]:
    checks = [
        {
            "name": "증거 수 30건 이상",
            "pass": insight.get("evidence_count", 0) >= 30,
            "value": insight.get("evidence_count", 0),
            "threshold": ">=30",
        },
        {
            "name": "소스 다양성 4개 이상",
            "pass": insight.get("unique_sources", 0) >= 4,
            "value": insight.get("unique_sources", 0),
            "threshold": ">=4",
        },
        {
            "name": "중복률 35% 이하",
            "pass": insight.get("duplicate_ratio", 1.0) <= 0.35,
            "value": insight.get("duplicate_ratio", 1.0),
            "threshold": "<=0.35",
        },
        {
            "name": "WTP 직접 문장 비율 5% 이상",
            "pass": insight.get("wtp_direct_ratio", 0.0) >= 0.05,
            "value": insight.get("wtp_direct_ratio", 0.0),
            "threshold": ">=0.05",
        },
        {
            "name": "점수 일관성(raw<=adjusted)",
            "pass": insight.get("raw_score", 0) <= insight.get("adjusted_score", 0),
            "value": f"raw={insight.get('raw_score', 0)}, adjusted={insight.get('adjusted_score', 0)}",
            "threshold": "raw<=adjusted",
        },
    ]

    fail_count = sum(1 for c in checks if not c["pass"])
    return {
        "checks": checks,
        "fail_count": fail_count,
        "status": "pass" if fail_count == 0 else "fail",
    }


def build_decision_evidence(cards: List[EvidenceCard], limit: int = 10) -> List[Dict[str, Any]]:
    def recency_bonus(meta: Dict[str, Any]) -> float:
        raw = str((meta or {}).get("publishedAt") or (meta or {}).get("pubDate") or (meta or {}).get("created_at") or "")
        m = re.search(r"(20\d{2})", raw)
        if not m:
            return 0.0
        y = int(m.group(1))
        if y >= 2025:
            return 1.0
        if y >= 2023:
            return 0.5
        return 0.0

    scored: List[Tuple[float, EvidenceCard]] = []
    for c in cards:
        text = f"{c.title} {c.quote}".lower()
        score = 0.0
        # decision contribution: WTP > workaround-gap > pain > momentum
        if c.wtp >= 2:
            score += 5.0
        elif c.wtp == 1:
            score += 3.0
        if c.workaround_tags:
            score += 2.0
        if c.pain_tags:
            score += 1.5
        if c.source in {"naver-news", "naver-blog", "youtube-comment", "youtube-video", "appstore-review", "googleplay-review"}:
            score += 1.2
        if any(k in text for k in ["가격", "유료", "결제", "구독", "price", "pay", "pricing"]):
            score += 1.5
        if any(k in text for k in ["엑셀", "노션", "hts", "excel", "notion", "manual"]):
            score += 0.8
        score += recency_bonus(c.meta)

        if len((c.quote or "").strip()) < 40:
            score -= 1.0
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict[str, Any]] = []
    seen_urls = set()
    wtp_count = 0
    workaround_count = 0

    for score, c in scored:
        if c.source_url in seen_urls:
            continue
        # force composition quality for top section
        if len(out) < limit:
            if c.wtp > 0 and wtp_count < 3:
                wtp_count += 1
            elif c.workaround_tags and workaround_count < 3:
                workaround_count += 1
            elif len(out) < 6:
                pass

            seen_urls.add(c.source_url)
            so_what = "문제 공감 신호"
            if c.wtp > 0:
                so_what = "결제 의향 신호"
            elif c.workaround_tags:
                so_what = "기존 대안 한계 신호"

            out.append({
                "score": round(score, 2),
                "quote": compact_text(c.quote, 260),
                "source": c.source,
                "url": c.source_url,
                "pain_tags": c.pain_tags,
                "workaround_tags": c.workaround_tags,
                "wtp": c.wtp,
                "signal": so_what,
            })
        if len(out) >= limit:
            break

    return out


def build_segment_posts(intake: Dict[str, Any]) -> Dict[str, Any]:
    segments = intake.get("segments") or [
        "전업/고빈도 트레이더",
        "직장인 겸업 트레이더",
        "초보 트레이더",
    ]
    out = []
    for seg in segments[:3]:
        out.append({
            "segment": seg,
            "posts": [
                {
                    "type": "공감형",
                    "text": f"{seg}분들, 매매일지 기록은 하는데 주간 복기에서 가장 시간을 잡아먹는 구간이 어디인가요? (입력/정리/패턴분석 중 1개)",
                },
                {
                    "type": "인사이트형",
                    "text": f"{seg} 기준으로 보면 문제 체감은 높은데 지불 의향은 낮습니다. '자동기록+주간리포트'에서 꼭 필요한 지표 3개만 알려주세요.",
                },
                {
                    "type": "모집형",
                    "text": f"{seg} 대상 유료베타 10명 모집: 자동기록+주간 성과리포트 테스트. 관심 있으면 DM에 BETA 남겨주세요.",
                },
            ],
        })
    return {"segments": out}


def tag_interview_answers(intake: Dict[str, Any]) -> Dict[str, Any]:
    answers = intake.get("interview_answers") or []
    tagged = []
    tag_counts = {"pain": 0, "wtp": 0, "workaround": 0}

    for i, a in enumerate(answers, start=1):
        text = a.get("text", "") if isinstance(a, dict) else str(a)
        pain = infer_pain_tags(text)
        workaround = infer_workarounds(text)
        wtp = infer_wtp(text)
        if pain:
            tag_counts["pain"] += 1
        if workaround:
            tag_counts["workaround"] += 1
        if wtp > 0:
            tag_counts["wtp"] += 1
        tagged.append({
            "id": i,
            "text": text,
            "pain_tags": pain,
            "workaround_tags": workaround,
            "wtp_level": wtp,
            "tags": [
                *( ["Pain"] if pain else [] ),
                *( ["Workaround"] if workaround else [] ),
                *( ["WTP"] if wtp > 0 else [] ),
            ],
        })

    total = len(answers)
    return {
        "count": total,
        "tag_counts": tag_counts,
        "wtp_ratio": round((tag_counts["wtp"] / total), 4) if total else 0.0,
        "items": tagged,
    }


def rejudge_sprint(intake: Dict[str, Any], base_decision: str) -> Dict[str, Any]:
    m = intake.get("sprint_metrics") or {}
    visits = int(m.get("landing_visits", 0) or 0)
    cta_clicks = int(m.get("cta_clicks", 0) or 0)
    saves = int(m.get("saves_likes", 0) or 0)
    dm_comments = int(m.get("dm_comments", 0) or 0)
    interviews = int(m.get("interviews", 0) or 0)
    price_mentions = int(m.get("price_mentions", 0) or 0)
    reservation_clicks = int(m.get("reservation_clicks", 0) or 0)
    applications = int(m.get("applications", 0) or 0)
    wtp_sentences = int(m.get("wtp_sentences", 0) or 0)
    repeated_pain_count = int(m.get("repeated_pain_count", 0) or 0)

    cta_rate = (cta_clicks / visits) if visits else 0.0
    apply_rate = (applications / visits) if visits else 0.0

    checks = [
        {"name": "CTA 클릭률>=3%", "pass": cta_rate >= 0.03, "value": round(cta_rate, 4), "weight": 1.2},
        {"name": "저장/좋아요>=20", "pass": saves >= 20, "value": saves, "weight": 0.8},
        {"name": "DM/댓글>=10", "pass": dm_comments >= 10, "value": dm_comments, "weight": 1.0},
        {"name": "인터뷰>=5", "pass": interviews >= 5, "value": interviews, "weight": 1.0},
        {"name": "가격 언급>=3", "pass": price_mentions >= 3, "value": price_mentions, "weight": 1.3},
        {"name": "신청률>=2%(신청/랜딩방문)", "pass": apply_rate >= 0.02, "value": round(apply_rate, 4), "weight": 1.6},
        {"name": "반복 Pain>=3", "pass": repeated_pain_count >= 3, "value": repeated_pain_count, "weight": 0.9},
        {"name": "WTP 문장>=3", "pass": wtp_sentences >= 3, "value": wtp_sentences, "weight": 1.5},
    ]
    pass_count = sum(1 for c in checks if c["pass"])
    weighted_total = sum(c["weight"] for c in checks)
    weighted_pass = sum(c["weight"] for c in checks if c["pass"])
    weighted_ratio = round(weighted_pass / weighted_total, 4) if weighted_total else 0.0

    hard_go = (apply_rate >= 0.02 and wtp_sentences >= 3 and interviews >= 5)
    if weighted_ratio >= 0.72 and hard_go:
        decision = "go"
    elif weighted_ratio >= 0.45:
        decision = "iterate"
    else:
        decision = "pivot"

    return {
        "base_decision": base_decision,
        "rejudged_decision": decision,
        "pass_count": pass_count,
        "total_checks": len(checks),
        "weighted_pass_ratio": weighted_ratio,
        "hard_go_conditions": {
            "apply_rate_ge_2pct": apply_rate >= 0.02,
            "wtp_sentences_ge_3": wtp_sentences >= 3,
            "interviews_ge_5": interviews >= 5,
        },
        "metrics": {
            "landing_visits": visits,
            "cta_clicks": cta_clicks,
            "cta_rate": round(cta_rate, 4),
            "saves_likes": saves,
            "dm_comments": dm_comments,
            "interviews": interviews,
            "price_mentions": price_mentions,
            "reservation_clicks": reservation_clicks,
            "applications": applications,
            "application_rate": round(apply_rate, 4),
            "wtp_sentences": wtp_sentences,
            "repeated_pain_count": repeated_pain_count,
        },
        "checks": checks,
    }


def build_execution_pack(intake: Dict[str, Any], score: Dict[str, Any], insight: Dict[str, Any], sprint_rejudge: Dict[str, Any], top_pains: List[Tuple[str, int]]) -> Dict[str, Any]:
    target = intake.get("target") or "핵심 타겟"
    top_pain = top_pains[0][0] if top_pains else "기록/복기"
    decision = sprint_rejudge.get("rejudged_decision") or score.get("decision", "iterate")
    apply_rate = (sprint_rejudge.get("metrics") or {}).get("application_rate", 0.0)
    wtp_ratio = insight.get("wtp_direct_ratio", 0.0)

    if decision == "go":
        tone = "검증 통과. 전환 실험을 확장합니다."
    elif decision == "iterate":
        tone = "핵심 가설 유지, 오퍼 문구/세그먼트 조정이 필요합니다."
    else:
        tone = "오퍼 피벗 권장. 결과물 중심 패키지로 재실험합니다."

    post = (
        f"[{target}] 지금 {top_pain} 문제를 해결할 '자동기록+주간리포트' 유료베타를 검증 중입니다. "
        f"신청률 {apply_rate:.2%}, WTP비율 {wtp_ratio:.2%}. 관심 있으면 DM에 BETA 남겨주세요."
    )
    dm_script = [
        "문의 감사합니다. 15분만 인터뷰하고 베타 우선 초대 드릴게요.",
        "현재 기록 도구(엑셀/노션/앱)와 가장 큰 불편 1가지를 알려주세요.",
        "자동기록+주간리포트가 해결된다면 월 얼마가 적정한지 알려주세요.",
    ]
    tweak_rule = "신청률<2%면 혜택문구 2줄 교체, WTP문장<3이면 가격 질문 문구를 전면에 배치"

    return {
        "status_tone": tone,
        "today_action": {
            "channel": "threads",
            "target": target,
            "timebox_minutes": 60,
            "post_copy": post,
            "kpi": {
                "apply_rate_target": 0.02,
                "dm_target": 10,
                "wtp_sentence_target": 3,
            },
            "failure_rule": tweak_rule,
        },
        "dm_script": dm_script,
    }


def build_report(intake: Dict[str, Any], cards: List[EvidenceCard], score: Dict[str, Any]) -> Dict[str, Any]:
    top_pains = top_tags(cards, "pain_tags", 3)
    top_workarounds = top_tags(cards, "workaround_tags", 3)
    top_wtp_quotes = [c.quote for c in cards if c.wtp > 0][:5]

    llm = llm_upgrade_summary(intake, score, cards)

    # Cost optimization: run fewer LLM modules when evidence is sparse or budget mode is on.
    budget_mode = os.getenv("COST_OPTIMIZED", "1") == "1"
    enough_evidence = len(cards) >= int(os.getenv("MIN_EVIDENCE_FOR_FULL_LLM", "25"))

    replacement_map = module_replacement_map(intake, cards)
    quality = module_quality_explainer(intake, score, cards)

    if budget_mode and not enough_evidence:
        offers = {"confidence": "low", "offers": [], "risk_reversal": [], "pricing_experiments": []}
        plan_7d = {"confidence": "low", "experiment_goal": "", "kpi_cutlines": [], "day_by_day_plan": [], "pivot_rules": [], "templates": {}}
        template_pack = {"confidence": "low", "threads_posts": [], "reddit_posts": [], "landing_copy": [], "faqs": []}
        pivot_map = {"confidence": "low", "why_not_now": [], "pivot_candidates": [], "keep_doing": [], "stop_doing": []}
    else:
        offers = module_offer_generator(intake, replacement_map, cards)
        plan_7d = module_7day_plan(intake, offers, score.get("decision", "iterate"))
        template_pack = module_template_pack(intake, offers, cards)
        pivot_map = module_pivot_map(intake, score, replacement_map, cards)

    # explainable score decomposition (B)
    stats = score.get("stats", {})
    total = max(1, stats.get("evidence_count", 1))
    pain_component = round(45 * (stats.get("pain_hits", 0) / total), 1)
    wallet_component = round(35 * ((stats.get("wtp_level1_plus", 0) + 2 * stats.get("wtp_level2", 0)) / (2 * total)), 1)
    workaround_component = round(15 * (stats.get("workaround_hits", 0) / total), 1)
    momentum_component = round(5 * min(1.0, stats.get("source_count", 0) / 4), 1)

    score_breakdown = {
        "pain": pain_component,
        "wallet": wallet_component,
        "workaround": workaround_component,
        "momentum": momentum_component,
        "total": round(pain_component + wallet_component + workaround_component + momentum_component, 1),
        "unlock_conditions": [
            "WTP 강신호(유료/가격) 증거를 추가 확보",
            "예약금/유료베타 전환 지표 확보",
            "핵심 세그먼트 1개에서 반복 반응 확인",
        ],
    }
    insight_metrics = compute_insight_metrics(cards, score, score_breakdown)
    self_checklist = build_self_checklist(insight_metrics)
    segment_posts = build_segment_posts(intake)
    interview_tagging = tag_interview_answers(intake)
    sprint_rejudge = rejudge_sprint(intake, score.get("decision", "iterate"))
    execution_pack = build_execution_pack(intake, score, insight_metrics, sprint_rejudge, top_pains)
    decision_evidence = build_decision_evidence(cards, limit=10)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "intake": intake,
        "summary": {
            "demand_score": score["demand_score"],
            "confidence": score["confidence"],
            "decision": score["decision"],
            "one_liner": llm.get("executive_one_liner") or f"수요 신호 {('강' if score['demand_score'] >= 70 else '중' if score['demand_score'] >=45 else '약')} / confidence {score['confidence']}",
            "why_this_score": llm.get("why_this_score", ""),
            "actionable_next_step": llm.get("actionable_next_step", ""),
        },
        "stats": score.get("stats", {}),
        "score_breakdown": score_breakdown,
        "insight_metrics": insight_metrics,
        "self_checklist": self_checklist,
        "segment_posts": segment_posts,
        "interview_tagging": interview_tagging,
        "sprint_rejudge": sprint_rejudge,
        "execution_pack": execution_pack,
        "decision_evidence": decision_evidence,
        "top_pains": [{"tag": t, "count": c} for t, c in top_pains],
        "top_workarounds": [{"tag": t, "count": c} for t, c in top_workarounds],
        "wtp_quotes": top_wtp_quotes,
        "next_experiment": llm.get("actionable_next_step") or recommend_next_experiment(score["decision"]),
        "replacement_map": replacement_map,
        "offers": offers,
        "plan_7d": plan_7d,
        "quality": quality,
        "template_pack": template_pack,
        "pivot_map": pivot_map,
        "llm_usage": estimate_llm_cost_usd(),
        "evidence": [asdict(c) for c in cards],
    }


def render_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    stats = report.get("stats", {})
    decision = s.get("decision", "iterate")

    if decision == "go":
        decision_text = "지금은 실행(Go) 구간입니다. 단, 작은 실험 단위로 빠르게 검증을 권장합니다."
    elif decision == "iterate":
        decision_text = "핵심 가설은 유지하되, 타겟/가치제안 문구를 조정하는 Iterate가 적절합니다."
    else:
        decision_text = "현재 데이터 기준으론 Pivot(문제정의/타겟 재설계) 쪽이 더 안전합니다."

    lines = []
    lines.append("# Signal Scan Premium Report")
    lines.append("")
    lines.append("## 1) Executive Summary")
    lines.append(f"- 생성 시각: {report['generated_at']}")
    lines.append(f"- 수요 점수: **{s['demand_score']} / 100**")
    lines.append(f"- 신뢰도: **{s['confidence']}**")
    lines.append(f"- 권고: **{s['decision'].upper()}**")
    lines.append(f"- 한 줄 결론: {s['one_liner']}")
    lines.append(f"- 해석: {decision_text}")
    if s.get("why_this_score"):
        lines.append(f"- 모델 해설: {s.get('why_this_score')}")

    lines.append("\n## 2) 왜 이런 결과가 나왔나 (Score Breakdown)")
    lines.append(f"- 증거 수: **{stats.get('evidence_count', 0)}**")
    lines.append(f"- Pain 신호: **{stats.get('pain_hits', 0)}**")
    lines.append(f"- WTP(가격/유료) 신호: **{stats.get('wtp_level1_plus', 0)}**")
    lines.append(f"- 강한 WTP 신호: **{stats.get('wtp_level2', 0)}**")
    lines.append(f"- 기존 대안 언급: **{stats.get('workaround_hits', 0)}**")
    lines.append(f"- 소스 다양성: **{stats.get('source_count', 0)}**")
    sb = report.get("score_breakdown", {})
    if sb:
        lines.append(f"- 분해 점수: Pain {sb.get('pain',0)} / Wallet {sb.get('wallet',0)} / Workaround {sb.get('workaround',0)} / Momentum {sb.get('momentum',0)}")
        lines.append("- Go로 바꾸기 위한 조건:")
        for u in sb.get("unlock_conditions", [])[:3]:
            lines.append(f"  - {u}")

    im = report.get("insight_metrics", {})
    if im:
        lines.append("- 정량 검증 지표:")
        lines.append(f"  - Raw/Adjusted: {im.get('raw_score')} / {im.get('adjusted_score')} (gap {im.get('score_gap')})")
        lines.append(f"  - WTP 문장 비율: {im.get('wtp_direct_count')}/{im.get('evidence_count')} = {im.get('wtp_direct_ratio')}")
        lines.append(f"  - 중복률: {im.get('duplicate_ratio')} / 소스 다양성: {im.get('unique_sources')}")

    sc = report.get("self_checklist", {})
    if sc:
        lines.append(f"- 셀프 체크리스트 상태: **{sc.get('status','fail')}** (실패 {sc.get('fail_count',0)}개)")

    lines.append("\n## 3) 핵심 인사이트")
    top_pains = report.get("top_pains", [])
    top_works = report.get("top_workarounds", [])
    if top_pains:
        lines.append("- 주요 Pain TOP:")
        for x in top_pains[:3]:
            lines.append(f"  - {x['tag']} ({x['count']})")
    else:
        lines.append("- 뚜렷한 Pain 신호가 부족합니다. (추가 데이터 수집 필요)")

    if top_works:
        lines.append("- 현재 사용 대안 TOP:")
        for x in top_works[:5]:
            lines.append(f"  - {x['tag']} ({x['count']})")

    wtp_quotes = report.get("wtp_quotes", [])
    if wtp_quotes:
        lines.append("- WTP(가격/유료) 관련 근거 문장:")
        for q in wtp_quotes[:3]:
            lines.append(f"  - \"{q}\"")
    else:
        lines.append("- WTP 신호가 약합니다. 가격 검증 질문을 포함한 추가 탐색이 필요합니다.")

    lines.append("\n## 4) 바로 실행할 다음 실험 (48시간)")
    lines.append(f"- 권장 실험: **{report.get('next_experiment', '')}**")
    lines.append("- 실험 체크리스트:")
    lines.append("  1. 타겟 1개 세그먼트로 고정")
    lines.append("  2. 문제문장 1개 + 가치제안 1개로 검증 글 작성")
    lines.append("  3. 반응(댓글/저장/문의) 수치 기록")

    sp = report.get("segment_posts", {})
    lines.append("\n## 5) 세그먼트별 모집글 3종")
    for seg in (sp.get("segments") or [])[:3]:
        lines.append(f"- [{seg.get('segment','세그먼트')}]")
        for post in seg.get("posts", [])[:3]:
            lines.append(f"  - ({post.get('type','')}) {post.get('text','')}")

    it = report.get("interview_tagging", {})
    lines.append("\n## 6) 인터뷰 자동 태깅")
    lines.append(f"- 인터뷰 답변 수: {it.get('count', 0)}")
    tc = it.get("tag_counts", {})
    lines.append(f"- 태깅 집계: Pain {tc.get('pain',0)} / WTP {tc.get('wtp',0)} / Workaround {tc.get('workaround',0)}")
    lines.append(f"- WTP 비율: {it.get('wtp_ratio',0.0)}")

    sr = report.get("sprint_rejudge", {})
    lines.append("\n## 7) 스프린트 자동 재판정")
    lines.append(f"- 기존 판정: {sr.get('base_decision','iterate')} -> 재판정: **{sr.get('rejudged_decision','iterate')}**")
    lines.append(f"- 체크 통과: {sr.get('pass_count',0)}/{sr.get('total_checks',0)}")
    if sr:
        lines.append(f"- 가중 통과율: {sr.get('weighted_pass_ratio',0)}")

    ep = report.get("execution_pack", {})
    if ep:
        lines.append("\n## 7-1) 실행 산출물 자동 제안")
        lines.append(f"- 상태 요약: {ep.get('status_tone','')}")
        ta = ep.get("today_action", {})
        if ta:
            lines.append(f"- 오늘 액션: {ta.get('channel','threads')} / 타겟: {ta.get('target','')}")
            lines.append(f"- 복붙 포스트: {ta.get('post_copy','')}")
            k = ta.get("kpi", {})
            lines.append(f"- KPI: 신청률>={k.get('apply_rate_target',0)} / DM>={k.get('dm_target',0)} / WTP문장>={k.get('wtp_sentence_target',0)}")
            lines.append(f"- 실패 규칙: {ta.get('failure_rule','')}")

    # Replacement map
    rep = report.get("replacement_map", {})
    lines.append("\n## 8) Replacement Map (현재 대안/불만)")
    for i, alt in enumerate((rep.get("replacement_map") or [])[:5], start=1):
        lines.append(f"- {i}. {alt.get('alternative_name','-')} ({alt.get('type','other')})")
        if alt.get("opportunity_gap"):
            lines.append(f"  - Opportunity gap: {alt.get('opportunity_gap')}")

    # Offers
    offers = (report.get("offers", {}) or {}).get("offers", [])
    lines.append("\n## 9) Offer 3안")
    for o in offers[:3]:
        pa = (o.get("price_anchor") or {})
        lines.append(f"- [{o.get('tier','mid')}] {o.get('offer_name','')} — {o.get('core_promise','')}")
        if pa:
            lines.append(f"  - 가격 가이드: {pa.get('range','')} {pa.get('currency','')}")

    # Pivot map (E)
    pm = report.get("pivot_map", {})
    lines.append("\n## 10) Pivot Map (3방향)")
    for p in (pm.get("pivot_candidates") or [])[:3]:
        lines.append(f"- {p.get('pivot_title','Pivot')}: {p.get('new_angle','')}")
        lines.append(f"  - 이유: {p.get('why_this_might_work','')}")
        lines.append(f"  - 다음 테스트: {p.get('next_test','')}")

    # 7-day plan
    plan = report.get("plan_7d", {})
    lines.append("\n## 11) 7일 실험 플랜")
    for d in (plan.get("day_by_day_plan") or [])[:7]:
        lines.append(f"- Day {d.get('day')}: {d.get('action','')} (산출물: {d.get('deliverable','')})")

    # Template pack
    tp = report.get("template_pack", {})
    lines.append("\n## 12) 복붙 템플릿")
    for t in (tp.get("threads_posts") or [])[:2]:
        lines.append(f"- Threads({t.get('type','post')}): {t.get('text','')}")
    for lp in (tp.get("landing_copy") or [])[:2]:
        lines.append(f"- Landing[{lp.get('variant','short')}]: {lp.get('headline','')} / {lp.get('subheadline','')}")

    # Quality
    q = report.get("quality", {})
    lines.append("\n## 13) Evidence Quality")
    if q:
        lines.append(f"- Quality Score: **{q.get('quality_score','-')}**")
        cg = q.get("conclusion_guardrail", {})
        lines.append(f"- 강한 결론 허용: {cg.get('allow_strong_conclusion', False)} ({cg.get('reason','')})")

    lines.append("\n## 14) Evidence Highlights Top 10")
    for e in report.get("evidence", [])[:10]:
        lines.append(f"- [{e['source']}] {e['title']}")
        lines.append(f"  - 링크: {e['source_url']}")
        lines.append(f"  - 인용: {e['quote']}")

    lu = report.get("llm_usage", {})
    if lu:
        lines.append("\n## 15) LLM 사용량/원가(추정)")
        lines.append(f"- 모델: {lu.get('model')} / 호출수: {lu.get('calls', 0)}")
        lines.append(f"- 토큰: prompt {lu.get('prompt_tokens',0)}, completion {lu.get('completion_tokens',0)}, total {lu.get('total_tokens',0)}")
        c = lu.get("cost_usd", {})
        lines.append(f"- 비용(USD): input ${c.get('input',0)} + output ${c.get('output',0)} = **${c.get('total',0)}**")

    lines.append("\n## 16) 신뢰도 안내")
    lines.append("- 이 리포트는 커뮤니티 신호 기반 분석입니다. 예측이 아니라 현재 관측된 근거의 요약입니다.")
    lines.append("- 신뢰도를 올리려면: 소스 다양성 확대 + WTP 문장 추가 수집이 필요합니다.")

    return "\n".join(lines) + "\n"


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Signal Scan CLI - demand signal reporter")
    parser.add_argument("--input", required=True, help="Path to intake JSON")
    parser.add_argument("--outdir", default="./out", help="Output directory")
    parser.add_argument("--no-reddit", action="store_true", help="Skip reddit fetch")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        intake = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)

    plan = build_query_plan(intake)

    cards: List[EvidenceCard] = []

    # Product Hunt
    ph_token = os.getenv("PRODUCTHUNT_TOKEN")
    if ph_token:
        for term in plan["producthunt_queries"][:4]:
            try:
                cards.extend(fetch_producthunt(ph_token, term, first=4))
            except Exception as e:
                print(f"[warn] producthunt failed for '{term}': {e}", file=sys.stderr)
    else:
        print("[warn] PRODUCTHUNT_TOKEN not set; skipping Product Hunt", file=sys.stderr)

    # DuckDuckGo (web search first, instant-answer fallback)
    for q in plan["ddg_queries"][:8]:
        try:
            cards.extend(fetch_ddg_search(q, max_results=8))
        except Exception as e:
            print(f"[warn] ddg-search failed for '{q}': {e}", file=sys.stderr)
            try:
                cards.extend(fetch_ddg(q, max_results=5))
            except Exception as e2:
                print(f"[warn] ddg fallback failed for '{q}': {e2}", file=sys.stderr)

    # Naver OpenAPI (KR signal boost)
    naver_id = os.getenv("NAVER_CLIENT_ID")
    naver_secret = os.getenv("NAVER_CLIENT_SECRET")
    if naver_id and naver_secret:
        for q in plan["naver_queries"][:8]:
            try:
                cards.extend(fetch_naver_search(naver_id, naver_secret, q, display=8))
            except Exception as e:
                print(f"[warn] naver failed for '{q}': {e}", file=sys.stderr)
    else:
        print("[warn] NAVER_CLIENT_ID / NAVER_CLIENT_SECRET not set; skipping Naver", file=sys.stderr)

    # YouTube Data API
    yt_key = os.getenv("YOUTUBE_API_KEY")
    if yt_key:
        for q in plan["ddg_queries"][:4]:
            try:
                cards.extend(fetch_youtube_search_and_comments(yt_key, q, max_videos=3, max_comments=8))
            except Exception as e:
                print(f"[warn] youtube failed for '{q}': {e}", file=sys.stderr)
    else:
        print("[warn] YOUTUBE_API_KEY not set; skipping YouTube", file=sys.stderr)

    # Google Play signals (optional)
    if os.getenv("ENABLE_GOOGLE_PLAY", "1") == "1":
        gp_lang = os.getenv("GOOGLE_PLAY_LANG", "ko")
        gp_country = os.getenv("GOOGLE_PLAY_COUNTRY", "kr")
        for q in plan["ddg_queries"][:3]:
            try:
                cards.extend(fetch_google_play_search_and_reviews(
                    q,
                    lang=gp_lang,
                    country=gp_country,
                    max_apps=int(os.getenv("GOOGLE_PLAY_MAX_APPS", "4")),
                    max_reviews=int(os.getenv("GOOGLE_PLAY_MAX_REVIEWS", "12")),
                ))
            except Exception as e:
                print(f"[warn] googleplay failed for '{q}': {e}", file=sys.stderr)

    # Apple App Store signals (optional; requires app_store_targets in intake)
    if os.getenv("ENABLE_APP_STORE", "1") == "1":
        app_store_targets = intake.get("app_store_targets") or []
        if app_store_targets:
            try:
                cards.extend(fetch_appstore_reviews(
                    app_store_targets,
                    review_count=int(os.getenv("APP_STORE_MAX_REVIEWS", "12")),
                ))
            except Exception as e:
                print(f"[warn] appstore failed: {e}", file=sys.stderr)

    # Hacker News (Algolia search)
    for q in plan["ddg_queries"][:4]:
        try:
            cards.extend(fetch_hn_algolia_search(q, max_hits=12))
        except Exception as e:
            print(f"[warn] hackernews failed for '{q}': {e}", file=sys.stderr)

    # Reddit
    if not args.no_reddit:
        reddit_token = os.getenv("REDDIT_ACCESS_TOKEN")
        if reddit_token:
            for q in plan["reddit_queries"][:6]:
                try:
                    cards.extend(fetch_reddit_oauth(reddit_token, q, limit=6))
                except Exception as e:
                    print(f"[warn] reddit failed for '{q}': {e}", file=sys.stderr)
        else:
            print("[warn] REDDIT_ACCESS_TOKEN not set; skipping Reddit", file=sys.stderr)

    # Apify collectors (task or actor, optional high-quality boost)
    apify_token = os.getenv("APIFY_TOKEN")
    if apify_token:
        apify_task_map = [
            ("reddit", os.getenv("APIFY_TASK_REDDIT")),
            ("naver", os.getenv("APIFY_TASK_NAVER")),
            ("youtube", os.getenv("APIFY_TASK_YOUTUBE")),
        ]
        use_actor_reddit = os.getenv("APIFY_USE_ACTOR_REDDIT", "0") == "1"
        apify_actor_map = [
            ("reddit", os.getenv("APIFY_ACTOR_REDDIT") if use_actor_reddit else None),
            ("naver", os.getenv("APIFY_ACTOR_NAVER")),
            ("youtube", os.getenv("APIFY_ACTOR_YOUTUBE")),
        ]

        def _apify_input(q: str) -> Dict[str, Any]:
            return {
                "query": q,
                "searchTerms": [q],
                "searches": [q],
                "startUrls": [],
                "ignoreStartUrls": True,
                "maxItems": int(os.getenv("APIFY_MAX_ITEMS", "20")),
                "sort": "new",
            }

        apify_queries = []
        apify_queries.extend((plan.get("reddit_queries") or [])[:3])
        apify_queries.extend((plan.get("ddg_queries") or [])[:2])
        apify_queries.extend([intake.get("target", ""), intake.get("problem", "")])
        apify_queries = [q.strip() for q in apify_queries if q and q.strip()]
        apify_queries = list(dict.fromkeys(apify_queries))[:6]

        reddit_focused_queries = build_reddit_english_queries(intake, plan)
        extra_q = [x.strip() for x in (os.getenv("APIFY_REDDIT_EXTRA_QUERIES", "").split(",")) if x.strip()]
        reddit_focused_queries.extend(extra_q)
        reddit_focused_queries = [q for q in reddit_focused_queries if q]
        reddit_focused_queries = list(dict.fromkeys(reddit_focused_queries))[:10]

        for source_name, task_id in apify_task_map:
            if not task_id:
                continue
            source_queries = reddit_focused_queries if source_name == "reddit" else apify_queries
            source_max_items = int(os.getenv("APIFY_MAX_ITEMS_REDDIT", os.getenv("APIFY_MAX_ITEMS", "20"))) if source_name == "reddit" else int(os.getenv("APIFY_MAX_ITEMS", "20"))
            for q in source_queries:
                try:
                    items = fetch_apify_task(
                        apify_token,
                        task_id,
                        run_input=_apify_input(q),
                        limit=source_max_items,
                    )
                    cards.extend(normalize_apify_items(items, source_name, q))
                except Exception as e:
                    print(f"[warn] apify-task-{source_name} failed for '{q}': {e}", file=sys.stderr)

        for source_name, actor_id in apify_actor_map:
            if not actor_id:
                continue
            source_queries = reddit_focused_queries if source_name == "reddit" else apify_queries
            source_max_items = int(os.getenv("APIFY_MAX_ITEMS_REDDIT", os.getenv("APIFY_MAX_ITEMS", "20"))) if source_name == "reddit" else int(os.getenv("APIFY_MAX_ITEMS", "20"))
            for q in source_queries:
                try:
                    actor_input = _apify_input(q)
                    actor_input["maxItems"] = source_max_items
                    if source_name == "reddit":
                        actor_input.update({
                            "searchPosts": True,
                            "searchComments": False,
                            "searchCommunities": False,
                            "searchUsers": False,
                            "maxPostCount": source_max_items,
                            "maxComments": 0,
                            "sort": "new",
                            "includeNSFW": False,
                            "skipCommunity": True,
                            "skipUserPosts": True,
                        })
                    items = fetch_apify_actor(
                        apify_token,
                        actor_id,
                        run_input=actor_input,
                        limit=source_max_items,
                    )
                    cards.extend(normalize_apify_items(items, source_name, q))
                except Exception as e:
                    print(f"[warn] apify-actor-{source_name} failed for '{q}': {e}", file=sys.stderr)

    # Manual links from intake (Threads/Reddit etc.)
    for u in (intake.get("thread_links") or []):
        cards.append(EvidenceCard(
            source="threads-manual",
            source_url=u,
            title="Manual Threads link",
            quote="사용자 제공 Threads 링크",
            pain_tags=[],
            wtp=0,
            workaround_tags=[],
            meta={"manual": True},
        ))
    for u in (intake.get("reddit_links") or []):
        cards.append(EvidenceCard(
            source="reddit-manual",
            source_url=u,
            title="Manual Reddit link",
            quote="사용자 제공 Reddit 링크",
            pain_tags=[],
            wtp=0,
            workaround_tags=[],
            meta={"manual": True},
        ))

    # Keyword gate: keep only relevant evidence
    dynamic_terms = build_dynamic_terms(intake)
    filtered: List[EvidenceCard] = []
    dropped = 0
    for c in cards:
        # keep user-manual links always
        if c.meta.get("manual"):
            filtered.append(c)
            continue

        blob = f"{c.title} {c.quote}".strip()

        # stricter for generic DDG sources to reduce noise
        th = 2
        if c.source in {"duckduckgo", "reddit-web", "threads"}:
            th = 3

        # reddit source quality filter
        if c.source.startswith("apify-reddit") or c.source in {"reddit", "reddit-web"}:
            u = (c.source_url or "").lower()
            if "/r/" in u and not any(f"/r/{sub}" in u for sub in REDDIT_ALLOWED_SUBS):
                dropped += 1
                continue

        # relax threshold slightly when dynamic terms exist (generic ideas)
        if dynamic_terms and c.source.startswith("apify-"):
            th = max(1, th - 1)

        if is_relevant_text(blob, threshold=th, extra_terms=dynamic_terms):
            filtered.append(c)
        else:
            dropped += 1

    cards = filtered
    if dropped:
        print(f"[info] keyword-gate dropped {dropped} low-relevance evidence(s)", file=sys.stderr)

    # Dedup by quote+url
    uniq = {}
    for c in cards:
        key = (c.source_url, c.quote)
        if key not in uniq:
            uniq[key] = c
    cards = list(uniq.values())

    score = score_report(cards)
    report = build_report(intake, cards, score)

    with open(os.path.join(args.outdir, "query_plan.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_markdown(report))

    with open(os.path.join(args.outdir, "evidence.jsonl"), "w", encoding="utf-8") as f:
        for c in cards:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    print(json.dumps({
        "ok": True,
        "demand_score": score["demand_score"],
        "confidence": score["confidence"],
        "decision": score["decision"],
        "evidence_count": score.get("stats", {}).get("evidence_count", 0),
        "outdir": os.path.abspath(args.outdir),
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
