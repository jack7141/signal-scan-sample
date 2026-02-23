#!/usr/bin/env python3
import argparse
import hashlib
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

SERVICE_CONTEXT_TERMS = [
    "서비스", "도구", "툴", "앱", "프로그램", "솔루션", "플랫폼", "SaaS",
    "구독", "결제", "유료", "플랜", "요금", "가격표", "베타", "리포트", "컨설팅",
    "웹서비스", "소프트웨어", "software", "app", "tool", "platform", "service",
    "subscription", "beta", "saas", "report", "consulting",
]

NON_PRODUCT_PRICE_TERMS = [
    "아파트", "전세", "분양", "부동산", "금 가격", "리튜", "원자재", "환율",
    "주가", "코스피", "코스닥", "종목", "2차전지", "금리", "대출",
    "상가", "토지", "재개발", "공매도", "테마주", "리딩방",
    "gold price", "oil price", "stock price", "real estate",
]

WTP_STRONG_PATTERNS = [
    r"이미\s*(구독|결제|유료)\s*중",
    r"(월|년)\s*\d+(\.\d+)?\s*(만원|원)\s*내고",
    r"돈\s*내고\s*써",
    r"유료로라도\s*쓰",
    r"pay for", r"paid for", r"already paying",
    r"지불할 의향", r"낼 의향", r"구매 의향",
    r"buy this", r"구독 중",
]

WTP_WEAK_PATTERNS = [
    r"(가격|요금|비용)\s*(얼마|어떻게|좀|대략)",
    r"(구독|결제)\s*(할|하고|해볼)",
    r"worth it", r"pricing",
    r"얼마.{0,5}낼", r"how much",
]


def build_relevance_terms(intake: Dict[str, Any]) -> List[str]:
    """intake 기반으로 관련성 판단 용어를 동적 생성한다."""
    parts: List[str] = []
    for field in ["target", "problem", "current_alternative", "promised_outcome"]:
        v = str(intake.get(field) or "")
        if v:
            parts.append(v)
    for k in (intake.get("keywords") or []):
        parts.append(str(k))
    blob = " ".join(parts)
    # 2글자 이상 토큰 추출 + 원문 구절도 포함
    tokens = re.split(r"[\s,/·]+", blob)
    out: List[str] = []
    seen: set = set()
    # 원문 구절(필드 값 자체)을 relevance term 으로 우선 등록
    for field in ["target", "problem"]:
        v = str(intake.get(field) or "").strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    for t in tokens:
        t = t.strip()
        if len(t) < 2:
            continue
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return out[:40]


def build_problem_terms(intake: Dict[str, Any]) -> List[str]:
    """intake 기반으로 문제/대안 관련 용어를 동적 생성한다."""
    parts: List[str] = []
    for field in ["current_alternative", "problem"]:
        v = str(intake.get(field) or "")
        if v:
            parts.append(v)
    blob = " ".join(parts)
    tokens = re.split(r"[\s,/·]+", blob)
    out: List[str] = []
    seen: set = set()
    for t in tokens:
        t = t.strip()
        if len(t) < 2:
            continue
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t)
    return out[:20]


BAD_URL_KEYWORDS = [
    "lotto", "arclink", "games-cn", "meme", "horror", "/r/ssss", "clickbait",
]

REDDIT_ALLOWED_SUBS = [
    "startups", "Entrepreneur", "SaaS", "smallbusiness", "sideproject", "nocode", "indiehackers",
]

PREFERRED_DOMAIN_HINTS = [
    "naver.com", "tistory.com", "brunch.co.kr", "velog.io", "youtube.com",
    "reddit.com", "threads.net", "medium.com", "blog.naver.com", "cafe.naver.com",
    "quora.com", "producthunt.com", "indiehackers.com",
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


STOPWORDS_KR = {"내", "나의", "아이디어", "시장에서", "있을지", "어렵다", "방법", "하는", "것", "수",
                "이", "가", "을", "를", "에", "에서", "로", "으로", "와", "과", "도", "만",
                "했을", "할", "한", "된", "되는", "하고", "하면", "없다", "있다", "같은",
                "때", "더", "의", "및", "또는", "그", "등", "위한", "대한", "통한"}


def extract_topic_terms(intake: Dict[str, Any]) -> List[str]:
    """intake에서 핵심 토픽 명사를 추출한다. WTP 문맥 판단과 적합도 필터에 사용."""
    # 1) keywords 우선
    terms = list(intake.get("keywords") or [])
    # 2) problem에서 명사 후보 (단순 휴리스틱)
    problem = str(intake.get("problem") or "")
    tokens = re.split(r'[\s,./!?()\[\]{}:;"\'""'']+', problem)
    tokens = [t.strip() for t in tokens if len(t.strip()) >= 2]
    tokens = [t for t in tokens if t not in STOPWORDS_KR]
    # 3) target에서도 추출
    target = str(intake.get("target") or "")
    t_tokens = re.split(r'[\s,./!?()\[\]{}:;"\'""'']+', target)
    t_tokens = [t.strip() for t in t_tokens if len(t.strip()) >= 2]
    t_tokens = [t for t in t_tokens if t not in STOPWORDS_KR]
    # 4) 합치고 dedup
    out: List[str] = []
    seen: set = set()
    for t in terms + tokens + t_tokens:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out[:12]


def infer_wtp(text: str, topic_terms: Optional[List[str]] = None) -> int:
    """WTP 탐지: 서비스/토픽 문맥이 있을 때만 인정. 부동산/주식/원자재 가격은 제외."""
    t = (text or "").lower()

    # 0) 가격/결제 단어 자체가 없으면 WTP 아님
    if not re.search(r"(가격|요금|비용|유료|결제|구독|지불|의향|내고|pay|price|pricing|subscribe|worth|paying)", t):
        return 0

    # 1) 부동산/원자재/주식 '가격' 문맥은 WTP에서 제외
    if any(k.lower() in t for k in NON_PRODUCT_PRICE_TERMS):
        return 0

    # 2) 서비스 문맥 or 토픽 문맥이 같이 있어야 함 (둘 다 없으면 오탐)
    has_service_context = any(k.lower() in t for k in SERVICE_CONTEXT_TERMS)
    has_topic_context = any(k.lower() in t for k in (topic_terms or []))
    if not (has_service_context or has_topic_context):
        return 0

    # 3) 강/약 신호 분리
    if any(re.search(p, t) for p in WTP_STRONG_PATTERNS):
        return 2
    if any(re.search(p, t) for p in WTP_WEAK_PATTERNS):
        return 1

    # 서비스 문맥 + 가격 단어가 같이 있으면 최소 WTP=1 (약한 신호)
    if has_service_context or has_topic_context:
        return 1

    return 0


def compact_text(s: str, max_len: int = 240) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len] + ("…" if len(s) > max_len else "")


NOISE_TERMS = [
    "토지", "부동산", "아파트", "재개발", "정책", "코스피", "코스닥", "공매", "인버스", "섹터", "2차전지",
    "주식 추천", "테마주", "리딩방", "상가", "금리", "대출", "dc-link", "현대차", "lg전자",
]


def relevance_score(text: str, extra_terms: Optional[List[str]] = None,
                    relevance_terms: Optional[List[str]] = None,
                    problem_terms: Optional[List[str]] = None) -> int:
    low = (text or "").lower()
    s = 0
    for t in (relevance_terms or []):
        if t.lower() in low:
            s += 2
    for t in (problem_terms or []):
        if t.lower() in low:
            s += 1
    for t in (extra_terms or []):
        tt = (t or "").strip().lower()
        if len(tt) >= 2 and tt in low:
            s += 2

    neg_hits = sum(1 for n in NOISE_TERMS if n.lower() in low)
    if neg_hits:
        s -= min(6, neg_hits * 2)
    return s


def build_project_id(intake: Dict[str, Any]) -> str:
    key = json.dumps({
        "target": intake.get("target", ""),
        "problem": intake.get("problem", ""),
        "keywords": intake.get("keywords", []),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


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
    "대시보드": "dashboard",
    "엑셀": "excel",
    "노션": "notion",
    "자동화": "automation",
    "시장조사": "market research",
    "검증": "validation",
    "창업": "startup",
    "고객": "customer",
    "유료": "paid",
    "수익화": "monetization",
    "가격": "pricing",
    "경쟁": "competition",
}


def is_idea_validation_context(intake: Dict[str, Any]) -> bool:
    blob = " ".join([
        str(intake.get("target") or ""),
        str(intake.get("problem") or ""),
        str(intake.get("current_alternative") or ""),
        " ".join(intake.get("keywords") or []),
    ]).lower()
    markers = ["idea", "validation", "wtp", "시장성", "시장조사", "수익화", "창업", "startup"]
    return any(m in blob for m in markers)


def build_reddit_english_queries(intake: Dict[str, Any], plan: Dict[str, Any]) -> List[str]:
    seed = []
    seed.extend((plan.get("reddit_queries") or [])[:4])
    seed.extend((intake.get("keywords") or [])[:8])
    seed.extend([str(intake.get("target") or ""), str(intake.get("problem") or "")])

    idea_mode = is_idea_validation_context(intake)
    if idea_mode:
        english = [
            "startup idea validation",
            "how to validate startup idea",
            "willingness to pay test",
            "customer discovery interview",
            "pre selling before build",
            "market demand validation",
            "go to market validation",
            "founder idea validation mistakes",
        ]
        required_tokens = ["validation", "startup", "idea", "customer", "market", "wtp", "pricing", "mvp", "pre selling", "discovery"]
    else:
        # intake 기반 동적 영문 쿼리 생성 (트레이딩 하드코딩 제거)
        english = []
        en_keywords = [k for k in (intake.get("keywords") or []) if re.search(r"[a-z]", k.lower())]
        target_en = str(intake.get("target") or "")
        problem_en = str(intake.get("problem") or "")
        for ek in en_keywords[:5]:
            english.append(ek)
            english.append(f"{ek} app")
            english.append(f"{ek} alternatives")
        if target_en:
            for kr, en in KR_TO_EN_HINTS.items():
                if kr in target_en:
                    english.append(en)
                    english.append(f"{en} app")
        # required_tokens: intake keywords에서 추출
        required_tokens = []
        for ek in en_keywords:
            required_tokens.extend(ek.lower().split())
        required_tokens = list(set(t for t in required_tokens if len(t) >= 3))[:15]
        if not required_tokens:
            required_tokens = ["app", "tool", "service", "review", "alternative"]

    for s in seed:
        ss = (s or "").strip()
        if not ss:
            continue
        low = ss.lower()
        if re.search(r"[a-z]", low):
            ascii_ratio = sum(1 for ch in low if ord(ch) < 128) / max(1, len(low))
            if ascii_ratio >= 0.9:
                english.append(low)
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
        qq = re.sub(r"\s+", " ", q).strip().lower()
        if len(qq) < 3:
            continue
        if len(qq.split()) == 1:
            continue
        if not any(k in qq for k in required_tokens):
            continue
        if qq in seen:
            continue
        seen.add(qq)
        out.append(qq)
    return out[:12]


def is_relevant_text(text: str, threshold: int = 2, extra_terms: Optional[List[str]] = None,
                     relevance_terms: Optional[List[str]] = None,
                     problem_terms: Optional[List[str]] = None) -> bool:
    return relevance_score(text, extra_terms=extra_terms,
                           relevance_terms=relevance_terms,
                           problem_terms=problem_terms) >= threshold


# -----------------------------
# Data sources
# -----------------------------

def build_query_plan(intake: Dict[str, Any]) -> Dict[str, Any]:
    kws = intake.get("keywords", []) or []
    target = intake.get("target", "")
    problem = intake.get("problem", "")
    alt = intake.get("current_alternative", "")

    # 핵심 명사 추출: full sentence 대신 토큰화된 핵심 용어 사용
    topic_terms = extract_topic_terms(intake)
    # 쿼리용 base_terms: keywords + topic_terms (중복 제거)
    seen: set = set()
    base_terms: List[str] = []
    for t in kws + topic_terms:
        t = t.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            base_terms.append(t)
    # target/problem 원문은 너무 길면 포함하지 않음 (핵심 명사만 사용)
    if target and len(target) <= 15 and target.lower() not in seen:
        base_terms.insert(0, target)
    if alt and len(alt) <= 15 and alt.lower() not in seen:
        base_terms.append(alt)

    reddit_queries = []
    ddg_queries = []
    naver_queries = []
    ph_queries = []

    idea_mode = is_idea_validation_context(intake)

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

        # naver-friendly KR intent queries (context-aware)
        if idea_mode:
            naver_queries.extend([
                f"{term} 시장조사",
                f"{term} 시장성 검증",
                f"{term} 고객 인터뷰",
                f"{term} 유료 베타",
                f"{term} 가격 실험",
                f"{term} 창업 후기",
            ])
        else:
            naver_queries.extend([
                f"{term} 후기",
                f"{term} 불편",
                f"{term} 대안",
                f"{term} 유료",
                f"{term} 가격",
                f"{term} 자동화",
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

            # 도메인 필터 완화: 알려진 도메인이 아니어도 텍스트가 충분하면 허용
            if not any(h in low_url for h in PREFERRED_DOMAIN_HINTS):
                if len(text_blob) < 10:
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
        ("cafearticle", "https://openapi.naver.com/v1/search/cafearticle.json"),
    ]:
        try:
            r = requests.get(endpoint, headers=headers, params={"query": query, "display": display, "sort": "date"}, timeout=20)
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

def score_report(cards: List[EvidenceCard], intake: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not cards:
        return {
            "demand_score": 0,
            "confidence": "low",
            "decision": "iterate",
            "stats": {},
        }

    # topic_terms for fit_rate calculation
    topic_terms = extract_topic_terms(intake) if intake else []

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

    # Fit rate: topic_terms와 매칭되는 카드 비율
    def is_topic_relevant(c: EvidenceCard) -> bool:
        if not topic_terms:
            return True  # topic_terms 없으면 전부 relevant 취급
        blob = f"{c.title} {c.quote}".lower()
        hit = sum(1 for t in topic_terms if t.lower() in blob)
        return hit >= 2

    # Voice rate: 커뮤니티/리뷰/댓글 소스 비율
    COMMUNITY_SOURCES = {"reddit", "reddit-web", "threads", "threads-manual", "hackernews",
                         "apify-reddit", "apify-threads", "reddit-manual"}
    community_count = sum(1 for c in cards if c.source.lower() in COMMUNITY_SOURCES or c.source.startswith("apify-reddit"))

    relevant_cards = [c for c in cards if is_topic_relevant(c)]
    fit_rate = len(relevant_cards) / max(1, len(cards))
    voice_rate = community_count / max(1, len(cards))

    weights = [max(0.3, source_weight(c.source) * recency_weight(c.meta)) for c in cards]
    w_total = sum(weights)

    pain_w = sum(w for c, w in zip(cards, weights) if c.pain_tags)
    wtp1_only_w = sum(w for c, w in zip(cards, weights) if c.wtp == 1)
    wtp2_w = sum(w for c, w in zip(cards, weights) if c.wtp >= 2)
    workaround_w = sum(w for c, w in zip(cards, weights) if c.workaround_tags)
    sources = len(set(c.source for c in cards))

    pain_ratio = pain_w / max(1e-9, w_total)
    wtp_ratio = (wtp1_only_w + 2 * wtp2_w) / max(1e-9, (2 * w_total))  # 0~1 범위, wtp2에 2배 가중
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
    # confidence에 fit_rate + voice_rate 반영
    if total >= 30 and sources >= 4 and fit_rate >= 0.6 and voice_rate >= 0.4:
        conf = "high"
    elif total >= 12 and sources >= 2 and fit_rate >= 0.4:
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
            "fit_rate": round(fit_rate, 3),
            "voice_rate": round(voice_rate, 3),
            "weighted_total": round(w_total, 2),
            "weighted_pain": round(pain_w, 2),
            "weighted_wtp1_only": round(wtp1_only_w, 2),
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
            "more_evidence_needed": [{"goal": "돈 낼 의사 신호 보강", "how_to_collect": "가격/유료 언급 키워드 중심 추가 수집"}],
        }
    return out


def default_template_pack(intake: Dict[str, Any]) -> Dict[str, Any]:
    """intake 기반으로 기본 템플릿 팩을 동적 생성한다 (trading 하드코딩 제거)."""
    target = str(intake.get("target") or "핵심 타겟")
    problem = str(intake.get("problem") or "핵심 문제")
    alt = str(intake.get("current_alternative") or "기존 방법")
    outcome = str(intake.get("promised_outcome") or "문제 해결")
    return {
        "threads_posts": [
            {"type": "save", "text": f"[{target}] {problem} 해결하려고 할 때 가장 불편한 점 1가지만 알려주세요."},
            {"type": "repost", "text": f"{alt} 대신 {outcome} 할 수 있다면 어떤 기능이 가장 우선일까요?"},
            {"type": "recruit", "text": f"{target} 대상 유료베타 10명 모집합니다. 가장 필요한 기능 댓글로 주세요."},
        ],
        "reddit_posts": [
            {"type": "question", "title": f"{problem} - 어떻게 해결하고 계신가요?", "body": f"{alt} 외에 더 좋은 방법이 있는지 의견 부탁드립니다."},
            {"type": "feedback", "title": f"{outcome} 유료베타 피드백 부탁드립니다", "body": f"실제로 결제할 가치가 있는지 의견 부탁드립니다."},
        ],
        "landing_copy": [
            {"variant": "short", "headline": f"{problem} 해결", "subheadline": f"{alt} 대신 더 쉽게", "bullets": ["자동화", "대시보드", "퍼포먼스 추적"], "cta": "유료베타 신청"},
            {"variant": "medium", "headline": f"{target}을 위한 실행형 솔루션", "subheadline": outcome, "bullets": ["데이터 기반 분석", "인사이트 자동화", "시간 절감"], "cta": "지금 검증하기"},
            {"variant": "strong", "headline": f"{outcome}", "subheadline": f"기록이 아니라 의사결정을 돕는 서비스", "bullets": ["원인 분석", "패턴 감지", "다음 액션 추천"], "cta": "48시간 리포트 받기"},
        ],
        "faqs": [
            {"q": f"기존 방법({alt})과 뭐가 다른가요?", "a": f"자동화와 인사이트로 {problem} 해결 시간을 줄입니다.", "citations": []},
            {"q": "누구에게 맞나요?", "a": f"{target}에게 적합합니다.", "citations": []},
            {"q": "유료 전환 기준은요?", "a": "시간 절감/성과 개선 체감이 명확할 때 전환 가능성이 높습니다.", "citations": []},
            {"q": "데이터는 안전한가요?", "a": "민감정보 최소 수집과 접근제어를 적용합니다.", "citations": []},
            {"q": "모바일도 되나요?", "a": "핵심 기능은 모바일에서도 확인 가능하도록 설계합니다.", "citations": []},
            {"q": "초보자도 쓸 수 있나요?", "a": "템플릿 기반 입력으로 시작 장벽을 낮춥니다.", "citations": []},
            {"q": "기존 도구와 병행 가능한가요?", "a": "초기에는 병행 사용 후 점진 전환을 권장합니다.", "citations": []},
            {"q": "리포트는 얼마나 자주 받나요?", "a": "주간 리포트 기본, 필요시 일간 요약 확장 가능합니다.", "citations": []},
            {"q": "환불/보장 정책은?", "a": "유료베타는 재분석 1회 제공 같은 리스크 완화 정책을 권장합니다.", "citations": []},
            {"q": "지금 바로 뭘 해야 하나요?", "a": "타겟 1개 세그먼트로 랜딩/게시글 실험을 7일 실행하세요.", "citations": []},
        ],
    }


def module_template_pack(intake: Dict[str, Any], offers: Dict[str, Any], cards: List[EvidenceCard]) -> Dict[str, Any]:
    payload = {"project_brief": intake, "offers": offers, "evidence": [asdict(c) for c in cards[:30]]}
    schema = '{"confidence":"low|medium|high","threads_posts":[{"type":"save|recruit|repost","text":""}],"reddit_posts":[{"type":"question|feedback","title":"","body":""}],"landing_copy":[{"variant":"short|medium|strong","headline":"","subheadline":"","bullets":[""],"cta":""}],"faqs":[{"q":"","a":"","citations":[""]}]}'
    out = llm_json_call("Create copy-paste template pack", schema, payload) or {"confidence": "low", "threads_posts": [], "reddit_posts": [], "landing_copy": [], "faqs": []}

    # enforce minimum deliverables using intake-based defaults
    defaults = default_template_pack(intake)
    if len(out.get("threads_posts", [])) < 3:
        out["threads_posts"] = (out.get("threads_posts", []) + defaults["threads_posts"])[:3]
    if len(out.get("reddit_posts", [])) < 2:
        out["reddit_posts"] = (out.get("reddit_posts", []) + defaults["reddit_posts"])[:2]
    if len(out.get("landing_copy", [])) < 3:
        out["landing_copy"] = (out.get("landing_copy", []) + defaults["landing_copy"])[:3]
    if len(out.get("faqs", [])) < 10:
        out["faqs"] = (out.get("faqs", []) + defaults["faqs"])[:10]
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
            "why_not_now": [{"reason": "돈 낼 의사 강신호 부족", "citations": []}],
            "pivot_candidates": [
                {"pivot_title": "문제 피봇", "what_changes": "problem", "new_angle": "현재 문제 정의를 더 구체적인 페인 포인트로 재설정", "why_this_might_work": "구체적 문제일수록 결제 의향 높음", "supporting_evidence": [], "next_test": "재정의된 문제 기반 오퍼 게시"},
                {"pivot_title": "타겟 피봇", "what_changes": "target", "new_angle": "타겟 세그먼트를 더 좋첨", "why_this_might_work": "좋은 세그먼트일수록 전환률 높음", "supporting_evidence": [], "next_test": "좋은 세그먼트 전용 랜딩"},
                {"pivot_title": "오퍼 피봇", "what_changes": "offer", "new_angle": "지속 구독 대신 1회성 결과물 중심 패키지", "why_this_might_work": "돈/시간 절감 가치 명확", "supporting_evidence": [], "next_test": "1회성 결과물 결제 실험"},
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
            "name": "돈 낼 의사 직접 문장 비율 5% 이상",
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


def quick_translate_en_to_ko(text: str) -> str:
    api_key = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return text
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            max_tokens=220,
            messages=[
                {"role": "system", "content": "Translate concise EN->KO. Keep meaning, no extra commentary."},
                {"role": "user", "content": text[:500]},
            ],
        )
        out = (resp.choices[0].message.content or "").strip()
        return out or text
    except Exception:
        return text


def build_global_signals(cards: List[EvidenceCard], limit: int = 5) -> List[Dict[str, Any]]:
    en_sources = {"hackernews", "duckduckgo", "reddit", "reddit-web", "apify-reddit"}
    candidates: List[EvidenceCard] = []
    seen = set()
    for c in cards:
        # accept explicit EN-heavy cards regardless of source
        blob = f"{c.title} {c.quote}".strip()
        en_heavy = (sum(1 for ch in blob if ('a' <= ch.lower() <= 'z')) / max(1, len(blob))) > 0.35
        if c.source not in en_sources and not c.source.startswith("apify-reddit") and not en_heavy:
            continue
        blob = f"{c.title} {c.quote}".strip()
        if not re.search(r"[a-zA-Z]", blob):
            continue
        if c.source_url in seen:
            continue
        seen.add(c.source_url)
        candidates.append(c)

    # prioritize WTP/workaround/pain bearing signals
    def _score(c: EvidenceCard) -> float:
        s = 0.0
        if c.wtp >= 2:
            s += 4
        elif c.wtp == 1:
            s += 2
        if c.workaround_tags:
            s += 1.5
        if c.pain_tags:
            s += 1.0
        return s

    candidates.sort(key=_score, reverse=True)
    out: List[Dict[str, Any]] = []
    for c in candidates[:limit]:
        en_quote = compact_text(c.quote, 260)
        ko_quote = quick_translate_en_to_ko(en_quote)
        signal = "문제 공감"
        if c.wtp > 0:
            signal = "결제 의향"
        elif c.workaround_tags:
            signal = "대안 한계"
        out.append({
            "source": c.source,
            "url": c.source_url,
            "quote_en": en_quote,
            "quote_ko": ko_quote,
            "signal": signal,
        })
    return out


def build_decision_evidence(cards: List[EvidenceCard], intake: Optional[Dict[str, Any]] = None, limit: int = 10) -> List[Dict[str, Any]]:
    intake = intake or {}

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

    focus_terms = build_dynamic_terms(intake)
    required_intent_terms = [
        "idea validation", "market need", "willingness to pay", "startup validation", "mvp", "pmf",
        "시장성", "검증", "고객 인터뷰", "유료 베타", "수요", "니즈", "수익화", "가격 실험",
    ]
    negative_noise = ["주식 추천", "종목", "수익 인증", "코인 시황", "급등", "테마주", "리딩방", "밈주식", "공매도"]
    strict_offtopic_terms = ["코스닥", "코스피", "공매", "아파트", "부동산", "2차전지", "현대차", "lg전자", "dc-link", "capex"]
    strict_mode = is_idea_validation_context(intake)

    scored: List[Tuple[float, EvidenceCard]] = []
    for c in cards:
        text = f"{c.title} {c.quote}".lower()
        score = 0.0

        # hard reject noisy market commentary not tied to user problem
        if any(n in text for n in [n.lower() for n in negative_noise]):
            continue
        if strict_mode and any(n in text for n in [n.lower() for n in strict_offtopic_terms]):
            continue

        # require at least one clear intent term for this service type
        if not any(t in text for t in [t.lower() for t in required_intent_terms]):
            continue

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

        # explicit buying language
        if any(k in text for k in ["가격", "유료", "결제", "구독", "price", "pay", "pricing", "돈 내", "willingness to pay"]):
            score += 1.8

        # explicit alternative/switching signal
        if any(k in text for k in ["엑셀", "노션", "hts", "excel", "notion", "manual", "대안", "alternatives"]):
            score += 1.0

        # service-fit: must overlap with user focus terms
        fit = relevance_score(text, extra_terms=focus_terms)
        score += min(3.0, fit * 0.4)

        score += recency_bonus(c.meta)

        if len((c.quote or "").strip()) < 50:
            score -= 1.2

        # minimum bar to avoid vague/irrelevant cards
        if score < 2.4:
            continue

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
    idea_mode = is_idea_validation_context(intake)
    if idea_mode:
        segments = intake.get("segments") or [
            "1인사업가",
            "초기 스타트업 대표",
            "예비창업자",
        ]
    else:
        target = str(intake.get("target") or "핵심 타겟")
        problem = str(intake.get("problem") or "핵심 문제")
        outcome = str(intake.get("promised_outcome") or "문제 해결")
        segments = intake.get("segments") or [
            target,
            f"{target} (초보)",
            f"{target} (고도 사용자)",
        ]

    out = []
    for seg in segments[:3]:
        if idea_mode:
            posts = [
                {
                    "type": "공감형",
                    "text": f"{seg}분들, 아이디어 시장성을 지금 어떻게 검증하고 계신가요? 가장 막히는 1가지만 알려주세요.",
                },
                {
                    "type": "인사이트형",
                    "text": f"{seg} 기준으로 문제 공감은 높은데 돈 낼 의사 문장은 부족합니다. 지금 당장 유료 검증에 필요한 질문 3개를 댓글로 받을게요.",
                },
                {
                    "type": "모집형",
                    "text": f"{seg} 대상 유료베타 검증 10명 모집: 48시간 내 Go/Iterate/Pivot 판단 리포트. 관심 있으면 DM에 BETA 남겨주세요.",
                },
            ]
        else:
            posts = [
                {
                    "type": "공감형",
                    "text": f"{seg}분들, {problem} 해결할 때 가장 시간을 잡아먹는 구간이 어디인가요?",
                },
                {
                    "type": "인사이트형",
                    "text": f"{seg} 기준으로 보면 문제 체감은 높은데 지불 의향은 낮습니다. {outcome}에서 꼭 필요한 기능 3개만 알려주세요.",
                },
                {
                    "type": "모집형",
                    "text": f"{seg} 대상 유료베타 10명 모집. 관심 있으면 DM에 BETA 남겨주세요.",
                },
            ]
        out.append({"segment": seg, "posts": posts})
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
    # sprint_metrics가 없거나 데이터가 모두 0이면 재판정 생략
    has_data = any(int(m.get(k, 0) or 0) > 0 for k in ["landing_visits", "cta_clicks", "applications", "interviews", "dm_comments"])
    if not m or not has_data:
        return {
            "base_decision": base_decision,
            "rejudged_decision": base_decision,
            "status": "not_run",
            "reason": "스프린트 실험 데이터가 없어 재판정을 실행하지 않았습니다.",
            "pass_count": 0,
            "total_checks": 0,
            "weighted_pass_ratio": 0.0,
            "hard_go_conditions": {},
            "metrics": {},
            "checks": [],
        }
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
        {"name": "돈 낼 의사 문장>=3", "pass": wtp_sentences >= 3, "value": wtp_sentences, "weight": 1.5},
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


def sanitize_domain_terms(text: str, intake: Dict[str, Any], cards: List[EvidenceCard]) -> str:
    """intake에 등장하지 않는 도메인 전문용어를 제거한다. blocked 리스트를 동적으로 구성."""
    low_intake = " ".join([
        str(intake.get("target") or ""), str(intake.get("problem") or ""),
        str(intake.get("current_alternative") or ""), " ".join(intake.get("keywords") or []),
    ]).lower()
    low_ev = " ".join((c.title + " " + c.quote) for c in cards[:50]).lower()
    allow_blob = low_intake + " " + low_ev
    # 동적 blocked: intake에 없는 도메인 특화 용어만 차단
    blocked: List[str] = []
    out = text
    for b in blocked:
        if b not in allow_blob:
            out = re.sub(re.escape(b), "", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def build_execution_pack(intake: Dict[str, Any], score: Dict[str, Any], insight: Dict[str, Any], sprint_rejudge: Dict[str, Any], top_pains: List[Tuple[str, int]], cards: Optional[List[EvidenceCard]] = None) -> Dict[str, Any]:
    target = intake.get("target") or "핵심 타겟"
    idea_mode = is_idea_validation_context(intake)
    top_pain = top_pains[0][0] if top_pains else ("시장성/수익화 판단" if idea_mode else "기록/복기")
    if idea_mode and (not top_pain or top_pain.lower() in {"time", "pain", "problem"}):
        top_pain = "시장성/수익화 판단"
    decision = sprint_rejudge.get("rejudged_decision") or score.get("decision", "iterate")
    apply_rate = (sprint_rejudge.get("metrics") or {}).get("application_rate", 0.0)
    wtp_ratio = insight.get("wtp_direct_ratio", 0.0)

    if decision == "go":
        tone = "검증 통과. 전환 실험을 확장합니다."
    elif decision == "iterate":
        tone = "핵심 가설 유지, 오퍼 문구/세그먼트 조정이 필요합니다."
    else:
        tone = "오퍼 피벗 권장. 결과물 중심 패키지로 재실험합니다."

    measured = any((sprint_rejudge.get("metrics") or {}).get(k, 0) > 0 for k in ["landing_visits", "cta_clicks", "applications", "interviews"])

    if idea_mode:
        if measured:
            post = (
                f"[{target}] 지금 '{top_pain}' 문제를 해결할 아이디어 시장성 검증 유료베타를 테스트 중입니다. "
                f"신청률 {apply_rate:.2%}, 돈 낼 의사 문장 비율 {wtp_ratio:.2%}. 관심 있으면 DM에 BETA 남겨주세요."
            )
        else:
            post = f"[{target}] 지금 '{top_pain}' 관련 사용자 인터뷰를 모집 중입니다. 실제 불편/대안/가격 인식 확인을 위해 DM 참여 부탁드려요."
        dm_script = [
            "문의 감사합니다. 15분만 인터뷰하고 베타 우선 초대 드릴게요.",
            "지금 아이디어 검증에서 가장 막히는 1가지를 알려주세요.",
            "이 문제가 해결된다면 실제로 지불 가능한 가격대를 알려주세요.",
        ]
        tweak_rule = "신청률<2%면 타겟 문구를 더 좁히고, 돈 낼 의사 문장<3이면 가격 질문을 첫 문장으로 이동"
    else:
        problem = str(intake.get("problem") or top_pain)
        outcome = str(intake.get("promised_outcome") or "문제 해결")
        post = (
            f"[{target}] 지금 {top_pain} 문제를 해결할 유료베타를 검증 중입니다. "
            f"신청률 {apply_rate:.2%}, 돈 낼 의사 비율 {wtp_ratio:.2%}. 관심 있으면 DM에 BETA 남겨주세요."
        )
        dm_script = [
            "문의 감사합니다. 15분만 인터뷰하고 베타 우선 초대 드릴게요.",
            f"현재 {problem} 해결에서 가장 큰 불편 1가지를 알려주세요.",
            f"{outcome}이 해결된다면 월 얼마가 적정한지 알려주세요.",
        ]
        tweak_rule = "신청률<2%면 혜택문구 2줄 교체, 돈 낼 의사 문장<3이면 가격 질문 문구를 전면에 배치"

    post = sanitize_domain_terms(post, intake, cards or [])

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


def heuristic_summary(intake: Dict[str, Any], score: Dict[str, Any], stats: Dict[str, Any]) -> Tuple[str, str]:
    """LLM 없이도 why_this_score / actionable_next_step을 채우는 fallback."""
    evidence_count = stats.get("evidence_count", 0)
    fit_rate = stats.get("fit_rate", 0.0)
    wtp_count = stats.get("wtp_level1_plus", 0)
    source_count = stats.get("source_count", 0)
    decision = score.get("decision", "iterate")

    if evidence_count < 20 or fit_rate < 0.4:
        why = "데이터가 충분하지 않거나 주제와 직접 관련된 근거 비율이 낮아서 결론을 강하게 내리기 어렵습니다."
        next_step = "커뮤니티/리뷰/댓글에서 '문제+대안+가격'이 같이 언급된 근거를 30개 이상 모은 뒤 재판정하세요."
        return why, next_step

    if wtp_count == 0:
        why = "문제 공감은 일부 확인되지만, '이걸 위해 돈을 내겠다'는 문장이 거의 없어 유료 서비스로는 근거가 약합니다."
        next_step = "가격 질문이 포함된 모집글/랜딩(A/B)로 '돈 낼 의사' 문장을 3개 이상 확보하세요."
        return why, next_step

    if decision == "go":
        why = f"근거 {evidence_count}건, 적합도 {fit_rate:.0%}, 돈 낼 의사 신호 {wtp_count}건으로 양호한 수준입니다."
        next_step = "결제 의향 검증용 스모크 테스트 랜딩(가격 3안) + 72시간 트래픽 테스트를 실행하세요."
    elif decision == "iterate":
        why = f"근거의 질과 양은 확인되지만 (적합도 {fit_rate:.0%}, 돈 낼 의사 {wtp_count}건), 결제 전환 근거가 부족해 오퍼/타겟 재정의가 필요합니다."
        next_step = "가장 반응이 강한 세그먼트 1개로 좁혀 7일 스프린트를 실행하세요."
    else:
        why = f"적합도 {fit_rate:.0%}, 돈 낼 의사 신호 {wtp_count}건으로 현 가설의 시장 수요 근거가 약합니다."
        next_step = "핵심 문제 가설을 재정의하고 다른 세그먼트(타겟)로 쿼리를 재수집하세요."

    return why, next_step


def build_report(intake: Dict[str, Any], cards: List[EvidenceCard], score: Dict[str, Any]) -> Dict[str, Any]:
    top_pains = top_tags(cards, "pain_tags", 3)
    top_workarounds = top_tags(cards, "workaround_tags", 3)
    top_wtp_quotes = [c.quote for c in cards if c.wtp > 0][:5]

    llm = llm_upgrade_summary(intake, score, cards)

    # LLM이 비었을 때 heuristic fallback
    if not llm.get("why_this_score") or not llm.get("actionable_next_step"):
        h_why, h_next = heuristic_summary(intake, score, score.get("stats", {}))
        if not llm.get("why_this_score"):
            llm["why_this_score"] = h_why
        if not llm.get("actionable_next_step"):
            llm["actionable_next_step"] = h_next
        if not llm.get("executive_one_liner"):
            decision = score.get("decision", "iterate")
            if decision == "go":
                llm["executive_one_liner"] = "핵심 가설이 검증되어 실행 단계로 진입할 수 있습니다."
            elif decision == "iterate":
                llm["executive_one_liner"] = "가설 유지, 오퍼/타겟 문구 조정 후 재검증이 필요합니다."
            else:
                llm["executive_one_liner"] = "현재 근거로는 피벗이 권장됩니다. 문제/타겟 재정의를 검토하세요."

    # Cost optimization: run fewer LLM modules when evidence is sparse or budget mode is on.
    budget_mode = os.getenv("COST_OPTIMIZED", "1") == "1"
    enough_evidence = len(cards) >= int(os.getenv("MIN_EVIDENCE_FOR_FULL_LLM", "25"))

    replacement_map = module_replacement_map(intake, cards)
    quality = module_quality_explainer(intake, score, cards)
    # hard guardrail override
    st0 = score.get("stats", {})
    wtp_direct = sum(1 for c in cards if c.wtp > 0)
    allow_strong = (st0.get("evidence_count", 0) >= 30) and (st0.get("source_count", 0) >= 4) and (wtp_direct >= 3) and (score.get("confidence") != "low")
    quality.setdefault("conclusion_guardrail", {})
    quality["conclusion_guardrail"]["allow_strong_conclusion"] = bool(allow_strong)
    if not allow_strong:
        quality["conclusion_guardrail"]["reason"] = "증거 수/소스/돈 낼 의사 문장 기준 미달"

    if budget_mode and not enough_evidence:
        offers = {"confidence": "low", "offers": [], "risk_reversal": [], "pricing_experiments": []}
        plan_7d = {"confidence": "low", "experiment_goal": "", "kpi_cutlines": [], "day_by_day_plan": [], "pivot_rules": [], "templates": {}}
        if is_idea_validation_context(intake):
            template_pack = {
                "confidence": "medium",
                "threads_posts": [
                    {"type": "question", "text": "아이디어 시장성 검증, 다들 지금 어떻게 하고 계세요? 감/지인 피드백 말고 실제로 통했던 방식 1개만 알려주세요."},
                    {"type": "recruit", "text": "1인사업가/초기팀 대상 아이디어 검증 인터뷰 10명 모집합니다. 15분 인터뷰 참여 가능하면 DM 주세요."},
                    {"type": "pricing", "text": "‘48시간 시장성 검증 리포트’가 있다면 얼마까지 지불 의향 있으세요? (무료/1만원대/3만원대/5만원+)"},
                ],
                "reddit_posts": [],
                "landing_copy": [],
                "faqs": [],
            }
        else:
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
            "돈 낼 의사 강신호(유료/가격) 증거를 추가 확보",
            "예약금/유료베타 전환 지표 확보",
            "핵심 세그먼트 1개에서 반복 반응 확인",
        ],
    }
    insight_metrics = compute_insight_metrics(cards, score, score_breakdown)
    self_checklist = build_self_checklist(insight_metrics)
    segment_posts = build_segment_posts(intake)
    interview_tagging = tag_interview_answers(intake)
    sprint_rejudge = rejudge_sprint(intake, score.get("decision", "iterate"))
    execution_pack = build_execution_pack(intake, score, insight_metrics, sprint_rejudge, top_pains, cards)
    decision_evidence = build_decision_evidence(cards, intake=intake, limit=10)
    global_signals = build_global_signals(cards, limit=5)

    # guardrail: when data is too thin, avoid strong conclusion language
    min_evidence = int(os.getenv("MIN_EVIDENCE_GATE", "30"))
    min_sources = int(os.getenv("MIN_SOURCES_GATE", "4"))
    insufficient_data = (stats.get("evidence_count", 0) < min_evidence) or (stats.get("source_count", 0) < min_sources)
    gated_decision = score["decision"]
    gated_next_step = llm.get("actionable_next_step", "")
    gated_why = llm.get("why_this_score", "")
    if insufficient_data:
        gated_decision = "collect_more"
        gated_why = gated_why or "현재 근거 수/소스 다양성이 기준 미달이라 판정을 보류하고 추가 수집이 필요합니다."
        gated_next_step = "타겟 적합 증거 30건, 소스 4개 이상 확보 후 재판정"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_id": build_project_id(intake),
        "intake": intake,
        "summary": {
            "demand_score": score["demand_score"],
            "confidence": score["confidence"],
            "decision": gated_decision,
            "one_liner": llm.get("executive_one_liner") or f"수요 신호 {('강' if score['demand_score'] >= 70 else '중' if score['demand_score'] >=45 else '약')} / confidence {score['confidence']}",
            "why_this_score": gated_why,
            "actionable_next_step": gated_next_step,
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
        "global_signals": global_signals,
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
    lines.append(f"- 돈 낼 의사(가격/유료) 신호: **{stats.get('wtp_level1_plus', 0)}**")
    lines.append(f"- 강한 돈 낼 의사 신호: **{stats.get('wtp_level2', 0)}**")
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
        lines.append(f"  - 돈 낼 의사 문장 비율: {im.get('wtp_direct_count')}/{im.get('evidence_count')} = {im.get('wtp_direct_ratio')}")
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
        lines.append("- 돈 낼 의사(가격/유료) 관련 근거 문장:")
        for q in wtp_quotes[:3]:
            lines.append(f"  - \"{q}\"")
    else:
        lines.append("- 돈 낼 의사 신호가 약합니다. 가격 검증 질문을 포함한 추가 탐색이 필요합니다.")

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
    lines.append(f"- 태깅 집계: Pain {tc.get('pain',0)} / 돈 낼 의사 {tc.get('wtp',0)} / Workaround {tc.get('workaround',0)}")
    lines.append(f"- 돈 낼 의사 비율: {it.get('wtp_ratio',0.0)}")

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
            lines.append(f"- KPI: 신청률>={k.get('apply_rate_target',0)} / DM>={k.get('dm_target',0)} / 돈 낼 의사 문장>={k.get('wtp_sentence_target',0)}")
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
    lines.append("- 신뢰도를 올리려면: 소스 다양성 확대 + 돈 낼 의사 문장 추가 수집이 필요합니다.")

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

    # Reddit (temporarily disabled by default for quality control)
    disable_reddit = os.getenv("DISABLE_REDDIT", "1") == "1"
    if disable_reddit:
        print("[info] Reddit disabled by DISABLE_REDDIT=1", file=sys.stderr)
    elif not args.no_reddit:
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
        apify_reddit_enabled = os.getenv("DISABLE_REDDIT", "1") != "1"
        apify_task_map = [
            ("reddit", os.getenv("APIFY_TASK_REDDIT") if apify_reddit_enabled else None),
            ("naver", os.getenv("APIFY_TASK_NAVER")),
            ("youtube", os.getenv("APIFY_TASK_YOUTUBE")),
        ]
        use_actor_reddit = (os.getenv("APIFY_USE_ACTOR_REDDIT", "0") == "1") and apify_reddit_enabled
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
    rel_terms = build_relevance_terms(intake)
    prob_terms = build_problem_terms(intake)
    idea_mode = is_idea_validation_context(intake)
    filtered: List[EvidenceCard] = []
    dropped = 0
    for c in cards:
        # keep user-manual links always
        if c.meta.get("manual"):
            filtered.append(c)
            continue

        blob = f"{c.title} {c.quote}".strip()

        # stricter base threshold to reduce noise
        th = int(os.getenv("RELEVANCE_MIN", "3"))
        if c.source in {"duckduckgo", "reddit-web", "threads"}:
            th = max(th, 4)

        # reddit source quality filter (idea_mode일 때는 서브레딧 필터 비활성화)
        if c.source.startswith("apify-reddit") or c.source in {"reddit", "reddit-web"}:
            u = (c.source_url or "").lower()
            if not idea_mode and "/r/" in u and not any(f"/r/{sub}" in u for sub in REDDIT_ALLOWED_SUBS):
                dropped += 1
                continue

        # relax threshold slightly when dynamic terms exist (generic ideas)
        if dynamic_terms and c.source.startswith("apify-"):
            th = max(1, th - 1)

        if is_relevant_text(blob, threshold=th, extra_terms=dynamic_terms,
                            relevance_terms=rel_terms, problem_terms=prob_terms):
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

    # WTP 재태깅: intake의 topic_terms 기반으로 모든 카드의 WTP를 다시 판정
    topic_terms = extract_topic_terms(intake)
    retag_count = 0
    for c in cards:
        blob = f"{c.title} {c.quote}".strip()
        new_wtp = infer_wtp(blob, topic_terms=topic_terms)
        if new_wtp != c.wtp:
            retag_count += 1
            c.wtp = new_wtp
    if retag_count:
        print(f"[info] WTP re-tagged {retag_count} card(s) with topic context", file=sys.stderr)

    score = score_report(cards, intake=intake)
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
