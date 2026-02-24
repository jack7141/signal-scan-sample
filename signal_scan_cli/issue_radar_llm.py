#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
                pass
    return rows


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        text = " ".join(
            [
                str(r.get("title", "") or "").strip(),
                str(r.get("text", "") or "").strip(),
                str(r.get("body", "") or "").strip(),
                str(r.get("snippet", "") or "").strip(),
            ]
        ).strip()
        url = r.get("url") or r.get("link") or r.get("href") or ""
        source = (r.get("source") or r.get("platform") or "unknown").lower()
        if not text:
            continue
        out.append({"source": source, "text": text[:600], "url": url})
    return out


def fallback_collect_more(input_meta: dict[str, Any]) -> dict[str, Any]:
    card = {
        "title": "지금은 근거가 부족해서 결론 보류",
        "why_trending": [
            "수집된 커뮤니티 게시글 수가 적거나 관련도가 낮음",
            "현재 데이터로는 왜 뜨는지/어디가 뜨는지 단정하기 어려움",
        ],
        "top_links": [],
        "what_to_do": {
            "content": "키워드별로 질문형 포스트 3개를 먼저 올려 반응 수집",
            "product": "가장 많이 반복된 문제 1개만 선택해 초간단 해결안 테스트",
            "comment": "관련 커뮤니티 상위 글 10개에 경험 기반 댓글 남겨 반응 확인",
        },
        "score": 0,
        "evidence_count": 0,
    }
    return {
        "input": input_meta,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tabs": {"today": [card], "week": [card], "month": [card]},
        "decision": "collect_more",
    }


def build_with_openai(input_meta: dict[str, Any], rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sample = rows[:120]

    system = (
        "You generate a Korean issue radar JSON. "
        "Output valid JSON only. "
        "Need tabs: today/week/month, each up to 5 cards. "
        "Each card must contain: title, why_trending(array), top_links(max3), "
        "what_to_do{content,product,comment}, score, evidence_count. "
        "If evidence weak, still output one collect_more-style card per tab."
    )

    user = {
        "input": input_meta,
        "rows": sample,
        "instruction": "키워드/카테고리와 관련된 이슈만 남기고, 쉬운 한국어로 작성",
    }

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )
    text = resp.choices[0].message.content
    obj = json.loads(text)
    obj.setdefault("input", input_meta)
    obj.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    return obj


def main():
    ap = argparse.ArgumentParser(description="Issue radar with OpenAI post-processing")
    ap.add_argument("--category", required=True)
    ap.add_argument("--keywords", required=True)
    ap.add_argument("--sources", default="reddit,threads")
    ap.add_argument("--in", dest="input_jsonl", required=True)
    ap.add_argument("--out", dest="output_json", required=True)
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    args = ap.parse_args()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    sources = [s.strip().lower() for s in args.sources.split(",") if s.strip()]
    input_meta = {"category": args.category, "keywords": keywords, "sources": sources}

    rows = normalize_rows(load_jsonl(Path(args.input_jsonl)))

    if not rows:
        out = fallback_collect_more(input_meta)
    elif not os.getenv("OPENAI_API_KEY"):
        out = fallback_collect_more({**input_meta, "note": "OPENAI_API_KEY missing"})
    else:
        try:
            out = build_with_openai(input_meta, rows, model=args.model)
        except Exception as e:
            out = fallback_collect_more({**input_meta, "note": f"llm_error: {type(e).__name__}"})

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote {args.output_json}")


if __name__ == "__main__":
    main()
