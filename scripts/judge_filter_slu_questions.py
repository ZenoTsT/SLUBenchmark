import os
import json
import time
import random
import argparse
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from openai import OpenAI

from scripts.cost_utils import (
    extract_usage_from_response,
    estimate_cost,
    empty_usage,
)

#                              comand to run this script:                                #
#                                                                                        #
# python scripts/judge_filter_slu_questions.py \                                         #
#  --input results/slu_questions_promptWexamples_gpt-5.1_none_1280.json \                #
#  --prompt prompts/question_generation_shortprompt.txt \                                #
#  --output_prefix results/judged_v1 \                                                   #
#  --model gpt-5.1 \                                                                     #
#  --concurrency 8                                                                       #           
#                                                                                        #
#                                                                                        #
# one line: python scripts/judge_filter_slu_questions.py --input results/slu_questions_promptWexamples_gpt-5.1_none_1280.json --prompt prompts/question_generation_shortprompt.txt --output_prefix results/judged_v1 --model gpt-5.1 --concurrency 8

ALLOWED_CATEGORIES = {"temporal", "logical", "pragmatic", "details", "global"}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def compute_validators(q: Dict[str, Any]) -> Dict[str, Any]:
    correct = q.get("correct_answer", "")
    distractors = q.get("distractors", [])
    validators = {}

    validators["has_4_distractors"] = isinstance(distractors, list) and len(distractors) == 4

    def norm(s: str) -> str:
        return " ".join(str(s).strip().lower().split())

    options = [correct] + (distractors if isinstance(distractors, list) else [])
    norm_opts = [norm(o) for o in options if isinstance(o, str)]

    validators["no_duplicate_options"] = len(norm_opts) == len(set(norm_opts))
    validators["correct_not_in_distractors"] = (
        isinstance(correct, str)
        and isinstance(distractors, list)
        and norm(correct) not in {norm(d) for d in distractors if isinstance(d, str)}
    )

    lengths = [len(str(o).split()) for o in options]
    if lengths:
        mn, mx = min(lengths), max(lengths)
        validators["option_length_min_words"] = mn
        validators["option_length_max_words"] = mx
        validators["option_length_ratio_ok"] = (mn == 0) or (mx / max(mn, 1) <= 2.5)
    else:
        validators["option_length_ratio_ok"] = False

    return validators


def build_judge_input(category: str, sentence: str, q: Dict[str, Any], validators: Dict[str, Any]) -> str:
    payload = {
        "category": category,
        "sentence": sentence,
        "question": q.get("question", ""),
        "correct_answer": q.get("correct_answer", ""),
        "distractors": q.get("distractors", []),
        "validators": validators,
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_verdict(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip().upper()
    first = t.split()[0] if t else ""
    return first if first in {"YES", "NO"} else ""


def is_retryable_error(e: Exception) -> bool:
    msg = str(e).lower()
    retry_signals = [
        "rate limit", "429",
        "timeout", "timed out",
        "temporarily unavailable",
        "server error", "500", "502", "503", "504",
        "connection reset", "connection aborted", "connection error",
    ]
    return any(s in msg for s in retry_signals)


def call_judge(
    client: OpenAI,
    model: str,
    prompt_text: str,
    judge_input_str: str,
    reasoning_effort: str = "none",
    max_output_tokens: int = 5,
    max_retries: int = 3,
    base_backoff: float = 0.8,
) -> Tuple[str, Dict[str, int], Dict[str, float], str]:
    """
    Returns:
      verdict: "YES"/"NO"/""
      usage_dict: full usage dict (includes cached)
      cost_dict: estimated cost dict
      raw_output: raw output text
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.create(
                model=model,
                instructions=prompt_text,
                input=judge_input_str,
                reasoning={"effort": reasoning_effort},
                text={"verbosity": "low"},
                max_output_tokens=max_output_tokens,
            )

            if getattr(resp, "status", None) != "completed":
                raise RuntimeError(f"not completed: {getattr(resp,'status',None)}")

            raw = getattr(resp, "output_text", "") or ""
            verdict = parse_verdict(raw)

            usage_dict = extract_usage_from_response(resp)
            cost_dict = estimate_cost(usage_dict)

            if verdict in {"YES", "NO"}:
                return verdict, usage_dict, cost_dict, raw

            raise ValueError(f"Invalid verdict output: {raw[:120]}")

        except Exception as e:
            last_err = str(e)
            if attempt < max_retries and is_retryable_error(e):
                sleep_s = min(2.0, base_backoff * (2 ** (attempt - 1))) + random.uniform(0.0, 0.2)
                time.sleep(sleep_s)
                continue
            break

    return "", empty_usage(), {"input_cost": 0.0, "cached_input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}, f"ERROR: {last_err}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file (generated questions)")
    parser.add_argument("--prompt", required=True, help="Judge prompt txt file")
    parser.add_argument("--output_prefix", default="judged", help="Prefix for outputs")
    parser.add_argument("--model", default="gpt-5.1", help="Model name")
    parser.add_argument("--concurrency", type=int, default=8, help="Parallel API calls")
    parser.add_argument("--max_output_tokens", type=int, default=5, help="Max output tokens for judge")
    parser.add_argument("--reasoning_effort", default="none", help="none/minimal/low/medium/high")
    args = parser.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY missing in env.")

    data = load_json(args.input)
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    sentences: List[Dict[str, Any]] = data.get("sentences", [])
    total_questions = sum(len(s.get("questions", []) or []) for s in sentences)

    print(f"Loaded sentences: {len(sentences)}")
    print(f"Total questions:  {total_questions}")
    print(f"Model: {args.model} | concurrency={args.concurrency}")

    tasks = []
    for si, s in enumerate(sentences):
        for qi, _ in enumerate(s.get("questions", []) or []):
            tasks.append((si, qi))

    verdict_counter = Counter()
    verdict_by_cat = Counter()
    cat_total = Counter()

    total_usage = empty_usage()
    total_cost = 0.0

    client = OpenAI()

    def process_one(task):
        si, qi = task
        s = sentences[si]
        q = (s.get("questions", []) or [])[qi]

        cat = q.get("category", "")
        sentence_text = s.get("text", "")

        if cat not in ALLOWED_CATEGORIES:
            return si, qi, "NO", {"bad_category": True}, empty_usage(), {"total_cost": 0.0}, "BAD_CATEGORY"

        validators = compute_validators(q)
        judge_input_str = build_judge_input(cat, sentence_text, q, validators)

        verdict, usage_dict, cost_dict, raw_output = call_judge(
            client=client,
            model=args.model,
            prompt_text=prompt_text,
            judge_input_str=judge_input_str,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            max_retries=3,
        )

        if verdict == "":
            verdict = "NO"
            raw_output = f"INVALID_JUDGE_OUTPUT: {raw_output}"

        return si, qi, verdict, validators, usage_dict, cost_dict, raw_output

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        with tqdm(total=len(tasks), desc="Judging") as pbar:
            for si, qi, verdict, validators, usage_dict, cost_dict, raw_output in ex.map(process_one, tasks):
                q = sentences[si]["questions"][qi]
                cat = q.get("category", "unknown")
                cat_total[cat] += 1

                q["judge"] = {
                    "verdict": verdict,
                    "validators": validators,
                    "usage": usage_dict,
                    "estimated_cost": cost_dict,
                    "raw_output": raw_output,
                }

                verdict_counter[verdict] += 1
                verdict_by_cat[(cat, verdict)] += 1

                # accumulate totals
                total_usage = {k: total_usage.get(k, 0) + usage_dict.get(k, 0) for k in total_usage.keys()}
                total_cost += float(cost_dict.get("total_cost", 0.0))

                pbar.update(1)

    # Create filtered/rejected (deep copy)
    filtered = json.loads(json.dumps(data))
    rejected = json.loads(json.dumps(data))

    for s in filtered["sentences"]:
        qs = s.get("questions", []) or []
        s["questions"] = [q for q in qs if (q.get("judge", {}).get("verdict") == "YES")]

    for s in rejected["sentences"]:
        qs = s.get("questions", []) or []
        s["questions"] = [q for q in qs if (q.get("judge", {}).get("verdict") == "NO")]

    # Meta additions
    data.setdefault("meta", {})
    data["meta"]["judge_model"] = args.model
    data["meta"]["judge_reasoning_effort"] = args.reasoning_effort
    data["meta"]["judge_max_output_tokens"] = args.max_output_tokens
    data["meta"]["judge_cost_summary"] = {
        "total_input_tokens": total_usage["input_tokens"],
        "total_cached_input_tokens": total_usage["cached_tokens"],
        "total_output_tokens": total_usage["output_tokens"],
        "estimated_total_cost_usd": total_cost,
        "pricing": {
            "price_per_m_input": 0.05,
            "price_per_m_cached_input": 0.005,
            "price_per_m_output": 0.4,
        },
    }

    out_all = f"{args.output_prefix}_all.json"
    out_yes = f"{args.output_prefix}_filtered_yes.json"
    out_no = f"{args.output_prefix}_rejected_no.json"

    save_json(out_all, data)
    save_json(out_yes, filtered)
    save_json(out_no, rejected)

    print("\n=== JUDGE SUMMARY ===")
    print(f"YES: {verdict_counter['YES']} | NO: {verdict_counter['NO']} | Total: {sum(verdict_counter.values())}")

    print("\nBy category:")
    for cat in ["temporal", "logical", "pragmatic", "details", "global"]:
        tot = cat_total.get(cat, 0)
        yes = verdict_by_cat.get((cat, "YES"), 0)
        no = verdict_by_cat.get((cat, "NO"), 0)
        if tot > 0:
            print(f"- {cat:9s}: YES {yes:5d} | NO {no:5d} | tot {tot:5d} | YES% {100*yes/tot:5.1f}")
        else:
            print(f"- {cat:9s}: (no questions)")

    print("\n=== COST SUMMARY (estimated, local) ===")
    print(f"Total input tokens:        {total_usage['input_tokens']}")
    print(f"Total cached input tokens: {total_usage['cached_tokens']}")
    print(f"Total output tokens:       {total_usage['output_tokens']}")
    print(f"Estimated total cost:      ${total_cost:.6f}")

    print(f"\nSaved:\n- {out_all}\n- {out_yes}\n- {out_no}\n")


if __name__ == "__main__":
    main()