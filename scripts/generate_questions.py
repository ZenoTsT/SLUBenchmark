import os
import json
import time
import random
import datetime
from typing import List, Dict, Any, Tuple, Optional

from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

from scripts.cost_utils import (
    extract_usage_from_response,
    estimate_cost,
    empty_usage,
)


# ---------------------------------
# CONFIG
# ---------------------------------
MODEL = "gpt-5.1"
REASONING_EFFORT = "none"  # "none" / "minimal" / "low" / "medium" / "high"

DATASET_PATH = "dataset/how2sign_dataset.json"
PROMPT_NAME = "promptWexamples"
PROMPT_PATH = f"prompts/{PROMPT_NAME}.txt"

MAX_SENTENCES = 10
OUTPUT_PATH = f"results/slu_questions_{PROMPT_NAME}_{MODEL}_{REASONING_EFFORT}_{MAX_SENTENCES}.json"

SAVE_EVERY = 2
COST_LOG_EVERY = 2
CONCURRENCY = 1
MAX_OUTPUT_TOKENS = 1024

MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 1.0


# ---------------------------------
# UTILS
# ---------------------------------

def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    splits = data["splits"]

    for split_name, items in splits.items():
        for ex in items:
            samples.append(
                {
                    "sentence_id": ex["sentence_id"],
                    "source_split": split_name,
                    "text": ex["sentence"],
                    "video_id": ex.get("video_id"),
                    "clip_id": ex.get("sentence_name"),
                }
            )
    return samples


def extract_output_text(response) -> str:
    txt = getattr(response, "output_text", None)
    if isinstance(txt, str) and txt.strip():
        return txt

    pieces = []
    output = getattr(response, "output", None)
    if output:
        for item in output:
            content = getattr(item, "content", None)
            if not content:
                continue
            for block in content:
                text_obj = getattr(block, "text", None)
                if isinstance(text_obj, str):
                    pieces.append(text_obj)
                elif hasattr(text_obj, "value"):
                    pieces.append(text_obj.value)

    return "".join(pieces).strip()


def strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


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


def call_model_on_sentence(
    client: OpenAI,
    prompt_text: str,
    sentence_text: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    input_obj = {"text": sentence_text}
    input_str = json.dumps(input_obj, ensure_ascii=False)

    last_exc: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL,
                instructions=prompt_text,
                input=input_str,
                reasoning={"effort": REASONING_EFFORT},
                text={"verbosity": "low"},
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )

            if getattr(response, "status", None) != "completed":
                raise RuntimeError(
                    f"Model response not completed. "
                    f"status={getattr(response, 'status', None)}, "
                    f"incomplete_details={getattr(response, 'incomplete_details', None)}"
                )

            usage_dict = extract_usage_from_response(response)
            output_text = extract_output_text(response)

            extra = {
                "raw_output": output_text[:6000] if output_text else "",
                "parse_error": None,
            }

            if not output_text or not output_text.strip():
                return [], usage_dict, extra

            cleaned = strip_code_fences(output_text)

            try:
                out_json = json.loads(cleaned)
            except json.JSONDecodeError as je:
                extra["parse_error"] = str(je)
                return [], usage_dict, extra

            if not isinstance(out_json, dict):
                return [], usage_dict, extra

            questions = out_json.get("questions", [])
            if not isinstance(questions, list):
                questions = []

            return questions, usage_dict, extra

        except Exception as e:
            last_exc = e
            if attempt < MAX_RETRIES and is_retryable_error(e):
                sleep_s = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                sleep_s += random.uniform(0, 0.25)
                time.sleep(sleep_s)
                continue
            raise e

    raise RuntimeError(f"Unexpected failure: {last_exc}")


# ---------------------------------
# MAIN
# ---------------------------------

def main():
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY non trovata nell'ambiente.")

    print("Loading dataset...")
    samples = load_dataset(DATASET_PATH)
    print(f"Total sentences in dataset: {len(samples)}")

    if MAX_SENTENCES is not None:
        samples = samples[:MAX_SENTENCES]
        print(f"Restricting to first {len(samples)} sentences (MAX_SENTENCES).")

    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    today = datetime.date.today().isoformat()
    result = {
        "meta": {
            "base_dataset": "How2Sign",
            "task": "SLU_question_generation",
            "llm_model": MODEL,
            "question_categories": ["temporal", "logical", "pragmatic", "details", "global"],
            "creation_date": today,
            "max_sentences": MAX_SENTENCES,
            "batch_size": 1,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "reasoning_effort": REASONING_EFFORT,
            "pricing": {
                "price_per_m_input": 0.05,
                "price_per_m_cached_input": 0.005,
                "price_per_m_output": 0.4,
            },
        },
        "sentences": [],
    }

    total_usage = empty_usage()
    total_cost = 0.0

    indexed_samples = list(enumerate(samples, start=1))
    print(f"Calling {MODEL} with concurrency={CONCURRENCY} ...")

    def process_one(args):
        idx, s = args
        client_local = OpenAI()

        error_str = None
        questions: List[Dict[str, Any]] = []
        usage_dict = empty_usage()
        extra: Dict[str, Any] = {"raw_output": "", "parse_error": None}

        try:
            questions, usage_dict, extra = call_model_on_sentence(
                client_local, prompt_text, s["text"]
            )
        except Exception as e:
            error_str = str(e)

        enriched = {
            "sentence_id": s["sentence_id"],
            "source_split": s["source_split"],
            "text": s["text"],
            "video_id": s.get("video_id"),
            "clip_id": s.get("clip_id"),
            "questions": questions,
            "generation": {
                "status": "ok" if error_str is None else "error",
                "error": error_str,
                "usage": usage_dict,
                "estimated_cost": estimate_cost(usage_dict),
                "parse_error": extra.get("parse_error"),
                "raw_output": extra.get("raw_output"),
            },
        }

        return idx, enriched, usage_dict

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        with tqdm(total=len(indexed_samples), desc="Sentences") as pbar:
            for idx, enriched, usage_dict in executor.map(process_one, indexed_samples):
                # accumula usage/costi
                total_usage = {
                    k: total_usage.get(k, 0) + usage_dict.get(k, 0)
                    for k in total_usage.keys()
                }
                total_cost += estimate_cost(usage_dict)["total_cost"]

                result["sentences"].append(enriched)
                pbar.update(1)

                if idx == 1:
                    c = estimate_cost(usage_dict)["total_cost"]
                    print(
                        f"[SENT {idx}/{len(samples)}] "
                        f"in={usage_dict.get('input_tokens', 0)} "
                        f"(cached {usage_dict.get('cached_tokens', 0)}), "
                        f"out={usage_dict.get('output_tokens', 0)}, "
                        f"tot={usage_dict.get('total_tokens', 0)} | cost ~ ${c:.6f}"
                    )

                if idx % COST_LOG_EVERY == 0:
                    print(
                        f"[COST] After {idx} sentences: total_cost ~ ${total_cost:.6f} "
                        f"(input_tokens={total_usage['input_tokens']}, "
                        f"cached_input_tokens={total_usage['cached_tokens']}, "
                        f"output_tokens={total_usage['output_tokens']})"
                    )

                if idx % SAVE_EVERY == 0:
                    tmp_path = OUTPUT_PATH + ".tmp"
                    with open(tmp_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
                    print(f"[BACKUP] Wrote temporary backup to {tmp_path} after {idx} sentences.")

    print("\n=== USAGE SUMMARY ===")
    print(f"Total input tokens:        {total_usage['input_tokens']}")
    print(f"Total cached input tokens: {total_usage['cached_tokens']}")
    print(f"Total output tokens:       {total_usage['output_tokens']}")
    print(f"Estimated total cost:      ${total_cost:.6f}")

    print(f"\nWriting final output to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Done. Wrote {len(result['sentences'])} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()