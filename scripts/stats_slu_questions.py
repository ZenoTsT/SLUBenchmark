import json
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List


#                              comand to run this script:                                       #
#                                                                                               #
# python stats_slu_questions.py results/slu_questions_promptWexamples_gpt-5.1_none_1280.json    #
#                                                                                               #
#                                                                                               #


ALLOWED_CATEGORIES = {"temporal", "logical", "pragmatic", "details", "global"}


def safe_len(x) -> int:
    return len(x) if isinstance(x, list) else 0


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_question_valid(q: Dict[str, Any]) -> bool:
    """
    Validazione minima per individuare output 'formalmente' ok.
    (Non giudica qualità semantica, solo struttura).
    """
    if not isinstance(q, dict):
        return False
    cat = q.get("category")
    if cat not in ALLOWED_CATEGORIES:
        return False
    question = q.get("question")
    correct = q.get("correct_answer")
    distractors = q.get("distractors")

    if not isinstance(question, str) or not question.strip():
        return False
    if not isinstance(correct, str) or not correct.strip():
        return False
    if not isinstance(distractors, list) or len(distractors) != 4:
        return False
    if any((not isinstance(d, str) or not d.strip()) for d in distractors):
        return False

    # Evita duplicati triviali
    options = [correct] + distractors
    norm = [o.strip().lower() for o in options]
    if len(set(norm)) != len(norm):
        return False

    # Correct non deve stare nei distractors (normalizzato)
    if correct.strip().lower() in {d.strip().lower() for d in distractors}:
        return False

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_slu_questions.py <slu_questions.json>")
        sys.exit(1)

    path = sys.argv[1]
    data = load_json(path)

    sentences: List[Dict[str, Any]] = data.get("sentences", [])
    n = len(sentences)

    # --- Core counters ---
    gen_status = Counter()  # ok / error / missing
    parse_error_count = 0

    num_questions_dist = Counter()  # 0,1,2,3...
    category_counts = Counter()      # total questions per category
    sentences_with_category = Counter()  # sentences that contain >=1 question of category

    # Quality-ish diagnostics (structure level)
    valid_questions = 0
    invalid_questions = 0
    invalid_reasons = Counter()  # coarse buckets

    # Token / cost stats
    total_usage = Counter()
    total_cost = 0.0
    per_sentence_costs = []

    # For extra insight
    empty_but_ok = 0        # generation ok but 0 questions
    error_but_has_q = 0     # generation error but has questions (rare)
    missing_generation = 0  # no generation field at all

    for s in sentences:
        questions = s.get("questions", [])
        qn = safe_len(questions)
        num_questions_dist[qn] += 1

        # generation status
        gen = s.get("generation")
        if not isinstance(gen, dict):
            gen_status["missing"] += 1
            missing_generation += 1
        else:
            st = gen.get("status", "missing")
            gen_status[st] += 1

            if gen.get("parse_error") is not None:
                parse_error_count += 1

            usage = gen.get("usage", {})
            if isinstance(usage, dict):
                for k in ["input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens"]:
                    total_usage[k] += int(usage.get(k, 0) or 0)

            est = gen.get("estimated_cost", {})
            if isinstance(est, dict):
                c = float(est.get("total_cost", 0.0) or 0.0)
                total_cost += c
                per_sentence_costs.append(c)

            if st == "ok" and qn == 0:
                empty_but_ok += 1
            if st != "ok" and qn > 0:
                error_but_has_q += 1

        # category counts
        seen_cats = set()
        if isinstance(questions, list):
            for q in questions:
                if isinstance(q, dict):
                    cat = q.get("category")
                    if cat in ALLOWED_CATEGORIES:
                        category_counts[cat] += 1
                        seen_cats.add(cat)
                    # validate structure
                    if is_question_valid(q):
                        valid_questions += 1
                    else:
                        invalid_questions += 1
                        # coarse diagnosis
                        if not isinstance(cat, str) or cat not in ALLOWED_CATEGORIES:
                            invalid_reasons["bad_category"] += 1
                        elif not isinstance(q.get("distractors"), list) or len(q.get("distractors", [])) != 4:
                            invalid_reasons["bad_distractors_len"] += 1
                        else:
                            invalid_reasons["other_format_issue"] += 1

        for c in seen_cats:
            sentences_with_category[c] += 1

    # --- Print report ---
    print("\n====================")
    print("SLU Generation Report")
    print("====================\n")

    print(f"File: {path}")
    print(f"Total sentences: {n}")

    # Generation status
    print("\n--- Generation status (sentence-level) ---")
    for k in ["ok", "error", "missing"]:
        if k in gen_status:
            print(f"{k:>7}: {gen_status[k]}")
    if parse_error_count:
        print(f"parse_error sentences: {parse_error_count}")
    if missing_generation:
        print(f"sentences missing 'generation' field: {missing_generation}")
    if empty_but_ok:
        print(f"sentences with status=ok but 0 questions: {empty_but_ok}")
    if error_but_has_q:
        print(f"sentences with status!=ok but >0 questions: {error_but_has_q}")

    # Questions per sentence distribution
    print("\n--- Questions per sentence ---")
    for k in sorted(num_questions_dist.keys()):
        print(f"{k} question(s): {num_questions_dist[k]}")

    # Questions per category (total)
    total_q = sum(category_counts.values())
    print("\n--- Questions by category (total questions) ---")
    print(f"Total questions: {total_q}")
    for cat in ["temporal", "logical", "pragmatic", "details", "global"]:
        cnt = category_counts.get(cat, 0)
        pct = (cnt / total_q * 100.0) if total_q else 0.0
        print(f"{cat:>9}: {cnt:5d}  ({pct:5.1f}%)")

    # Sentences coverage by category
    print("\n--- Sentences containing each category (coverage) ---")
    for cat in ["temporal", "logical", "pragmatic", "details", "global"]:
        cnt = sentences_with_category.get(cat, 0)
        pct = (cnt / n * 100.0) if n else 0.0
        print(f"{cat:>9}: {cnt:5d} / {n}  ({pct:5.1f}%)")

    # Format validity (lightweight)
    print("\n--- Structural validity (question-level) ---")
    print(f"Valid questions:   {valid_questions}")
    print(f"Invalid questions: {invalid_questions}")
    if invalid_questions:
        print("Invalid reason breakdown:")
        for r, c in invalid_reasons.most_common():
            print(f"  - {r}: {c}")

    # Token/cost summary
    print("\n--- Usage & cost summary (from generation logs) ---")
    if per_sentence_costs:
        avg_cost = total_cost / len(per_sentence_costs)
        p95 = sorted(per_sentence_costs)[int(0.95 * (len(per_sentence_costs) - 1))]
        mx = max(per_sentence_costs)
        print(f"Estimated total cost: ${total_cost:.6f}")
        print(f"Avg cost / sentence:  ${avg_cost:.6f}")
        print(f"P95 cost / sentence:  ${p95:.6f}")
        print(f"Max cost / sentence:  ${mx:.6f}")
    else:
        print("No per-sentence cost info found (generation.estimated_cost missing).")

    for k in ["input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "total_tokens"]:
        if k in total_usage:
            print(f"{k:>16}: {total_usage[k]}")

    # Extra useful stats ideas:
    # - fraction of sentences where model produced >2 questions (if you want to enforce max 2)
    over2 = sum(v for qn, v in num_questions_dist.items() if qn > 2)
    if over2:
        print(f"\nNote: sentences with >2 questions: {over2} (you may want to enforce a max)")

    print("\nDone.\n")


if __name__ == "__main__":
    main()