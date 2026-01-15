import json
import sys
import random
from collections import defaultdict
from typing import Dict, Any, List

#                              comand to run this script:                                               #
#                                                                                                       #
# python scripts/sample_slu_questions.py results/slu_questions_promptWexamples_gpt-5.1_none_1280.json   #
#                                                                                                       #
#                                                                                                       #


CATEGORIES = ["temporal", "logical", "pragmatic", "details", "global"]
SAMPLES_PER_CATEGORY = 10


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_slu_samples_txt.py <slu_questions.json> [output.txt]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results/slu_sample_questions.txt"

    data = load_json(input_path)
    sentences: List[Dict[str, Any]] = data.get("sentences", [])

    # category -> list of (sentence_id, sentence_text, question_dict)
    bucket = defaultdict(list)

    for s in sentences:
        sentence_id = s.get("sentence_id", "")
        sentence_text = s.get("text", "")
        questions = s.get("questions", [])

        for q in questions:
            cat = q.get("category")
            if cat in CATEGORIES:
                bucket[cat].append((sentence_id, sentence_text, q))

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("SLU BENCHMARK — QUALITATIVE SAMPLES\n")
        out.write("=" * 50 + "\n\n")

        for cat in CATEGORIES:
            examples = bucket.get(cat, [])
            if not examples:
                out.write(f"\n--- CATEGORY: {cat.upper()} ---\n")
                out.write("No examples found.\n\n")
                continue

            k = min(SAMPLES_PER_CATEGORY, len(examples))
            sampled = random.sample(examples, k)

            out.write(f"\n--- CATEGORY: {cat.upper()} ({k}/{len(examples)}) ---\n\n")

            for i, (sentence_id, sentence_text, q) in enumerate(sampled, start=1):
                question = q.get("question", "")
                correct = q.get("correct_answer", "")
                distractors = q.get("distractors", [])

                out.write(f"[{i}] Sentence ID: {sentence_id}\n")
                out.write("Sentence:\n")
                out.write(f"  {sentence_text}\n\n")

                out.write("Question:\n")
                out.write(f"  {question}\n\n")

                out.write("Options:\n")
                out.write(f"  [CORRECT]   {correct}\n")
                for d in distractors:
                    out.write(f"  [DISTRACTOR] {d}\n")

                # Quick checks (human-readable)
                checks = []
                checks.append("4 distractors ✓" if len(distractors) == 4 else "WRONG #distractors ✗")
                checks.append("correct not duplicated ✓" if correct not in distractors else "correct duplicated ✗")
                checks.append(f"category = {cat}")

                out.write("\nChecks:\n")
                for c in checks:
                    out.write(f"  - {c}\n")

                out.write("\n" + "-" * 40 + "\n\n")

    print(f"Saved qualitative samples to: {output_path}")


if __name__ == "__main__":
    main()