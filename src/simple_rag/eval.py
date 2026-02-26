"""
RAG Evaluation Script — LLM-as-a-Judge

Tests the RAG system against a set of known question/answer pairs from the
"Attention Is All You Need" paper and uses Gemini to judge whether the RAG
answers are correct.

Usage:
    pixi run python -m simple_rag.eval [path/to/arxiv.pdf]

If no path is given, defaults to src/simple_rag/arxiv.pdf
"""

import json
import os
import sys
import time

from simple_rag.rag import (
    ingest_pdf_path,
    query_document,
    client,
    CHAT_MODEL,
    _retry_with_backoff,
)

# ---------------------------------------------------------------------------
# Test cases: (question, expected_answer, difficulty)
# ---------------------------------------------------------------------------

TEST_CASES = [
    # --- Easy: facts stated directly ---
    {
        "question": "Who are the authors of this paper?",
        "expected": (
            "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, "
            "Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin."
        ),
        "difficulty": "easy",
    },
    {
        "question": "How many layers does the encoder have?",
        "expected": ("The encoder is composed of a stack of N = 6 identical layers."),
        "difficulty": "easy",
    },
    {
        "question": "What is the model dimensionality (d_model) used in the base model?",
        "expected": "512.",
        "difficulty": "easy",
    },
    {
        "question": "What BLEU score did the big Transformer achieve on English-to-German translation?",
        "expected": "28.4 BLEU.",
        "difficulty": "easy",
    },
    {
        "question": "What is the name of the model architecture introduced in this paper?",
        "expected": "The Transformer.",
        "difficulty": "easy",
    },
    # --- Medium: requires synthesizing information ---
    {
        "question": "How does multi-head attention work?",
        "expected": (
            "Multi-head attention projects queries, keys, and values h times "
            "(h=8 in the base model) with different learned linear projections "
            "to d_k=64 dimensions each. Attention is applied in parallel on "
            "each projected version, and the outputs are concatenated and "
            "projected again to produce the final result."
        ),
        "difficulty": "medium",
    },
    {
        "question": "Why did the authors use positional encodings instead of learned positional embeddings?",
        "expected": (
            "Because the Transformer has no recurrence or convolution, it needs "
            "positional encodings to give the model information about the order "
            "of tokens in the sequence. They used sine and cosine functions of "
            "different frequencies. They found that learned positional embeddings "
            "produced nearly identical results."
        ),
        "difficulty": "medium",
    },
    {
        "question": "What are the advantages of self-attention over recurrent layers?",
        "expected": (
            "Self-attention connects all positions with O(1) sequential "
            "operations (vs O(n) for recurrence), enabling much more "
            "parallelization. It also provides shorter maximum path lengths "
            "for long-range dependencies and lower total computational cost "
            "per layer when sequence length is less than the representation "
            "dimensionality."
        ),
        "difficulty": "medium",
    },
    # --- Hard: requires reasoning across sections ---
    {
        "question": "How does the training configuration differ between the base and big Transformer models?",
        "expected": (
            "The base model was trained for 100K steps (~12 hours on 8 P100 GPUs) "
            "with d_model=512, d_ff=2048, h=8, and dropout P_drop=0.1. "
            "The big model was trained for 300K steps (~3.5 days on 8 P100 GPUs) "
            "with d_model=1024, d_ff=4096, h=16, and higher dropout P_drop=0.3."
        ),
        "difficulty": "hard",
    },
    {
        "question": "What regularization techniques were used during training?",
        "expected": (
            "Two regularization techniques were used: (1) Residual Dropout — "
            "dropout applied to the output of each sub-layer before it is added "
            "to the sub-layer input and normalized, as well as to the sums of "
            "the embeddings and positional encodings; and (2) Label Smoothing "
            "with epsilon_ls = 0.1, which hurts perplexity but improves accuracy "
            "and BLEU score."
        ),
        "difficulty": "hard",
    },
]


# ---------------------------------------------------------------------------
# Judge: use Gemini to evaluate whether the RAG answer is correct
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are an impartial judge evaluating whether a RAG system's answer is correct.

You will be given:
- A QUESTION
- An EXPECTED ANSWER (the ground truth)
- The RAG SYSTEM'S ANSWER

Your job is to decide whether the RAG answer is:
- CORRECT: The answer captures the key facts from the expected answer. It does
  not need to be word-for-word identical — paraphrasing, different ordering, or
  including additional correct details are all fine. Minor omissions of non-critical
  details are acceptable.
- PARTIALLY_CORRECT: The answer contains some correct information but is missing
  important key facts, or contains a mix of correct and incorrect statements.
- INCORRECT: The answer is wrong, says "I don't know", or fails to address the
  question in a meaningful way.

Respond with ONLY a JSON object in this exact format (no markdown fences):
{{"verdict": "CORRECT" | "PARTIALLY_CORRECT" | "INCORRECT", "explanation": "brief reason"}}

QUESTION:
{question}

EXPECTED ANSWER:
{expected}

RAG SYSTEM'S ANSWER:
{actual}
"""


def judge_answer(question: str, expected: str, actual: str) -> dict:
    """Ask Gemini to judge whether *actual* correctly answers *question*
    given the *expected* ground-truth answer. Returns a dict with
    'verdict' and 'explanation'."""

    prompt = JUDGE_PROMPT.format(question=question, expected=expected, actual=actual)

    response = _retry_with_backoff(
        client.models.generate_content, model=CHAT_MODEL, contents=prompt
    )

    text = response.text.strip()

    # Strip markdown code fences if the model wraps its response
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove first and last lines (the fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"verdict": "PARSE_ERROR", "explanation": text}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# Verdict styling
_VERDICT_STYLE = {
    "CORRECT": "\033[92m✔ CORRECT\033[0m",
    "PARTIALLY_CORRECT": "\033[93m◐ PARTIALLY CORRECT\033[0m",
    "INCORRECT": "\033[91m✘ INCORRECT\033[0m",
    "PARSE_ERROR": "\033[95m⚠ PARSE ERROR\033[0m",
}


def run_eval(pdf_path: str) -> dict:
    """Ingest the PDF, run all test cases, judge each answer, and return
    a summary report."""

    print("=" * 70)
    print("  RAG Evaluation — LLM-as-a-Judge")
    print("=" * 70)
    print()

    # 1. Ingest
    print(f"📄 Ingesting: {pdf_path}")
    doc_id = ingest_pdf_path(pdf_path)
    print(f"   Document ID: {doc_id}")
    print()

    # 2. Run each test case
    results = []
    for i, tc in enumerate(TEST_CASES, 1):
        question = tc["question"]
        expected = tc["expected"]
        difficulty = tc["difficulty"]

        print(f"─── Question {i}/{len(TEST_CASES)} [{difficulty.upper()}] ───")
        print(f"  Q: {question}")

        # Query the RAG
        try:
            rag_result = query_document(doc_id, question)
            actual = rag_result["answer"]
        except Exception as e:
            actual = f"[ERROR] {e}"

        print(f"  A: {actual[:200]}{'...' if len(actual) > 200 else ''}")

        # Judge
        verdict = judge_answer(question, expected, actual)
        verdict_label = verdict.get("verdict", "PARSE_ERROR")
        styled = _VERDICT_STYLE.get(verdict_label, verdict_label)

        print(f"  → {styled}")
        print(f"    {verdict.get('explanation', '')}")
        print()

        results.append(
            {
                "question": question,
                "expected": expected,
                "actual": actual,
                "difficulty": difficulty,
                "verdict": verdict_label,
                "explanation": verdict.get("explanation", ""),
            }
        )

        # Small delay to be polite to the free tier
        time.sleep(2)

    # 3. Summary
    total = len(results)
    correct = sum(1 for r in results if r["verdict"] == "CORRECT")
    partial = sum(1 for r in results if r["verdict"] == "PARTIALLY_CORRECT")
    incorrect = sum(1 for r in results if r["verdict"] == "INCORRECT")
    errors = sum(1 for r in results if r["verdict"] == "PARSE_ERROR")

    # Weighted score: correct=1.0, partial=0.5, incorrect/error=0.0
    score = (correct + partial * 0.5) / total * 100 if total else 0

    # Per-difficulty breakdown
    difficulties = ["easy", "medium", "hard"]
    breakdown = {}
    for diff in difficulties:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            diff_correct = sum(1 for r in diff_results if r["verdict"] == "CORRECT")
            diff_partial = sum(
                1 for r in diff_results if r["verdict"] == "PARTIALLY_CORRECT"
            )
            diff_total = len(diff_results)
            diff_score = (diff_correct + diff_partial * 0.5) / diff_total * 100
            breakdown[diff] = {
                "total": diff_total,
                "correct": diff_correct,
                "partial": diff_partial,
                "score": round(diff_score, 1),
            }

    summary = {
        "total": total,
        "correct": correct,
        "partially_correct": partial,
        "incorrect": incorrect,
        "parse_errors": errors,
        "score": round(score, 1),
        "breakdown": breakdown,
    }

    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"  Overall Score:  {summary['score']}%")
    print(f"  Total:          {total}")
    print(f"  ✔ Correct:      {correct}")
    print(f"  ◐ Partial:      {partial}")
    print(f"  ✘ Incorrect:    {incorrect}")
    if errors:
        print(f"  ⚠ Parse Errors: {errors}")
    print()

    for diff in difficulties:
        if diff in breakdown:
            b = breakdown[diff]
            print(
                f"  {diff.upper():>8s}:  {b['score']:5.1f}%  "
                f"({b['correct']}/{b['total']} correct, "
                f"{b['partial']} partial)"
            )
    print()
    print("=" * 70)

    return {"results": results, "summary": summary}


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main():
    default_path = os.path.join(os.path.dirname(__file__), "arxiv.pdf")
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else default_path

    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        sys.exit(1)

    report = run_eval(pdf_path)

    # Save full report to JSON for later analysis
    report_path = os.path.join(os.path.dirname(__file__), "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Full report saved to: {report_path}")
    print()


if __name__ == "__main__":
    main()
