"""
evaluate.py
===========
Quantitative Evaluation Script for AI Medical Report Analyzer
Uses ROUGE and BLEU metrics to measure AI output quality against human-verified reference data.

Usage:
    python evaluate.py --api_key YOUR_GEMINI_API_KEY
    OR set GEMINI_API_KEY in .env and run: python evaluate.py
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# -- ROUGE Scoring ------------------------------------------------------------
from rouge_score import rouge_scorer

# -- BLEU Scoring -------------------------------------------------------------
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# -- Gemini API ---------------------------------------------------------------
from google import genai

# -- Load .env ----------------------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# -- Paths --------------------------------------------------------------------
BASE_DIR    = Path(__file__).parent
DATA_PATH   = BASE_DIR / "data" / "reference_data.json"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# -- Config -------------------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"

# Set to 3 for ~1 min run. Set to 8 for full run (~4 mins).
MAX_SAMPLES = 3

# -- ANSI Colors --------------------------------------------------------------
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    WHITE  = "\033[97m"
    PURPLE = "\033[95m"
    BG_BLUE   = "\033[44m"
    BG_DARK   = "\033[100m"

def colored(text, *codes):
    return "".join(codes) + str(text) + C.RESET

# -- Prompt -------------------------------------------------------------------
EVAL_PROMPT = """You are a Medical Report Analysis AI.
Read the following lab report text and respond with ONLY a JSON object.

Rules for your response:
- Name the exact medical condition (e.g. "iron-deficiency anemia", "hypothyroidism", "Type 2 Diabetes")
- Use specific medical terms (e.g. "LDL", "hemoglobin", "eGFR", "HbA1c")
- Be direct: say "consult a cardiologist" not "see a healthcare provider"
- State the risk level cause explicitly (e.g. "elevated LDL raises cardiovascular risk")
- Give specific action steps, not vague advice

Respond with this EXACT JSON structure:
{
  "summary": "3-4 sentences naming the exact condition, abnormal values, and health impact in plain English",
  "recommendation": "2-3 specific action steps using exact specialist names and test names",
  "risk_level": "One of: Low, Moderate, High, Critical"
}
NO markdown. NO code fences. ONLY the JSON object.

Report: """


# =============================================================================
# Core Functions
# =============================================================================

def get_ai_analysis(client: genai.Client, report_text: str) -> dict:
    """Send a report to Gemini with retry + rate-limit handling."""
    import time
    prompt = EVAL_PROMPT + report_text
    for attempt in range(3):
        try:
            response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            raw = response.text.strip()
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e != -1:
                return json.loads(raw[s:e+1])
        except Exception as ex:
            err = str(ex)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 65 * (attempt + 1)
                print(colored(f"  Rate limit hit. Waiting {wait}s...", C.YELLOW))
                time.sleep(wait)
            elif "403" in err or "PERMISSION_DENIED" in err:
                print(colored("  API key error — update your GEMINI_API_KEY in .env", C.RED, C.BOLD))
                break
            else:
                print(colored(f"  API error: {err[:100]}", C.RED))
                break
    return {"summary": "", "recommendation": "", "risk_level": "UNKNOWN"}


def compute_rouge(hypothesis: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    s = scorer.score(reference, hypothesis)
    return {
        "rouge1_f1":       round(s["rouge1"].fmeasure,  4),
        "rouge2_f1":       round(s["rouge2"].fmeasure,  4),
        "rougeL_f1":       round(s["rougeL"].fmeasure,  4),
        "rouge1_precision": round(s["rouge1"].precision, 4),
        "rouge1_recall":   round(s["rouge1"].recall,    4),
    }


def compute_bleu(hypothesis: str, reference: str) -> float:
    hyp = hypothesis.lower().split()
    ref = reference.lower().split()
    return round(sentence_bleu([ref], hyp, smoothing_function=SmoothingFunction().method1), 4)


def score_bar(score: float, width: int = 20) -> str:
    """Draw a colored ASCII progress bar for a score (0.0 - 1.0)."""
    filled = int(score * width)
    bar    = "█" * filled + "░" * (width - filled)
    if score >= 0.5: color = C.GREEN
    elif score >= 0.3: color = C.YELLOW
    else: color = C.RED
    return colored(bar, color)


def score_label(score: float) -> str:
    if score >= 0.7:   return colored("Excellent", C.GREEN,  C.BOLD)
    elif score >= 0.5: return colored("Good",      C.GREEN)
    elif score >= 0.3: return colored("Fair",      C.YELLOW)
    else:              return colored("Poor",       C.RED)


def risk_colored(risk: str) -> str:
    r = risk.lower()
    if r == "high" or r == "critical": return colored(risk.upper(), C.RED,    C.BOLD)
    elif r == "moderate":              return colored(risk.upper(), C.YELLOW, C.BOLD)
    elif r == "low":                   return colored(risk.upper(), C.GREEN,  C.BOLD)
    return colored(risk.upper(), C.DIM)


# =============================================================================
# Main Evaluation
# =============================================================================

def run_evaluation(api_key: str):
    import time

    W = 70  # total width

    # ── Banner ────────────────────────────────────────────────────────────────
    print()
    print(colored("=" * W, C.BLUE, C.BOLD))
    print(colored("  AI MEDICAL REPORT ANALYZER", C.WHITE, C.BOLD) +
          colored("  |  ROUGE & BLEU Evaluation", C.CYAN))
    print(colored("=" * W, C.BLUE, C.BOLD))
    print(colored(f"  Model   : ", C.DIM) + colored(GEMINI_MODEL, C.WHITE))
    print(colored(f"  Samples : ", C.DIM) + colored(f"{MAX_SAMPLES} of 8 reference reports", C.WHITE))
    print(colored(f"  Date    : ", C.DIM) + colored(datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S"), C.WHITE))
    print(colored("=" * W, C.BLUE, C.BOLD))
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reference_data = json.load(f)[:MAX_SAMPLES]

    client = genai.Client(api_key=api_key)

    all_results   = []
    rouge1_scores, rouge2_scores, rougeL_scores, bleu_scores = [], [], [], []

    # ── Per-sample loop ───────────────────────────────────────────────────────
    for i, sample in enumerate(reference_data, 1):
        sid         = sample["id"]
        rtype       = sample["report_type"]
        input_text  = sample["input_text"]
        ref_summary = sample["reference_summary"]
        ref_recom   = sample["reference_recommendation"]

        # Sample header
        print(colored(f"  Sample {i}/{len(reference_data)}", C.CYAN, C.BOLD) +
              colored(f"  ▸  {rtype}", C.WHITE))
        print(colored("  " + "─" * (W - 2), C.DIM))

        print(colored("  Querying Gemini AI...", C.DIM), end="", flush=True)
        ai_output = get_ai_analysis(client, input_text)
        print(colored("  done", C.GREEN))

        # Delay between calls (free-tier rate limit)
        if i < len(reference_data):
            time.sleep(13)

        ai_summary   = ai_output.get("summary",        "")
        ai_recom     = ai_output.get("recommendation", "")
        ai_risk      = ai_output.get("risk_level",     "UNKNOWN")
        ai_text      = f"{ai_summary} {ai_recom}"
        ref_text     = f"{ref_summary} {ref_recom}"

        rouge_s      = compute_rouge(ai_text, ref_text)
        bleu         = compute_bleu(ai_text, ref_text)
        ref_risk     = sample.get("reference_risk_level", "")
        risk_match   = ai_risk.lower() == ref_risk.lower()

        rouge1_scores.append(rouge_s["rouge1_f1"])
        rouge2_scores.append(rouge_s["rouge2_f1"])
        rougeL_scores.append(rouge_s["rougeL_f1"])
        bleu_scores.append(bleu)

        # Score display
        r1, r2, rL, bl = rouge_s["rouge1_f1"], rouge_s["rouge2_f1"], rouge_s["rougeL_f1"], bleu
        print()
        print(colored("  ROUGE-1", C.PURPLE, C.BOLD) + f"  {score_bar(r1)}  {r1:.4f}  {score_label(r1)}")
        print(colored("  ROUGE-2", C.PURPLE, C.BOLD) + f"  {score_bar(r2)}  {r2:.4f}  {score_label(r2)}")
        print(colored("  ROUGE-L", C.PURPLE, C.BOLD) + f"  {score_bar(rL)}  {rL:.4f}  {score_label(rL)}")
        print(colored("  BLEU   ", C.CYAN,   C.BOLD) + f"  {score_bar(bl)}  {bl:.4f}  {score_label(bl)}")
        print()

        match_str = colored("✔  MATCH",    C.GREEN, C.BOLD) if risk_match else colored("✘  MISMATCH", C.RED)
        print(colored("  Risk Level →", C.DIM) +
              f"  AI: {risk_colored(ai_risk):<20}  Ref: {risk_colored(ref_risk):<20}  {match_str}")
        print()

        # AI summary preview
        preview = (ai_summary[:120] + "...") if len(ai_summary) > 120 else ai_summary
        print(colored("  AI Summary:", C.DIM))
        print(colored(f"  {preview}", C.DIM))
        print()
        print(colored("  " + "─" * (W - 2), C.DIM))
        print()

        all_results.append({
            "id": sid, "report_type": rtype,
            "ai_summary": ai_summary, "ai_recommendation": ai_recom,
            "ai_risk_level": ai_risk, "ref_risk_level": ref_risk,
            "risk_level_match": risk_match,
            "rouge1_f1": r1, "rouge2_f1": r2, "rougeL_f1": rL,
            "rouge1_precision": rouge_s["rouge1_precision"],
            "rouge1_recall":    rouge_s["rouge1_recall"],
            "bleu_score": bleu,
        })

    # ── Averages ──────────────────────────────────────────────────────────────
    avg_r1 = round(sum(rouge1_scores) / len(rouge1_scores), 4)
    avg_r2 = round(sum(rouge2_scores) / len(rouge2_scores), 4)
    avg_rL = round(sum(rougeL_scores) / len(rougeL_scores), 4)
    avg_bl = round(sum(bleu_scores)   / len(bleu_scores),   4)
    risk_matches = sum(1 for r in all_results if r["risk_level_match"])

    # ── Final Summary ─────────────────────────────────────────────────────────
    print(colored("═" * W, C.BLUE, C.BOLD))
    print(colored("  EVALUATION SUMMARY", C.WHITE, C.BOLD))
    print(colored("═" * W, C.BLUE, C.BOLD))
    print()

    metrics = [
        ("ROUGE-1", "Word-level content overlap",    avg_r1),
        ("ROUGE-2", "Phrase-level content overlap",  avg_r2),
        ("ROUGE-L", "Sequence & fluency match",      avg_rL),
        ("BLEU   ", "Word precision score",          avg_bl),
    ]
    for name, desc, val in metrics:
        print(f"  {colored(name, C.PURPLE, C.BOLD)}  {score_bar(val, 25)}  "
              f"{colored(f'{val:.4f}', C.WHITE, C.BOLD)}  {score_label(val)}")
        print(colored(f"           {desc}", C.DIM))
        print()

    print(colored("  " + "─" * (W - 2), C.DIM))
    risk_pct = int(risk_matches / len(all_results) * 100)
    print(f"  {colored('Risk Level Accuracy', C.CYAN, C.BOLD)}   "
          f"{colored(f'{risk_matches}/{len(all_results)} samples', C.WHITE, C.BOLD)} "
          f"matched correctly  ({risk_pct}%)")
    print()
    print(colored("═" * W, C.BLUE, C.BOLD))

    # ── Legend ────────────────────────────────────────────────────────────────
    print()
    print(colored("  WHAT THESE SCORES MEAN:", C.WHITE, C.BOLD))
    print(colored("  ROUGE", C.PURPLE) + colored(" — How much content from the human reference the AI covered", C.DIM))
    print(colored("  BLEU ", C.CYAN)   + colored(" — How precisely the AI's words matched the reference", C.DIM))
    print(colored("  Note ", C.YELLOW) + colored(" — Lower BLEU is normal; AI paraphrases rather than copying", C.DIM))
    print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "evaluation_date": datetime.datetime.now().isoformat(),
        "model_used":      GEMINI_MODEL,
        "total_samples":   len(all_results),
        "average_scores":  {"rouge1_f1": avg_r1, "rouge2_f1": avg_r2, "rougeL_f1": avg_rL, "bleu_score": avg_bl},
        "risk_level_accuracy": f"{risk_matches}/{len(all_results)}",
        "per_sample_results":  all_results,
    }
    rpath = RESULTS_DIR / f"evaluation_{ts}.json"
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(colored(f"  Results saved → {rpath.name}", C.DIM))
    print()


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()

    key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        print(colored("[ERROR] GEMINI_API_KEY not found.", C.RED, C.BOLD))
        sys.exit(1)

    run_evaluation(key)
