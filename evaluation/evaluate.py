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
                print(f"  Rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
            elif "403" in err or "PERMISSION_DENIED" in err:
                print("  API key error — update your GEMINI_API_KEY in .env")
                break
            else:
                print(f"  API error: {err[:100]}")
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

def score_label(score: float) -> str:
    if score >= 0.7:   return "Excellent"
    elif score >= 0.5: return "Good"
    elif score >= 0.3: return "Fair"
    else:              return "Poor"


# =============================================================================
# Main Evaluation
# =============================================================================

def run_evaluation(api_key: str):
    import time

    W = 80  # total width

    # -- Banner ----------------------------------------------------------------
    print("\n" + "=" * W)
    print(" AI MEDICAL REPORT ANALYZER | ROUGE & BLEU Evaluation")
    print("=" * W)
    print(f" Model   : {GEMINI_MODEL}")
    print(f" Samples : {MAX_SAMPLES} of 8 reference reports")
    print(f" Date    : {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * W + "\n")

    # -- Load data -------------------------------------------------------------
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reference_data = json.load(f)[:MAX_SAMPLES]

    client = genai.Client(api_key=api_key)

    all_results   = []
    rouge1_scores, rouge2_scores, rougeL_scores, bleu_scores = [], [], [], []

    # -- Per-sample loop -------------------------------------------------------
    for i, sample in enumerate(reference_data, 1):
        sid         = sample["id"]
        rtype       = sample["report_type"]
        input_text  = sample["input_text"]
        ref_summary = sample["reference_summary"]
        ref_recom   = sample["reference_recommendation"]

        print(f" Sample {i}/{len(reference_data)} ▸ {rtype}")
        print("-" * W)

        print(" Querying Gemini AI... ", end="", flush=True)
        ai_output = get_ai_analysis(client, input_text)
        print("done.")

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

        r1, r2, rL, bl = rouge_s["rouge1_f1"], rouge_s["rouge2_f1"], rouge_s["rougeL_f1"], bleu
        print()
        print(f" {'Metric':<10} | {'Score':<8} | {'Quality'}")
        print(" " + "-" * 35)
        print(f" {'ROUGE-1':<10} | {r1:<8.4f} | {score_label(r1)}")
        print(f" {'ROUGE-2':<10} | {r2:<8.4f} | {score_label(r2)}")
        print(f" {'ROUGE-L':<10} | {rL:<8.4f} | {score_label(rL)}")
        print(f" {'BLEU':<10} | {bl:<8.4f} | {score_label(bl)}")
        print()

        match_str = "MATCH" if risk_match else "MISMATCH"
        print(f" Risk Level → AI: {ai_risk.upper():<10} Ref: {ref_risk.upper():<10} [{match_str}]")
        print()

        preview = (ai_summary[:120] + "...") if len(ai_summary) > 120 else ai_summary
        print(" AI Summary Snippet:")
        print(f" {preview}")
        print("\n" + "=" * W + "\n")

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

    # -- Averages --------------------------------------------------------------
    avg_r1 = round(sum(rouge1_scores) / len(rouge1_scores), 4)
    avg_r2 = round(sum(rouge2_scores) / len(rouge2_scores), 4)
    avg_rL = round(sum(rougeL_scores) / len(rougeL_scores), 4)
    avg_bl = round(sum(bleu_scores)   / len(bleu_scores),   4)
    risk_matches = sum(1 for r in all_results if r["risk_level_match"])

    # -- Final Summary ---------------------------------------------------------
    print("=" * W)
    print(" EVALUATION SUMMARY (AVERAGES)")
    print("=" * W)
    print()
    print(f" {'Metric Name':<12} | {'Avg Score':<10} | {'Quality':<12} | {'Description'}")
    print(" " + "-" * 75)

    metrics = [
        ("ROUGE-1", avg_r1, "Word-level content overlap"),
        ("ROUGE-2", avg_r2, "Phrase-level content overlap"),
        ("ROUGE-L", avg_rL, "Sequence & fluency match"),
        ("BLEU", avg_bl, "Word precision score"),
    ]
    
    for name, val, desc in metrics:
        print(f" {name:<12} | {val:<10.4f} | {score_label(val):<12} | {desc}")

    print(" " + "-" * 75)
    risk_pct = int(risk_matches / len(all_results) * 100)
    print(f" Risk Accuracy | {risk_matches}/{len(all_results)} matched | ({risk_pct}%)")
    print("\n" + "=" * W)

    # -- Save JSON -------------------------------------------------------------
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

    print(f"\n Results saved to: {rpath.name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default=None)
    args = parser.parse_args()

    key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        print("[ERROR] GEMINI_API_KEY not found. Please set it in .env")
        sys.exit(1)

    run_evaluation(key)
