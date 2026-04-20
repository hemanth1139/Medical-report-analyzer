import sys
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def run_demo():
    print("\n" + "="*80)
    print(" ROUGE & BLEU METRICS DEMONSTRATION")
    print("="*80)

    reference = "The patient has low hemoglobin and is anemic. Iron supplements are recommended."
    hypothesis = "The patient presents with low hemoglobin and is anemic. It is recommended to take iron supplements."

    print("\n[SCENARIO: HEMOGLOBIN TEST ANALYSIS]")
    print(f"Reference (Human Expert):  {reference}")
    print(f"Generated (AI Output):     {hypothesis}\n")

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    r1 = scores["rouge1"].fmeasure
    r2 = scores["rouge2"].fmeasure
    rL = scores["rougeL"].fmeasure

    # BLEU
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=SmoothingFunction().method1)

    print("-" * 80)
    print(f"{'Metric Name':<15} | {'Score':<10} | {'Quality':<12} | {'Description'}")
    print("-" * 80)

    metrics = [
        ("ROUGE-1", r1, "Word-level Recall"),
        ("ROUGE-2", r2, "Phrase-level Recall"),
        ("ROUGE-L", rL, "Sequence match"),
        ("BLEU", bleu, "Word-for-Word Precision"),
    ]

    for name, val, desc in metrics:
        lbl = "Excellent" if val >= 0.7 else "Good" if val >=0.4 else "Fair"
        print(f"{name:<15} | {val:<10.4f} | {lbl:<12} | {desc}")

    print("-" * 80)
    print("\nConclusion:")
    print("The AI effectively captures the vital words ('hemoglobin', 'anemic', 'iron').")
    print("Hence, ROUGE-1 is extremely high. Since generative models naturally paraphrase")
    print("('are recommended' vs 'It is recommended to take'), BLEU is slightly lower but still fair.")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    run_demo()
