# Evaluation Suite — AI Medical Report Analyzer

This folder contains the **quantitative evaluation pipeline** for the AI Medical Report Analyzer.
It tests how well our Gemini-powered AI performs compared to **human-verified reference outputs**,
using industry-standard NLP metrics: **ROUGE** and **BLEU**.

---

## Folder Structure

```
evaluation/
├── README.md                    <- You are here
├── requirements.txt             <- Python dependencies
├── evaluate.py                  <- Main evaluation script (ROUGE + BLEU)
├── data/
│   └── reference_data.json      <- Human-verified medical report samples
└── results/
    └── evaluation_YYYYMMDD.json <- Auto-generated evaluation scores
```

---

## Setup

### Step 1: Install dependencies
```bash
cd evaluation
pip install -r requirements.txt
```

### Step 2: Set your API key
Make sure your `GEMINI_API_KEY` is set in the `.env` file in the main project:
```
GEMINI_API_KEY=your_key_here
```

### Step 3: Run the evaluation
```bash
python evaluate.py
```

---

## 1. Reference Data (`data/reference_data.json`)

### What it is
A collection of **8 human-verified medical report samples**.
Each sample includes:
- `input_text` — The raw lab report text (input to the AI)
- `reference_summary` — Ideal human-written summary
- `reference_recommendation` — Ideal human-written recommendation
- `reference_risk_level` — Human-assigned risk level (Low / Moderate / High)

### Why we need it
ROUGE and BLEU metrics work by **comparing AI output against a known correct answer**.
Without a reference, we cannot objectively measure quality. This dataset serves as the
**"answer key"** for the evaluation.

### Reports Covered
| # | Report Type              | Key Condition              |
|---|--------------------------|----------------------------|
| 001 | CBC                    | Iron-deficiency anemia     |
| 002 | Lipid Panel            | High cardiovascular risk   |
| 003 | Thyroid Panel          | Hashimoto's hypothyroidism |
| 004 | Diabetes Panel         | Uncontrolled diabetes      |
| 005 | Liver Function Test    | Active liver damage        |
| 006 | Kidney Function Test   | Stage 3B CKD               |
| 007 | Vitamin Panel          | Multiple deficiencies      |
| 008 | Comprehensive Metabolic | Pre-diabetic (borderline) |

---

## 2. Evaluation Script (`evaluate.py`)

### What it does
1. Loads all 8 reference samples from `data/reference_data.json`
2. Sends each report to **Gemini AI** for analysis
3. Compares the AI output vs the human reference using **ROUGE + BLEU**
4. Prints a detailed score table to the console
5. Saves the full results to `results/evaluation_TIMESTAMP.json`

### Understanding the Metrics

#### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
Measures **content coverage** — did the AI mention all the important things?

| Metric   | What it measures                          |
|----------|-------------------------------------------|
| ROUGE-1  | Individual word matches (unigrams)        |
| ROUGE-2  | Two-word phrase matches (bigrams)         |
| ROUGE-L  | Longest common sequence (fluency + order) |

> Score range: 0.0 to 1.0 — higher is better.

#### BLEU (Bilingual Evaluation Understudy)
Measures **word precision** — how accurate is the AI's language?

> Score range: 0.0 to 1.0 — higher is better.
> BLEU focuses on exact word matches, so it is naturally lower than ROUGE.

#### Score Interpretation
| Score Range | Quality     |
|-------------|-------------|
| 0.70 – 1.00 | Excellent   |
| 0.50 – 0.69 | Good        |
| 0.30 – 0.49 | Fair        |
| 0.00 – 0.29 | Poor        |

### Sample Output
```
====================================================================
   AI Medical Report Analyzer -- ROUGE & BLEU Evaluation
====================================================================
   Model  : gemini-2.5-flash
   Samples: 8
====================================================================

[1/8] Evaluating: CBC (ID: 001)
      ROUGE-1: 0.5821  ROUGE-2: 0.3142  ROUGE-L: 0.4923  BLEU: 0.2341
      Risk Level -- AI: Moderate   Reference: Moderate   [MATCH]

...

====================================================================
                    EVALUATION SUMMARY
====================================================================
  Metric                              Score   Quality
--------------------------------------------------------------------
  ROUGE-1  (word overlap)            0.5734   [Good]
  ROUGE-2  (phrase overlap)          0.3219   [Fair]
  ROUGE-L  (sequence match)          0.4861   [Fair]
  BLEU     (word precision)          0.2547   [Poor]
--------------------------------------------------------------------
  Risk Level Accuracy            7/8 samples matched correctly
====================================================================
```

---

## Concepts Explained Simply

| Concept | Simple explanation |
|---|---|
| **ROUGE** | Checking if the AI covered all important topics from the human reference |
| **BLEU** | Checking if the AI used the right words precisely |
| **Reference Data** | Human-written "correct answers" to compare the AI against |
| **Evaluation** | Objectively measuring how good the AI is — with real numbers |

---

## Notes

- The evaluation script respects the **Gemini free tier** (5 requests/minute) by adding a small delay between API calls.
- If a rate limit error occurs, it automatically waits and retries.
- All results are saved as JSON in the `results/` folder for reporting purposes.
