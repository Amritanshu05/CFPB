# CFPB Complaint-to-Resolution Assistant

A research-oriented, modular NLP pipeline that transforms raw CFPB consumer
complaint narratives into structured, evidence-grounded resolution outputs
for human review.

**This is a research prototype, not a product.** It combines supervised
classification, hybrid retrieval (BM25 + dense embeddings), and
retrieval-augmented generation (RAG) to produce measurable, explainable
outputs.

---

## Project Goal

> Given a raw consumer complaint narrative, the system classifies it,
> retrieves similar historical cases, and generates a structured JSON
> summary — grounded in real evidence — suitable for human analyst review.

Output schema (stable across every provider):

```json
{
  "complaint_summary": "...",
  "predicted_issue": "credit_reporting",
  "similar_cases": ["...", "..."],
  "recommended_next_step": "Contact company for response",
  "confidence": 0.82,
  "needs_human_review": false,
  "generation_mocked": false
}
```

`generation_mocked=true` signals that the summary was produced by the
deterministic rule-based mock generator (no LLM was called). See the
"What is mocked" section below.

---

## What is working / what is mocked / what is missing

| Area | Status | Notes |
| --- | --- | --- |
| Chunked ingest of 8GB CFPB CSV | Working | `cfpb_assistant.data.loader` streams the file; never fully loaded. |
| Cleaning + label normalization | Working | Post-clean length filter removes XXXX-only narratives. |
| Stratified sample + train/val/test splits | Working | Saved to `data/processed/*.parquet`. |
| Classical classifiers (LR, SVM, NB) | Working | TF-IDF + sklearn; confidence via `predict_proba`. |
| Transformer classifier (DistilBERT) | Working | GPU-aware (`torch.cuda`); see `configs/classification.yaml`. |
| BM25 retrieval (rank-bm25) | Working | Index saved to `data/indexes/bm25/`. |
| Dense retrieval (FAISS + sentence-transformers) | Working | GPU-accelerated encode; model-name validated on load. |
| Hybrid retrieval (RRF or weighted) | Working | Backed by the two above. |
| Rule-based extractor (urgency, product, etc.) | Working | Feeds the pipeline, not a final output. |
| Pipeline generator | Working | Preprocesses input with `clean_narrative` before classify/retrieve. |
| LLM generation (OpenAI / Anthropic) | Depends on API key | Fully implemented, requires `.env` with a real key. |
| Mock generation | Working and EXPLICIT | Emits `generation_mocked=true`, confidence = classifier prob. |
| Classification evaluation | Working | Accuracy / Macro-F1 / per-class / confusion matrix CSV. |
| Retrieval evaluation | Working (proxy) | Label-match proxy (product and product+issue) + per-class P@k breakdown. |
| Generation evaluation | Working (partially mocked) | Schema validity, label fidelity, grounding proxy, confidence distribution, abstention rate. See module docstring for limits. |
| Human-annotated relevance / faithfulness | **Missing** | Future work; the retrieval and generation eval clearly flag their proxies. |
| ROUGE / BERTScore against gold summaries | **Missing** | Dataset has no reference summaries; would need human annotation. |
| Calibration curves (ECE) | **Missing** | Hooks are in place (confidence distributions are logged). |

If something is not listed above, assume it's not implemented.

---

## Project Structure

```
ResearchInterm/
├── configs/                 # YAML configs (paths / preprocessing / classification / retrieval / generation)
├── data/
│   ├── raw/cfbp/            # Raw CFPB CSV (not tracked)
│   ├── processed/           # train/val/test parquet splits
│   ├── samples/             # Stratified sample parquet
│   └── indexes/             # BM25 + FAISS indexes
├── notebooks/
│   └── 01_eda.ipynb         # EDA on the processed sample
├── outputs/
│   ├── models/              # Saved classifiers (.pkl) and transformer dir
│   ├── results/             # Evaluation JSON + confusion matrix CSVs
│   └── logs/                # Daily log files
├── scripts/
│   ├── build_dataset.py     # Sample + splits from raw CSV
│   ├── train_classifier.py  # Train baselines and/or transformer
│   ├── build_index.py       # BM25 + dense FAISS indexes
│   ├── run_pipeline.py      # End-to-end inference
│   └── evaluate.py          # Classification / retrieval / generation eval
├── src/cfpb_assistant/
│   ├── data/                # loader, preprocessor, sampler
│   ├── classification/      # baselines.py (TF-IDF) + transformer.py (GPU)
│   ├── retrieval/           # bm25_retriever, dense_retriever, hybrid_retriever
│   ├── extraction/          # extractor.py (rule-based)
│   ├── generation/          # prompts.py, generator.py
│   ├── evaluation/          # classification_eval, retrieval_eval, generation_eval
│   └── utils/               # config.py, logging_utils.py
├── requirements.txt
├── setup.py
└── README.md
```

Notebooks 02/03/04 referenced in earlier drafts are not yet implemented.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # Linux/Mac

pip install -r requirements.txt
pip install -e .

# Optional — only for real LLM generation
copy .env.example .env
```

### GPU (recommended for the transformer classifier)

If you have an NVIDIA GPU, install a CUDA build of PyTorch:

```bash
pip install --upgrade torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

The transformer trainer and the dense retriever automatically use CUDA
when available and fall back to CPU otherwise. Check with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

---

## Reproducing all results from scratch

```bash
# 1. Build dataset (chunked ingest, stratified sample, splits)
python scripts/build_dataset.py --force

# 2. Build retrieval indexes (BM25 + dense FAISS)
python scripts/build_index.py --force

# 3. Train classifiers
python scripts/train_classifier.py --model baselines      # LR, SVM, NB
python scripts/train_classifier.py --model transformer    # DistilBERT on GPU

# 4. Run the full pipeline on one complaint (mock mode)
python scripts/run_pipeline.py --complaint "I was charged twice for the same transaction and the bank refuses to refund me."

# 5. Evaluate everything
python scripts/evaluate.py --task all --n-gen-samples 30
```

All artifacts end up under `data/processed/`, `data/indexes/`,
`outputs/models/`, and `outputs/results/`. No manual edits are needed.

---

## Data Source

**CFPB Consumer Complaint Database** — https://www.consumerfinance.gov/data-research/consumer-complaints/

The raw CSV (~8 GB, ~25M rows) is not tracked in this repository. Place it at:
```
data/raw/cfbp/consumer_complaints.csv
```

---

## Confidence / abstention semantics

- **Classical baselines (LR, SVM, NB)**: confidence = calibrated
  `predict_proba` max.
- **Transformer**: confidence = softmax max of the fine-tuned head.
- **Mock generation**: confidence = classifier probability ONLY. We do
  not emit a synthetic LLM confidence. `generation_mocked=true` is set
  so downstream consumers never mistake this for a calibrated LLM.
- **Real LLM generation (OpenAI / Anthropic)**: confidence =
  `min(classifier_prob, llm_reported_confidence)`.
- `needs_human_review = True` iff `confidence < confidence_threshold`
  (configured in `configs/generation.yaml`, default 0.6) OR the LLM
  explicitly flagged the case.

---

## Research Context

Designed to be compared against:

- Classical ML baselines (TF-IDF + SVM / NB / LR) — **implemented**
- Transformer classifier (DistilBERT) — **implemented**
- Retrieval-only systems (BM25, dense) — **implemented**
- Hybrid retrieval (RRF) — **implemented**
- RAG-augmented LLM (OpenAI / Anthropic) — **implemented, requires API key**
