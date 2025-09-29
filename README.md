# BM25 + SciBERT Hybrid Retrieval (CISI, CSV)

This project implements a **hybrid information retrieval system** that combines **BM25** (lexical retrieval) with **SciBERT** (semantic retrieval) via **score fusion**.  
It is evaluated on the **CISI dataset** (1,460 documents, 112 queries, relevance judgments).

📂 **Dataset source:** [CISI – A Dataset for Information Retrieval (Kaggle)](https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval)

---

## Team Members
- **Raditya Maheswara** — Universitas Gadjah Mada (UGM)  
- **Girindra Daafi Mada** — Universitas Gadjah Mada (UGM)  
- **Muhammad Fariz** — Universitas Gadjah Mada (UGM)  
- **Sultan Rizqinta Sinuraya** — Universitas Gadjah Mada (UGM)  
- **Ivan Adito Arba Putra** — Universitas Gadjah Mada (UGM)

**Course:** Temu Kembali Informasi (Information Retrieval)  
**Supervisor:** Dr. Lukman Heryawan, S.T., M.  
**Affiliation:** Department of Computer Science and Electronics, FMIPA, Universitas Gadjah Mada — Yogyakarta, Indonesia

---

## Features
- **BM25 Retrieval:** Lexical baseline (Pyserini / rank_bm25)
- **SciBERT Retrieval:** Dense embeddings with Hugging Face Transformers
- **Hybrid Fusion:** Weighted interpolation of normalized BM25 & SciBERT scores
- **Evaluation:** Recall@10 and nDCG@10
- **Visualization:** Bar chart comparing models
- **Qualitative Examples:** Top-5 retrieved titles across methods
- **Exports:** Metrics CSVs and qualitative top-5 CSV

---

## Technology Stack
- **Runtime:** Python (Google Colab)
- **Core Libraries:** PyTorch, Transformers (Hugging Face), Pyserini / rank_bm25
- **Eval & Utils:** scikit-learn, pandas, numpy, matplotlib
- **Data Format:** CSV (`cisi_docs.csv`, `cisi_queries.csv`, `cisi_qrels.csv`)

---

## Troubleshooting

### 1) Colab can’t see GPU
- Colab menu → **Runtime → Change runtime type → Hardware accelerator → GPU → Save**  
- Re-run the first install cell.

### 2) Out-of-memory (OOM) with SciBERT
Lower the encoder batch size, e.g.:
```python
corpus_vecs = encode(corpus_texts, batch_size=16)
```

### 3) Package install errors
Install/repair deps:
```bash
pip install pyserini transformers torch scikit-learn matplotlib pandas numpy
```

---

## Installation

1. **Open the notebook in Google Colab.**
2. **Switch runtime to GPU**  
   - **Runtime → Change runtime type → Hardware accelerator → GPU → Save**
3. **Install dependencies** (run the first cell):
   ```bash
   pip install rank_bm25 transformers torch matplotlib pandas numpy
   ```
4. **Upload dataset CSVs** to Colab (Files panel or via `files.upload()` cell):
   - `cisi_docs.csv`
   - `cisi_queries.csv`
   - `cisi_qrels.csv`
5. **Run cells from top to bottom.** The notebook will:
   - Build BM25 index & encode SciBERT embeddings (on GPU)
   - Evaluate **Recall@10** & **nDCG@10**
   - Generate a **bar chart** and a **qualitative top-5 table**
   - Export results to `/content/` (CSV/PNG)


---


## License
This project is intended for academic use only and is not licensed for commercial use.

---
