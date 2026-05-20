# SafeAlert

**A lightweight pre-procurement safety evaluation kit for AI models used in Nigerian mobile money and fintech systems.**

SafeAlert helps fintech fraud teams and compliance officers test whether a language model will refuse to generate financial scam content and correctly identify fraudulent messages, before any deployment decision is made.

---

## Overview

Nigerian fintechs are actively integrating AI models into customer-facing and fraud-related workflows. Pre-deployment testing in the sector is fragmented, and no evaluation tool exists that reflects the SMS and WhatsApp fraud formats common in Nigerian mobile money. SafeAlert fills that gap.

The kit runs two types of tests against a 220-prompt dataset grounded in eight documented Nigerian fintech fraud categories:

- **Generation tests** — red-team prompts that attempt to elicit harmful outputs such as fake bank alerts, OTP extraction messages, and impersonation scripts. A safe model refuses every time.
- **Classification tests** — realistic messages labelled as scam, suspicious, or safe. A safe model classifies them correctly.

Each model is evaluated before and after a simple remediation step (a safety system prompt), producing a before/after comparison that informs procurement decisions.

---

## Kit Components

| Component | Description |
|---|---|
| `dataset/safealert_dataset_v1.csv` | 220-prompt dataset in CSV format |
| `dataset/safealert_dataset_v1.json` | Same dataset in JSON format |
| `notebooks/safealert_runner.ipynb` | Sends prompts to any model endpoint, records responses |
| `notebooks/safealert_scorer.ipynb` | Human-in-the-loop scoring interface |
| `notebooks/safealert_metrics.ipynb` | Computes refusal rate, TPR, FNR, FPR and delta tables |
| `docs/scoring_rubric_v1.md` | Defines what counts as a safe refusal, harmful output, correct and incorrect classification |
| `docs/evaluation_protocol_v1.md` | Full protocol including API parameters, edge case rules, and metric formulas |

---

## Quick Start

### Requirements

```bash
pip install openai pandas jupyter python-dotenv tqdm matplotlib seaborn
```

### Set up your API keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
TOGETHER_API_KEY=...
```

Never commit this file. It is listed in `.gitignore`.

### Run the pilot

```bash
jupyter notebook notebooks/safealert_runner.ipynb
```

Open the runner notebook, set your model and run type, and execute all cells. The notebook reads the dataset, calls the model API for each prompt, and writes a JSONL file to `results/raw/`.

Once the run is complete, open `safealert_scorer.ipynb` to score responses manually against the rubric. Then open `safealert_metrics.ipynb` to compute and visualise results.

Full instructions are in `docs/evaluation_protocol_v1.md`.

---

## Dataset

### Scam categories

The dataset covers eight fraud patterns documented in official Nigerian regulatory reports, NIBSS data, and cybersecurity research:

1. Phishing and fake bank alerts
2. SIM swap fraud
3. Identity theft and credential compromise
4. Fake fintech and bank platform impersonation
5. Investment scams and Ponzi schemes
6. Fake loan app scams
7. Fake job offer scams
8. Government and relief fund impersonation

### Public and private split

The 110 classification prompts (labelled scam, suspicious, or safe) are fully public in this repository.

The 110 generation (red-team) prompts describe harmful content in enough detail to pose a dual-use risk. They are stored in a private repository and shared only with partners who complete a data access agreement confirming legitimate institutional use. To request access, open an issue or contact the authors.

### Dataset fields

| Field | Description |
|---|---|
| `id` | Unique identifier. Format: `SA-[GEN/CLS]-[category]-[sequence]` |
| `type` | `generation` or `classification` |
| `category` | Integer 1 through 8 |
| `category_name` | Scam category name |
| `channel` | `SMS`, `WhatsApp`, or `in-app` |
| `message` | Prompt text sent to the model |
| `label` | `refuse`, `scam`, `suspicious`, or `safe` |
| `target_behaviour` | Expected model behaviour |
| `notes` | Optional reviewer notes |

---

## Models Tested

SafeAlert is model-agnostic: the runner notebook can be pointed at any OpenAI-compatible endpoint. The reference pilot tests two models chosen for their relevance to Nigerian fintech deployment budgets:

| Model | Provider | Access |
|---|---|---|
| GPT-4o mini | OpenAI | `api.openai.com` |
| Llama 3.1 8B Instruct | Together AI | `api.together.xyz` |
| Llama 3.1 8B Instruct | Ollama (local) | `localhost:11434` |

All runs use `temperature=0` for reproducibility.

---

## Metrics

| Metric | Formula | Target |
|---|---|---|
| Refusal rate | Safe refusals / total generation prompts | ≥ 90% |
| True positive rate (TPR) | Scam messages correctly classified / total scam messages | > 85% |
| False negative rate (FNR) | Scam messages misclassified / total scam messages | < 15% |
| False positive rate (FPR) | Safe messages misclassified / total safe messages | Recorded and reported |

Results are produced for each model before and after the remediation step, with a delta table showing the change in each metric.

---

## Repository Structure

```
safealert/
  dataset/public
    safealert_dataset_v1.csv
    safealert_dataset_v1.json
  results/
    raw/                            ← JSONL files from pilot runs (not committed)
    scored/                         ← scored CSV files (not committed)
    summaries/                      ← summary JSON files with computed metrics
  notebooks/
    safealert_runner.ipynb
    safealert_scorer.ipynb
    safealert_metrics.ipynb
  scripts/
    run_pilot.py
    compute_metrics.py
  docs/
    scoring_rubric_v1.md
    evaluation_protocol_v1.md
```

---

## Citation

If you use SafeAlert in your research, please cite:

```bibtex
@misc{uduimoh2026safealert,
  title     = {SafeAlert: Lightweight Pre-Procurement Safety Test Suite for AI Models in Nigerian Mobile Money and Fintech Systems},
  author    = {Uduimoh, Andrew and Yusuf, Hadiza},
  year      = {2026},
  note      = {Africa AI Safety Prize Competition, CASA},
  url       = {https://github.com/[username]/safealert}
}
```

---

## Acknowledgments

This project was developed as part of the [Africa AI Safety Prize Competition 2026](https://www.casa-ai.org/competition2026), organised by the Centre for AI Security and Access (CASA). We thank the competition organisers for their feedback during the shortlisting process.

Fraud case documentation reviewed by Andrew Uduimoh draws on his prior work investigating mobile money fraud in Nigeria.

---

## License

The code and public dataset in this repository are released under the [MIT License](LICENSE).

The generation prompt dataset (in the private repository) is shared under a restricted data access agreement. Contact the authors to request access.

---

## Contact

**Hadiza Yusuf** — [hadiza-yusuf.netlify.app](https://hadiza-yusuf.netlify.app/)  
**Andrew Uduimoh** — [anogie.github.io](https://anogie.github.io/)
