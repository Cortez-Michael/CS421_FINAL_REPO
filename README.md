# The Dialectal Gap: Evaluating Emotion Detection Reliability Across Regional Spanish Varieties

# Spanish Dialect Emotion Classifier

A multilingual emotion classification pipeline for Spanish tweet dialects (Mexican, Argentinian, Venezuelan, Spain Spanish). Supports transformer-based models (mBERT, XLM-RoBERTa) and LLM-based models (Gemma, Llama) with both zero-shot and few-shot prompting, plus visualization tools.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Hugging Face Setup](#hugging-face-setup)
- [Ollama Setup](#ollama-setup)
- [Input Data Format](#input-data-format)
- [Running the Models](#running-the-models)
- [Running the Parser](#running-the-parser)
- [Running the Visualizations](#running-the-visualizations)

---

## Project Structure

```
project/
├── input_data/                      # Place your dialect CSVs here
│   ├── mexican.csv
│   ├── argentinian.csv
│   ├── venezuelan.csv
│   └── spanish_spain.csv
├── models/
│   ├── XLM_roberta.py               # XLM-RoBERTa transformer model
│   ├── emo_parse.py                 # Dialect + emotion CSV parser
│   ├── gemma.py                     # Gemma zero-shot (Ollama)
│   ├── gemma_few_shot.py            # Gemma few-shot (Ollama)
│   ├── llama.py                     # Llama zero-shot (Ollama)
│   ├── llama_few_shot.py            # Llama few-shot (Ollama)
│   └── mBERT.py                     # mBERT transformer model
├── raw_data/                        # Raw/unprocessed source data
├── results/
│   └── All_results/                 # Output predictions and reports (auto-created)
│       ├── Gemma/
│       ├── Gemma_Few_Shot/
│       ├── Llama/
│       ├── Llama_Few_Shot/
│       ├── XLM-RoBERTa/
│       └── mBERT/
├── visualize/
│   ├── plots/                       # All generated plots (auto-created)
│   │   ├── analysis/                # Cleanliness scores, label distributions
│   │   ├── comparison/              # Per-model and per-dialect F1 comparisons
│   │   ├── confusion/               # Confusion matrix heatmaps
│   │   ├── heatmaps/                # Emotion F1 heatmaps
│   │   ├── keywords/                # Raw token keyword bar charts
│   │   ├── keywords_full_words/     # Filtered full-word keyword bar charts
│   │   ├── average_macro_f1_bar.png
│   │   ├── average_weighted_f1_bar.png
│   │   └── single_grouped_f1_comparison.png
│   ├── keyword_plots.py             # Keyword frequency plots
│   └── visualize.py                 # F1 scores, confusion matrices, heatmaps
└── README.md
```

---

## Requirements

**Python version:** 3.9 or higher

### Core Dependencies

```
torch
transformers
huggingface_hub
python-dotenv
scikit-learn
pandas
numpy
scipy
shap
requests
matplotlib
seaborn
wordcloud
spacy
```

---

## Installation

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 2. Install all Python packages

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install transformers huggingface_hub python-dotenv scikit-learn pandas numpy scipy shap requests matplotlib seaborn wordcloud spacy
```

### 3. Download the spaCy Spanish language model

Required for keyword filtering in `keyword_plots.py`:

```bash
python -m spacy download es_core_news_sm
```

---

## Hugging Face Setup

The transformer models (`mBERT.py`, `XLM_roberta.py`) download their weights from Hugging Face. Some models require authentication.

### 1. Create a Hugging Face account and generate an access token

Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a token with **Read** permissions.

### 2. Create a `.env` file in the project root

```
HF_TOKEN=hf_your_token_here
```

The scripts load this automatically via `python-dotenv`.

### Models used

| Script | Hugging Face Model |
|---|---|
| `mBERT.py` | `AnasAlokla/multilingual_go_emotions` |
| `XLM_roberta.py` | `tabularisai/multilingual-emotion-classification` |


---

## Ollama Setup

Gemma and Llama models run locally via [Ollama](https://ollama.com).

### 1. Install Ollama

Download and install from [https://ollama.com/download](https://ollama.com/download).

### 2. Start the Ollama server

```bash
ollama serve
```

Keep this running in a separate terminal for the duration of your session.

### 3. Pull the required models

```bash
ollama pull gemma4:e4b
ollama pull llama3.1:8b
ollama pull qwen2.5:14b        # Required for emo_parse.py only
```

---

## Input Data Format

Each dialect CSV should be placed in the `input_data/` folder and must contain at minimum these two columns:

| Column | Description |
|---|---|
| `tweet` | The raw tweet text |
| `emotion` | The ground truth label: `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`, or `others` |

Placeholder tokens (`HASHTAG`, `URL`, `USER`) in tweet text are automatically stripped by all models before inference.

---

## Running the Models

All model scripts are run from inside the `models/` directory, or adjust the `input_data/` path accordingly.

### mBERT

Runs on all four dialects automatically.

```bash
python models/mBERT.py
```

Output is saved to `results/mBERT/`.

### XLM-RoBERTa

Runs on all four dialects automatically.

```bash
python models/XLM_roberta.py
```

Output is saved to `results/XLM-RoBERTa/`.

### Gemma (Zero-Shot)

Runs on all CSVs found in `input_data/`.

```bash
python models/gemma.py
```

Output is saved to `results/Gemma/`.

### Gemma (Few-Shot)

Edit the `input_file` variable at the bottom of the script to point to the dialect you want to run, then:

```bash
python models/gemma_few_shot.py
```

Output is saved to `results/few-shot/Gemma/`.

### Llama (Zero-Shot)

Runs on all CSVs found in `input_data/`.

```bash
python models/llama.py
```

Output is saved to `results/Llama/`.

### Llama (Few-Shot)

Edit the `input_file` variable at the bottom of the script to point to the dialect you want, then:

```bash
python models/llama_few_shot.py
```

This script runs inference concurrently using 8 threads. Output is saved to `results/few-shot/Llama/`.

---

## Running the Parser

`emo_parse.py` takes a raw dataset with binary emotion columns or a single label column, classifies each tweet's Spanish dialect using Qwen via Ollama, and splits the output into per-dialect CSV files.

Make sure `ollama serve` is running and `qwen2.5:14b` is pulled before running.

```bash
python models/emo_parse.py --input path/to/your_data.csv
```

### Optional arguments

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to the input CSV |
| `--output_dir` | `dev_kit/` | Folder to write per-dialect output CSVs |
| `--model` | `qwen2.5:14b` | Ollama model to use for dialect classification |
| `--delimiter` | `\t` (tab) | Column delimiter in the input file |

### Expected input columns

The script supports two formats:

- **Binary format:** columns named `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise` each set to `0` or `1`
- **Single label format:** a column named `label` containing the emotion string

It also expects either a `tweet` or `text` column for the tweet content.

---

## Running the Visualizations

Both visualization scripts expect results to be organized under `results/All_results/<ModelName>/` with prediction and classification report CSVs inside. Adjust the glob paths at the top of each script if your folder layout differs.

### Main visualization suite

Generates F1 bar charts, macro F1 comparisons, confusion matrices, heatmaps, weighted F1 graphs, and label distribution pie charts.

```bash
python visualize/visualize.py
```

Plots are saved to `visualize/plots/` in subdirectories (`comparison/`, `confusion/`, `heatmaps/`, `analysis/`).

### Keyword plots

Generates per-dialect and combined keyword frequency bar charts grouped by emotion.

```bash
python visualize/keyword_plots.py
```

Plots are saved to `visualize/plots/keywords/` and `visualize/plots/keywords_full_words/`.

> **Note:** `keyword_plots.py` requires the spaCy Spanish model. If it is not installed, the script falls back to basic length-based filtering and logs a warning.

---

## Quick Start Checklist

```
[ ] Python 3.9+ installed
[ ] Virtual environment created and activated
[ ] pip packages installed (torch CPU build + all others)
[ ] spaCy es_core_news_sm downloaded
[ ] .env file created with HF_TOKEN
[ ] Ollama installed and ollama serve running
[ ] Required Ollama models pulled (gemma4:e4b, llama3.1:8b, qwen2.5:14b)
[ ] Dialect CSVs placed in input_data/
[ ] Run desired model script
[ ] Run visualize/visualize.py to generate plots
```
