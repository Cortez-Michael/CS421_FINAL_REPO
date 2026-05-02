"""
Emotion + Dialect Classifier
------------------------------
Parses a CSV with columns: id, text/tweet, anger, disgust, fear, joy, sadness, surprise
Combines active emotion columns into a single "emotion" field (e.g. "joy,sadness"),
uses Ollama to classify Spanish dialect, then writes separate CSV files per dialect.

Setup:
    1. Install Ollama: https://ollama.com
    2. Pull a model:
         ollama pull qwen2.5:14b
    3. Make sure Ollama is running:
         ollama serve

Usage:
    python parse_dialects.py --input data.csv
    python parse_dialects.py --input data.csv --model qwen2.5:14b
"""

import argparse
import csv
import json
import os
import sys
import time
import re
from collections import defaultdict

import requests

# ── Config ────────────────────────────────────────────────────────────────────

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:14b"
OUTPUT_DIR    = "dev_kit"

EMOTION_COLS   = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
OUTPUT_COLUMNS = ["id", "tweet", "emotion", "offensive", "dialect", "confidence"]

SYSTEM_PROMPT = """You are a Spanish dialect expert. Your only job is to identify
the regional variety of Spanish in a given sentence.

Choose ONE label from this list:
 mexican, argentinian, colombian, spanish_spain, chilean, peruvian, 
 venezuelan, cuban, puerto_rican, dominican, uruguayan, paraguayan,
 bolivian, ecuadorian, guatemalan, honduran, salvadoran, nicaraguan,
 costa_rican, panamanian, unknown

Reply with ONLY valid JSON in this exact format — no preamble, no explanation:
{"dialect": "<label>", "confidence": "<high|medium|low>"}"""

# ── Emotion resolver ──────────────────────────────────────────────────────────

def resolve_emotions(row):
    """Supports both a single 'label' column and binary emotion columns."""
    # New format: single label column
    if "label" in row and row["label"].strip():
        return row["label"].strip().lower()
    
    # Old format: binary emotion columns (anger=1, joy=1, etc.)
    active = [emotion for emotion in EMOTION_COLS if row.get(emotion, "0").strip() == "1"]
    return ",".join(active) if active else "none"


# ── Ollama classification ─────────────────────────────────────────────────────

def classify_dialect(text, model, retries=3):
    """Send a single tweet to Ollama and return the parsed dialect result."""
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": 'Classify the dialect of this Spanish sentence:\n\n"' + text + '"',
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            raw    = response.json().get("response", "{}")
            result = json.loads(raw)
            dialect    = result.get("dialect", "unknown").strip().lower().replace(" ", "_")
            confidence = result.get("confidence", "low")
            return {"dialect": dialect, "confidence": confidence}
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            print("  [WARN] Attempt {} failed: {}".format(attempt, e), file=sys.stderr)
            if attempt < retries:
                time.sleep(2 ** attempt)
    return {"dialect": "unknown", "confidence": "low"}


# ── CSV helpers ───────────────────────────────────────────────────────────────

def get_writer(dialect, output_dir, file_handles, writers):
    """Return (or create) a CSV writer for the given dialect."""
    if dialect not in writers:
        safe_name = dialect.replace("/", "_").replace("\\", "_")
        path = os.path.join(output_dir, safe_name + ".csv")
        fh = open(path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        file_handles[dialect] = fh
        writers[dialect] = writer
        print("  [INFO] Created output file: " + path)
    return writers[dialect]


# ── Input parsing ─────────────────────────────────────────────────────────────

def parse_input(path, delimiter):
    """Read the input file and return a list of row dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        reader.fieldnames = [c.strip() for c in reader.fieldnames]
        return [{k.strip(): v.strip() for k, v in row.items() if k is not None} for row in reader]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classify emotion+dialect for Spanish tweets")
    parser.add_argument("--input",      required=True,       help="Path to input CSV file")
    parser.add_argument("--output_dir", default=OUTPUT_DIR,  help="Output folder (default: taskA_regional/)")
    parser.add_argument("--model",      default=DEFAULT_MODEL, help="Ollama model (default: qwen2.5:14b)")
    parser.add_argument("--delimiter", default="\t", help="File delimiter (default: tab)")
    args = parser.parse_args()

    delimiter = args.delimiter.encode().decode("unicode_escape")

    try:
        requests.get("http://localhost:11434", timeout=5)
    except requests.ConnectionError:
        print("\n[ERROR] Cannot reach Ollama at localhost:11434.\nMake sure Ollama is running:  ollama serve\n", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    rows  = parse_input(args.input, delimiter)
    total = len(rows)

    print("Reading input  : " + args.input)
    print("Ollama model   : " + args.model)
    print("Output folder  : " + args.output_dir)
    print("Rows loaded    : " + str(total))
    print("")

    file_handles = {}
    writers      = {}
    stats        = defaultdict(int)

    for i, row in enumerate(rows, 1):
        # 1. Grab the original dirty tweet (Safely checks for 'tweet' OR 'text' columns)
        raw_text = row.get("tweet", row.get("text", ""))
        
        # 2. Make a clean copy specifically for Qwen to read
        clean_text = re.sub(r'\b(HASHTAG|URL|USER)\b', '', raw_text, flags=re.IGNORECASE)
        clean_text = " ".join(clean_text.split())
        
        emotion = resolve_emotions(row)
        preview = raw_text[:65] + "..." if len(raw_text) > 65 else raw_text
        print("[{}/{}] {}".format(i, total, preview))
        print("          -> emotion={}".format(emotion))

        # 3. Check the clean copy. If it's empty, skip Ollama.
        if not clean_text.strip():
            dialect = "unknown"
            confidence = "low"
            print("          -> dialect={}  confidence={} (Empty after cleanup)".format(dialect, confidence))
        else:
            # Send the clean copy to Qwen
            result     = classify_dialect(clean_text, args.model)
            dialect    = result["dialect"]
            confidence = result["confidence"]
            print("          -> dialect={}  confidence={}".format(dialect, confidence))

        out_row = {
            "id":         row.get("id", ""),
            "tweet":      raw_text,  # We save the original dirty text to the CSV
            "emotion":    emotion,
            "offensive":  row.get("offensive", "0"), # Grab 'offensive' if it exists, otherwise '0'
            "dialect":    dialect,
            "confidence": confidence,
        }

        writer = get_writer(dialect, args.output_dir, file_handles, writers)
        writer.writerow(out_row)
        stats[dialect] += 1

    for fh in file_handles.values():
        fh.close()

    print("\n-- Summary ------------------------------------------")
    for dialect, count in sorted(stats.items(), key=lambda x: -x[1]):
        print("  {:<25} {:>4} row(s)".format(dialect, count))
    print("  {:<25} {:>4}".format("TOTAL", total))
    print("\nDone! CSV files saved to: " + os.path.abspath(args.output_dir) + "/")


if __name__ == "__main__":
    main()