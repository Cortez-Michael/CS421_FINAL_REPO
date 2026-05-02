import os
import re
import json
import requests
import pandas as pd
from sklearn.metrics import classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==========================================
# 1. Configuration
# ==========================================
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

def save_report_to_csv(report_dict, input_path, output_folder, model_name="Llama"):
    """Converts a classification_report into a structured CSV (Identical to RoBERTuito)."""
    rows = []
    aggregate_keys = {"accuracy", "macro avg", "weighted avg"}
    
    for label, metrics in report_dict.items():
        if label in aggregate_keys:
            continue
        rows.append({
            "model":     model_name,
            "class":     label,
            "precision": round(metrics["precision"], 4),
            "recall":    round(metrics["recall"],    4),
            "f1_score":  round(metrics["f1-score"],  4),
            "support":   int(metrics["support"]),
            "row_type":  "per_class",
        })
 
    for agg_key in ["macro avg", "weighted avg"]:
        if agg_key in report_dict:
            m = report_dict[agg_key]
            rows.append({
                "model":     model_name,
                "class":     agg_key,
                "precision": round(m["precision"], 4),
                "recall":    round(m["recall"],    4),
                "f1_score":  round(m["f1-score"],  4),
                "support":   int(m["support"]),
                "row_type":  "aggregate",
            })
 
    if "accuracy" in report_dict:
        # Proper Pandas alignment for the accuracy row
        acc_row = pd.DataFrame([{
            "model":     model_name,
            "class":     "accuracy",
            "precision": "",
            "recall":    "",
            "f1_score":  round(report_dict["accuracy"], 4),
            "support":   "",
            "row_type":  "aggregate",
        }])
        
        df = pd.DataFrame(rows)
        df['precision'] = df['precision'].astype(object)
        df['recall'] = df['recall'].astype(object)
        df['support'] = df['support'].astype(object)
        df = pd.concat([df, acc_row], ignore_index=True)
    else:
        df = pd.DataFrame(rows)
 
    dialect = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_folder, f"{dialect}_{model_name}_classification_report.csv")
    
    df.to_csv(output_path, index=False)
    print(f"[✓] Report saved to: {output_path}")
    return df

def run_dialect_analysis(csv_file, output_folder):
    """Processes a single CSV using Llama and generates the prediction and report files concurrently."""
    dialect = os.path.splitext(os.path.basename(csv_file))[0]
    output_path_preds = os.path.join(output_folder, f"{dialect}_{OLLAMA_MODEL.split(':')[0]}_predictions.csv")

    # Load dataset
    df = pd.read_csv(csv_file, on_bad_lines='skip', engine='python')
    
    # --- THE FIX: KILL THE GHOST ROWS AND RESET INDEX ---
    # Resetting the index is mandatory for multithreading so our list indices map perfectly
    df = df.dropna(subset=['tweet', 'emotion']).reset_index(drop=True)
    
    if len(df) == 0:
        print(f"[!] {dialect}.csv is completely empty after cleaning. Skipping this file...")
        return
    # ------------------------------------
    
    num_rows = len(df)
    print(f"Starting concurrent inference on {num_rows} rows for {dialect}...")
    
    # Pre-allocate arrays to maintain strict order during multithreading
    predictions = [None] * num_rows
    keywords_list = [None] * num_rows
    reasoning_list = [None] * num_rows
    true_labels = [None] * num_rows

    def process_tweet(index, row):
        """Worker function to process a single tweet."""
        raw_text = str(row['tweet']) if pd.notna(row['tweet']) else ""
        true_emotion = str(row['emotion']).lower().strip() if pd.notna(row['emotion']) else "unknown"
        
        text_cleaned = re.sub(r'\b(HASHTAG|URL|USER)\b', '', raw_text, flags=re.IGNORECASE)
        text_cleaned = " ".join(text_cleaned.split())
        
        if not text_cleaned.strip():
            return index, "others", "", "Text was empty after stripping placeholders.", true_emotion
            
        prompt = f"""You are an expert linguist analyzing regional Spanish dialects.
Analyze the following text and classify its core emotion into exactly one of these: [anger, fear, joy, sadness, disgust, surprise, others].
If the emotion is ambiguous or does not strongly fit the first six categories, you MUST classify it as "others". Do not invent new emotions.

Also, extract the specific regional slang or keywords that influenced your decision, and briefly explain your reasoning.

--- EXAMPLES ---

Example 1:
Text: "¡Qué chido estuvo el festival, güey! La neta me la pasé a toda madre. 🔥"
{{
    "emotion": "joy",
    "keywords": ["chido", "güey", "La neta", "a toda madre"],
    "context": "'Chido' indicates that something is excellent or cool. 'A toda madre' is a highly expressive phrase signifying a fantastic experience, frequently utilized alongside 'güey' (dude)."
}}

Example 2:
Text: "Me caga que la gente maneje así. Estoy bien encabronado con este pinche tráfico. 🤬🚗"
{{
    "emotion": "anger",
    "keywords": ["Me caga", "encabronado", "pinche"],
    "context": "The phrase 'me caga' in this context expresses severe annoyance. 'Encabronado' indicates deep fury, and 'pinche' acts as a strong, colloquial intensifier."
}}

Example 3:
Text: "Chale, ando bien agüitado porque no alcancé boletos para el cine. Qué mala onda. 😞"
{{
    "emotion": "sadness",
    "keywords": ["Chale", "agüitado", "mala onda"],
    "context": "'Chale' expresses resignation or disappointment, while 'agüitado' is the standard Mexican terminology for feeling saddened, depressed, or disheartened."
}}

Example 4:
Text: "¡No manches! ¿Es neta que cancelaron las clases de mañana? Qué locura. 🤯"
{{
    "emotion": "surprise",
    "keywords": ["No manches", "neta"],
    "context": "'No manches' is a ubiquitous expression of shock or disbelief. '¿Es neta?' translates directly to 'Is it the truth/for real?', serving as a definitive marker of surprise."
}}

Example 5:
Text: "Ahorita salgo de la chamba y paso al Oxxo a comprar algo, te aviso cuando llegue. 🥤"
{{
    "emotion": "others",
    "keywords": ["Ahorita", "chamba", "Oxxo"],
    "context": "A factual, everyday statement. 'Ahorita' denotes an unspecified near timeframe, 'chamba' means work or employment, and the mention of 'Oxxo' is a convenience store doesnt conveying strong emotional polarity."
}}

--- NOW IT IS YOUR TURN ---

You MUST respond ONLY with a valid JSON object in this exact format. Do not add markdown, code blocks, or extra text.
{{
    "emotion": "emotion_word_here",
    "keywords": ["word1", "word2", ...],
    "context": "brief explanation of why these words indicate the emotion"
}}

Text: "{text_cleaned}"
"""
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_ctx": 4096
            }
        }
        
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            raw_output = response.json().get('response', '{}')
            result_dict = json.loads(raw_output)
            
            pred_emotion = result_dict.get('emotion', 'error').lower().strip()
            
            allowed_emotions = {"anger", "disgust", "fear", "joy", "sadness", "surprise", "others"}
            if pred_emotion not in allowed_emotions and pred_emotion != "error":
                pred_emotion = "others"
                
            keys = ", ".join(result_dict.get('keywords', []))
            context = result_dict.get('context', 'No context')
        except Exception as e:
            pred_emotion, keys, context = "error", "error", f"Model Error: {str(e)}"
            
        return index, pred_emotion, keys, context, true_emotion

    # --- EXECUTE MULTITHREADING ---
    completed = 0
    # max_workers=8 maps exactly to OLLAMA_NUM_PARALLEL=8
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all jobs to the pool
        futures = {executor.submit(process_tweet, i, row): i for i, row in df.iterrows()}
        
        # As each tweet finishes processing, lock it into its correct chronological spot
        for future in as_completed(futures):
            i, pred_emotion, keys, context, true_emotion = future.result()
            
            predictions[i] = pred_emotion
            keywords_list[i] = keys
            reasoning_list[i] = context
            true_labels[i] = true_emotion
            
            completed += 1
            if completed % 100 == 0:
                print(f"Processed {completed}/{num_rows} rows concurrently...")

    results_df = pd.DataFrame({
        "model_type": "Llama",
        "text": df['tweet'],
        "true_emotion": true_labels,
        "predicted_emotion": predictions,
        "top_keywords": keywords_list,
        "reasoning": reasoning_list
    })
    results_df.to_csv(output_path_preds, index=False)
    print(f"[✓] Predictions saved to: {output_path_preds}")

    valid_indices = [i for i, p in enumerate(predictions) if p != "error"]
    y_true = [true_labels[i] for i in valid_indices]
    y_pred = [predictions[i] for i in valid_indices]
    
    if len(y_true) > 0:
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        print("\n--- Evaluation Metrics ---")
        print(classification_report(y_true, y_pred, zero_division=0))
        save_report_to_csv(report_dict, csv_file, output_folder, model_name="Llama")
    else:
        print(f"[!] No valid predictions to evaluate for {dialect}.")

if __name__ == "__main__":
    input_file = 'input_data/mexican.csv' 
    output_folder = 'results/few-shot/Llama'
    
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"[!] Error: Could not find '{input_file}'. Make sure the file name and path are correct.")
    else:
        print(f"\n==============================================")
        print(f" Analyzing Dataset: {os.path.basename(input_file)}")
        print(f"==============================================")
        run_dialect_analysis(input_file, output_folder)
        
        print("\n✅ Dataset processed! You can find the results in the Llama folder.")
