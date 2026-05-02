import os
import glob
import re
import json
import requests
import pandas as pd
from sklearn.metrics import classification_report

#config
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

def save_report_to_csv(report_dict, input_path, output_folder, model_name="Llama"):
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
        rows.append({
            "model":     model_name,
            "class":     "accuracy",
            "precision": None,
            "recall":    None,
            "f1_score":  round(report_dict["accuracy"], 4),
            "support":   None,
            "row_type":  "aggregate",
        })
 
    df = pd.DataFrame(rows)
    dialect = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_folder, f"{dialect}_{model_name}_classification_report.csv")
    
    df.to_csv(output_path, index=False)
    print(f" Report saved to: {output_path}")
    return df

def run_dialect_analysis(csv_file, output_folder):
    dialect = os.path.splitext(os.path.basename(csv_file))[0]
    output_path_preds = os.path.join(output_folder, f"{dialect}_Llama_predictions.csv")

    # load dataset
    df = pd.read_csv(csv_file, on_bad_lines='skip', engine='python')
    
    predictions = []
    keywords_list = []
    reasoning_list = []
    true_labels = []
    
    print(f"Starting inference on {len(df)} rows for {dialect}...")
    
    for index, row in df.iterrows():
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{len(df)} rows...")
            
        raw_text = str(row['tweet']) if pd.notna(row['tweet']) else ""
        true_emotion = str(row['emotion']).lower().strip() if pd.notna(row['emotion']) else "unknown"
        
        text_cleaned = re.sub(r'\b(HASHTAG|URL|USER)\b', '', raw_text, flags=re.IGNORECASE)
        text_cleaned = " ".join(text_cleaned.split())
        
        if not text_cleaned.strip():
            predictions.append("others") 
            keywords_list.append("")
            reasoning_list.append("Text was empty after stripping placeholders.")
            true_labels.append(true_emotion)
            continue
            
        prompt = f"""You are an expert linguist analyzing regional Spanish dialects.
Analyze the following text and classify its core emotion into exactly one of these: [anger, fear, joy, sadness, disgust, surprise, others].
If the emotion is ambiguous or does not strongly fit the first six categories, you MUST classify it as "others". Do not invent new emotions.

Also, extract the specific regional slang or keywords that influenced your decision, and briefly explain your reasoning.

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
            "temperature": 0.1,
            "format": "json"
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
            
        predictions.append(pred_emotion)
        keywords_list.append(keys)
        reasoning_list.append(context)
        true_labels.append(true_emotion)

    results_df = pd.DataFrame({
        "model_type": "Llama",
        "text": df['tweet'],
        "true_emotion": true_labels,
        "predicted_emotion": predictions,
        "top_keywords": keywords_list,
        "reasoning": reasoning_list
    })
    results_df.to_csv(output_path_preds, index=False)
    print(f" Predictions saved to: {output_path_preds}")

    valid_indices = [i for i, p in enumerate(predictions) if p != "error"]
    y_true = [true_labels[i] for i in valid_indices]
    y_pred = [predictions[i] for i in valid_indices]
    
    if len(y_true) > 0:
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        print("\n--- Evaluation Metrics ---")
        print(classification_report(y_true, y_pred, zero_division=0))
        save_report_to_csv(report_dict, csv_file, output_folder, model_name="Llama")
    else:
        print(f" No valid predictions to evaluate for {dialect}.")

if __name__ == "__main__":
    input_folder = 'input_data'
    output_folder = 'results/All_results/Llama'
    
    os.makedirs(output_folder, exist_ok=True)
    
    search_pattern = os.path.join(input_folder, '*.csv')
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f" No CSV files found in {input_folder}. Please make sure the folder exists and has data.")
    else:
        print(f"Found {len(csv_files)} datasets. Starting batch analysis on Ollama...")
        
        # 3. Loop through and process each one
        for file_path in csv_files:
            print(f"\n==============================================")
            print(f" Analyzing Dataset: {os.path.basename(file_path)}")
            print(f"==============================================")
            run_dialect_analysis(file_path, output_folder)
            
        print("\nAll datasets processed! You can now compare the Llama and RoBERTuito folders.")