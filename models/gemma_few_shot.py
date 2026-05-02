import os
import re
import json
import requests
import pandas as pd
from sklearn.metrics import classification_report

# ==========================================
# 1. Configuration
# ==========================================
OLLAMA_MODEL = "gemma4:e4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

def save_report_to_csv(report_dict, input_path, output_folder, model_name="Gemma"):
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
    """Processes a single CSV using Gemma and generates the prediction and report files."""
    dialect = os.path.splitext(os.path.basename(csv_file))[0]
    output_path_preds = os.path.join(output_folder, f"{dialect}_Gemma_predictions.csv")

    # Load dataset
    df = pd.read_csv(csv_file, on_bad_lines='skip', engine='python')
    
    # --- THE FIX: KILL THE GHOST ROWS ---
    df = df.dropna(subset=['tweet', 'emotion'])
    
    # Auto-skip if the dataset is completely empty
    if len(df) == 0:
        print(f"[!] {dialect}.csv is completely empty after cleaning. Skipping this file...")
        return
    # ------------------------------------
    
    predictions = []
    keywords_list = []
    reasoning_list = []
    true_labels = []


#======SPAIN SPANISH==========
# Example 1:
# Text: "¡Qué guay el concierto de anoche, me lo pasé pipa! Vale la pena repetir. 🎸🔥"
# {{
#     "emotion": "joy",
#     "keywords": ["guay", "pasé pipa", "vale la pena"],
#     "context": "'Guay' (cool) and 'pasarlo pipa' (to have a great time) are quintessential Spain slang for enjoyment and excitement."
# }}

# Example 2:
# Text: "Hostia, qué fuerte lo que ha pasado en el centro. Me he quedado de piedra al verlo. 😱"
# {{
#     "emotion": "surprise",
#     "keywords": ["hostia", "qué fuerte", "quedado de piedra"],
#     "context": "'Hostia' is used here as an intensifier of shock, while 'quedarse de piedra' (to be stunned) explicitly indicates total disbelief."
# }}

# Example 3:
# Text: "Estoy flipando con el servicio de este sitio. Menuda mala leche tienen los camareros, de verdad. 😡"
# {{
#     "emotion": "anger",
#     "keywords": ["flipando", "mala leche", "menuda"],
#     "context": "While 'flipar' can mean surprise, 'mala leche' (bad mood/malice) clarifies the emotion as indignation and anger toward the service."
# }}

# Example 4:
# Text: "Mañana tengo el examen y estoy hasta las narices de estudiar. No me entra nada en el coco. 😫"
# {{
#     "emotion": "sadness",
#     "keywords": ["hasta las narices", "estudiar", "coco"],
#     "context": "'Estar hasta las narices' is a Spain-specific way to say 'I am fed up.' It expresses the exhaustion and low mood associated with burnout."
# }}

# Example 5:
# Text: "He quedado con los chavales en el parque a las cinco para dar una vuelta. 🌳"
# {{
#     "emotion": "others",
#     "keywords": ["chavales", "quedado", "dar una vuelta"],
#     "context": "A factual statement using common Spain terms like 'chavales' (kids/guys) and 'quedado' (met up), with no strong emotional markers."
# }}


#=======ARGENTINIAN==========
# Example 1:
# Text: "¡Naaa, me estás jodiendo! ¿En serio te cruzaste a Messi en Palermo? Me caigo de traste. 😲"
# {{
#     "emotion": "surprise",
#     "keywords": ["Naaa", "me estás jodiendo", "me caigo de traste"],
#     "context": "'Me estás jodiendo' (you're kidding me) and 'me caigo de traste' (falling on my butt) are heavily used in Argentina to express complete disbelief and shock."
# }}

# Example 2:
# Text: "Qué quilombo es el centro hoy. Estoy re caliente, siempre la misma historia con el tránsito. 🤬"
# {{
#     "emotion": "anger",
#     "keywords": ["quilombo", "re caliente", "tránsito"],
#     "context": "'Quilombo' means a mess or disaster, and 'estar re caliente' is the quintessential Argentinian phrase for being furious or extremely angry."
# }}

# Example 3:
# Text: "Ando medio bajón hoy. Extraño un montón las juntadas con los pibes y armar un asadito el finde. 😔"
# {{
#     "emotion": "sadness",
#     "keywords": ["bajón", "juntadas", "pibes"],
#     "context": "'Estar bajón' or 'un bajón' is the standard Argentinian slang for feeling depressed, down, or bummed out."
# }}

# Example 4:
# Text: "¡Qué golazo, papá! Somos campeones, qué locura hermosa, la puta madre. 🏆"
# {{
#     "emotion": "joy",
#     "keywords": ["golazo", "papá", "locura hermosa", "la puta madre"],
#     "context": "'Papá' is used as an affectionate hype term, and 'la puta madre', while a swear word, is frequently used as a massive intensifier for extreme joy in Argentina."
# }}

# Example 5:
# Text: "Che, acordate que mañana a las ocho paso por tu casa a buscar los apuntes de la facu. 📚"
# {{
#     "emotion": "others",
#     "keywords": ["Che", "acordate", "facu"],
#     "context": "A neutral, logistical statement utilizing the iconic 'Che' (attention marker), 'acordate' (voseo form of remember), and 'facu' (short for university)."
# }}

#=====venezuelan==========
# Example 1:
# Text: "¡Qué de pinga estuvo la rumba de anoche! La pasamos brutal con todos los panas. 🍻"
# {{
#     "emotion": "joy",
#     "keywords": ["de pinga", "rumba", "brutal", "panas"],
#     "context": "'De pinga' and 'brutal' are very common Venezuelan expressions for something awesome or excellent, while 'panas' refers to good friends."
# }}

# Example 2:
# Text: "Qué arrechera tengo con los cortes de luz, siempre la misma vaina en esta ciudad. 🤬"
# {{
#     "emotion": "anger",
#     "keywords": ["arrechera", "vaina"],
#     "context": "In Venezuela, 'arrechera' explicitly means intense anger or fury. 'Vaina' is a universal filler word, used here to express deep frustration with a recurring annoyance."
# }}

# Example 3:
# Text: "Qué chimbo enterarme de esa noticia, me dejó el corazón arrugado todo el día. 😞"
# {{
#     "emotion": "sadness",
#     "keywords": ["chimbo", "corazón arrugado"],
#     "context": "'Chimbo' means sad, unfortunate, or lousy. 'Tener el corazón arrugado' (having a wrinkled heart) is a deeply regional metaphor for feeling nostalgic, sorrowful, or emotionally heavy."
# }}

# Example 4:
# Text: "¡Na guará, chamo! ¿Tú viste el tamaño de ese choque en la autopista? Qué locura. 😱"
# {{
#     "emotion": "surprise",
#     "keywords": ["Na guará", "chamo"],
#     "context": "'Na guará' is used to express great surprise or disbelief. 'Chamo' (dude/guy) used to address friends, acquaintances, or strangers."
# }}

# Example 5:
# Text: "Pana, avísame cuando llegues al edificio para bajarte las llaves de la reja. 🔑"
# {{
#     "emotion": "others",
#     "keywords": ["Pana", "bajarte", "reja"],
#     "context": "A neutral, logistical statement. 'Pana' is the quintessential Venezuelan word for friend or buddy, used here with no strong emotional polarity."
# }}

# Mexican
#Example 1:
# Text: "¡Qué chido estuvo el festival, güey! La neta me la pasé a toda madre. 🔥"
# {{
#     "emotion": "joy",
#     "keywords": ["chido", "güey", "La neta", "a toda madre"],
#     "context": "'Chido' indicates that something is excellent or cool. 'A toda madre' is a highly expressive phrase signifying a fantastic experience, frequently utilized alongside 'güey' (dude)."
# }}

# Example 2:
# Text: "Me caga que la gente maneje así. Estoy bien encabronado con este pinche tráfico. 🤬🚗"
# {{
#     "emotion": "anger",
#     "keywords": ["Me caga", "encabronado", "pinche"],
#     "context": "The phrase 'me caga' in this context expresses severe annoyance. 'Encabronado' indicates deep fury, and 'pinche' acts as a strong, colloquial intensifier."
# }}

# Example 3:
# Text: "Chale, ando bien agüitado porque no alcancé boletos para el cine. Qué mala onda. 😞"
# {{
#     "emotion": "sadness",
#     "keywords": ["Chale", "agüitado", "mala onda"],
#     "context": "'Chale' expresses resignation or disappointment, while 'agüitado' is the standard Mexican terminology for feeling saddened, depressed, or disheartened."
# }}

# Example 4:
# Text: "¡No manches! ¿Es neta que cancelaron las clases de mañana? Qué locura. 🤯"
# {{
#     "emotion": "surprise",
#     "keywords": ["No manches", "neta"],
#     "context": "'No manches' is a ubiquitous expression of shock or disbelief. '¿Es neta?' translates directly to 'Is it the truth/for real?', serving as a definitive marker of surprise."
# }}

# Example 5:
# Text: "Ahorita salgo de la chamba y paso al Oxxo a comprar algo, te aviso cuando llegue. 🥤"
# {{
#     "emotion": "others",
#     "keywords": ["Ahorita", "chamba", "Oxxo"],
#     "context": "A factual, everyday statement. 'Ahorita' denotes an unspecified near timeframe, 'chamba' means work or employment, and the mention of 'Oxxo' is a convenience store doesnt conveying strong emotional polarity."
# }}

    print(f"Starting inference on {len(df)} rows for {dialect}...")
    
    for index, row in df.iterrows():
        # Keep progress counter accurate to the actual row index
        if len(predictions) > 0 and len(predictions) % 100 == 0:
            print(f"Processed {len(predictions)}/{len(df)} rows...")
            
        raw_text = str(row['tweet']) if pd.notna(row['tweet']) else ""
        true_emotion = str(row['emotion']).lower().strip() if pd.notna(row['emotion']) else "unknown"
        
        # --- THE CLEANUP STEP ---
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
            "temperature": 0.1,
            "format": "json"
        }
        
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            raw_output = response.json().get('response', '{}')
            result_dict = json.loads(raw_output)
            
            pred_emotion = result_dict.get('emotion', 'error').lower().strip()
            
            # --- THE HALLUCINATION FILTER ---
            allowed_emotions = {"anger", "disgust", "fear", "joy", "sadness", "surprise", "others"}
            if pred_emotion not in allowed_emotions and pred_emotion != "error":
                pred_emotion = "others"
            # --------------------------------
                
            keys = ", ".join(result_dict.get('keywords', []))
            context = result_dict.get('context', 'No context')
        except Exception as e:
            pred_emotion, keys, context = "error", "error", f"Model Error: {str(e)}"
            
        predictions.append(pred_emotion)
        keywords_list.append(keys)
        reasoning_list.append(context)
        true_labels.append(true_emotion)

    results_df = pd.DataFrame({
        "model_type": "Gemma",
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
        save_report_to_csv(report_dict, csv_file, output_folder, model_name="Gemma")
    else:
        print(f"[!] No valid predictions to evaluate for {dialect}.")

if __name__ == "__main__":
    # --- EDIT THIS LINE ---
    input_file = 'input_data/mexican.csv' 
    output_folder = 'results/few-shot/Gemma'
    
    os.makedirs(output_folder, exist_ok=True)
    
    if not os.path.exists(input_file):
        print(f"[!] Error: Could not find '{input_file}'. Make sure the file name and path are correct.")
    else:
        print(f"\n==============================================")
        print(f" Analyzing Dataset: {os.path.basename(input_file)}")
        print(f"==============================================")
        run_dialect_analysis(input_file, output_folder)
        
        print("\n✅ Dataset processed! You can find the results in the Gemma folder.")