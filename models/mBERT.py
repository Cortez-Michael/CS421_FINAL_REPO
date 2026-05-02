import os
import glob
import torch
import shap
import string
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from huggingface_hub import login
from dotenv import load_dotenv

# setup to login and use hugging face model
load_dotenv()
my_hf_token = os.getenv("HF_TOKEN")
if my_hf_token:
    login(token=my_hf_token)
transformers.logging.set_verbosity(transformers.logging.ERROR)

# the emotions we actually care about
EMOTION_COLS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

MODEL_PATH = "AnasAlokla/multilingual_go_emotions"

# regions we are testing
TARGET_DIALECTS = ['mexican', 'spanish_spain', 'venezuelan', 'argentinian']

# set to true to run balance sample
USE_BALANCED_SAMPLE = False 
TARGET_PER_EMOTION = 10      

# maps the go emotions labels to the 7 emotions we look at
GO_EMOTIONS_MAP = {
    # joy
    'joy': 'joy', 'amusement': 'joy', 'approval': 'joy', 'caring': 'joy', 
    'desire': 'joy', 'excitement': 'joy', 'gratitude': 'joy', 'love': 'joy', 
    'optimism': 'joy', 'pride': 'joy', 'relief': 'joy',
    
    # anger
    'anger': 'anger', 'annoyance': 'anger', 'disapproval': 'anger',
    
    # sadness
    'sadness': 'sadness', 'disappointment': 'sadness', 'embarrassment': 'sadness', 
    'grief': 'sadness', 'remorse': 'sadness',
    
    # fear
    'fear': 'fear', 'nervousness': 'fear',
    
    # disgust & Surprise
    'disgust': 'disgust',
    'surprise': 'surprise', 'realization': 'surprise',
    
    # others
    'neutral': 'others', 'confusion': 'others', 'curiosity': 'others',
    'others': 'others', 'other': 'others'
}

# maps the labels, defaults to other if not found
def normalise_label(raw_label: str) -> str:
    return GO_EMOTIONS_MAP.get(raw_label.lower(), 'others')

# cleans some of the text to improve mBERT due to noise
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


# processes the csv and loads a balanced sample if wanted
def load_balanced_sample(csv_name, target_per_emotion=TARGET_PER_EMOTION):
    df = pd.read_csv(csv_name, on_bad_lines='skip', engine='python')

    # checks text column
    if 'text' in df.columns:
        text_col = 'text'
    else:
        text_col = 'tweet'

    # checks emotion column
    cols_to_check = [text_col]
    if 'emotion' in df.columns:
        cols_to_check.append('emotion')
        
    # drops rows with missing data
    df = df.dropna(subset=cols_to_check)
    
    # makes sure everything is a string
    df[text_col] = df[text_col].astype(str)
    if 'emotion' in df.columns:
        df['emotion'] = df['emotion'].astype(str)

    if 'emotion' in df.columns:
        df['_label'] = df['emotion'].str.lower().replace({'others': 'other'})
    else:
        df['_label'] = df[EMOTION_COLS].idxmax(axis=1)
    
    # if we arent balancing just return the whole thing
    if not USE_BALANCED_SAMPLE:
        print(f"Processing ALL {len(df)} lines from {os.path.basename(csv_name)}")
        return df, text_col

    # loops through and grabs the exact amount we need per emotion
    sampled_parts = []
    for emotion in EMOTION_COLS:
        subset = df[df['_label'] == emotion]
        available = len(subset)
        if available == 0:
            continue
        sampled_parts.append(subset.head(target_per_emotion))

    balanced_df = pd.concat(sampled_parts).reset_index(drop=True)
    return balanced_df, text_col


def mbert_preds(df, text_col, csv_name):
    # dialect name for file
    dialect = csv_name.split("/")[-1].split(".")[0]
    if dialect == "spanish_spain":
        dialect = "spain"
    
    os.makedirs("results/mBERT/", exist_ok=True)
    output_path = f"results/mBERT/{dialect}_mBERT_predictions.csv"

    if os.path.exists(output_path):
        os.remove(output_path)

    predictions = []

    print(f"Starting mBERT inference on {len(df)} rows...")
    
    # loop through prediction
    for idx, row in df.iterrows():
        sentence = row[text_col]
        true_label = normalise_label(row['_label']) 
        
        # run model
        result = run_analysis(sentence)
        result['true_emotion'] = true_label 
        
        write_result_to_csv(result, output_path)
        # prediction for math later
        predictions.append(result['emotion'])

    return predictions

def evaluate_results(y_true, y_pred):
    print("\n--- mBERT Evaluation Metrics ---")
    print(classification_report(y_true, y_pred, zero_division=0))
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


mbert_model = None
mbert_tokenizer = None
mbert_config = None

# gets and sets model
def _get_model():
    global mbert_model, mbert_tokenizer, mbert_config
    if mbert_model is None:
        mbert_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        mbert_config = AutoConfig.from_pretrained(MODEL_PATH)
        mbert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, use_safetensors=True)
        
        print("Model successfully loaded onto: CPU")
        
    return mbert_model, mbert_tokenizer, mbert_config

# runs the actual analysis for one sentence and gets the emotion
def run_analysis(text):
    # gets model and cleans text
    model, tokenizer, config = _get_model()
    processed_text = preprocess(text)
    
    # turns text into math for the model
    encoded_input = tokenizer(processed_text, return_tensors='pt')

    with torch.no_grad():
        output = model(**encoded_input)

    scores  = output[0][0].detach().numpy()
    scores  = softmax(scores)

    # gets winning emotion
    pred_idx = int(np.argmax(scores))
    raw_label = config.id2label[pred_idx]
    confidence = float(np.round(scores[pred_idx], 4))
    
    # maps to emotion
    emotion = normalise_label(raw_label)

    # keyword
    keywords = get_important_keywords(text)

    return {
        "model_type":   "mBERT",
        "text":         text,
        "emotion":      emotion,
        "confidence":   confidence,
        "top_keywords": keywords,
    }


def get_important_keywords(text):
    model, tokenizer, config = _get_model()

    # removes noise
    clean_text = text.replace("HASHTAG", "").replace("USER", "").replace("URL", "")
    clean_text = " ".join(clean_text.split())
    
    if not clean_text:
        return []

    processed_text = preprocess(clean_text)

    # helper function for shap to predict
    def predict(texts):
        inputs = tokenizer(
            [preprocess(t) for t in texts.tolist()],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()

    # runs shap
    explainer   = shap.Explainer(predict, tokenizer)
    shap_values = explainer([processed_text])

    # actual emotion picked
    encoded_input = tokenizer(processed_text, return_tensors='pt')
    
    with torch.no_grad():
        output = model(**encoded_input)
    
    scores   = softmax(output[0][0].detach().numpy())
    pred_idx = int(np.argmax(scores))

    word_importance = []
    # common spanish words we want to ignore
    stop_words = {"a", "de", "el", "la", "en", "que", "y", "los", "las", "un", "una", "se", "por", "con", "es", "me", "lo"}
    
    # finds most important words
    for i, word in enumerate(shap_values.data[0]):
        importance = shap_values.values[0][i][pred_idx]
        
        if importance > 0:
            clean_word = word.replace(' ', '').strip()
            is_punctuation = all(char in string.punctuation for char in clean_word)
            
            if clean_word and not is_punctuation and clean_word.lower() not in stop_words:
                word_importance.append((clean_word, importance))

    # most to least important words
    ranked = sorted(word_importance, key=lambda x: x[1], reverse=True)
    return [word for word, score in ranked]
    
# final report for csv
def save_report_to_csv(report_dict, input_path, model_name="mBERT"):
    rows = []
    aggregate_keys = {"accuracy", "macro avg", "weighted avg"}

    # loop through emotion
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

    # loop avg
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

    df = pd.DataFrame(rows)
    
    # accuracy
    if "accuracy" in report_dict:
        acc_row = pd.DataFrame([{
            "model":     model_name,
            "class":     "accuracy",
            "precision": "",
            "recall":    "",
            "f1_score":  round(report_dict["accuracy"], 4),
            "support":   "",
            "row_type":  "aggregate",
        }])
        
        df['precision'] = df['precision'].astype(object)
        df['recall'] = df['recall'].astype(object)
        df['support'] = df['support'].astype(object)
        
        df = pd.concat([df, acc_row], ignore_index=True)

    # saves the final math report
    dialect = input_path.split("/")[-1].split(".")[0]
    if dialect == "spanish_spain":
        dialect = "spain"
    output_path = f"results/mBERT/{dialect}_{model_name}_classification_report.csv"
    df.to_csv(output_path, index=False)
    print(f" Report saved to: {output_path}")
    return df

def write_result_to_csv(result_dict, output_path):
    # dataframe
    df = pd.DataFrame([{
        "model_type":        result_dict["model_type"],
        "text":              result_dict["text"],
        "true_emotion":      result_dict["true_emotion"],
        "predicted_emotion": result_dict["emotion"], 
        "confidence":        result_dict["confidence"],
        "top_keywords":      ", ".join(result_dict["top_keywords"]) if result_dict["top_keywords"] else ""
    }])

    # adds it to existing file or makes a new one
    if os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_path, mode="w", header=True, index=False)

def run_dialect_analysis(csv_file):
    df, text_col = load_balanced_sample(csv_file)
    
    gold_labels = [normalise_label(label) for label in df['_label'].tolist()]

    model_preds = mbert_preds(df, text_col, csv_file)

    report = evaluate_results(gold_labels, model_preds)
    save_report_to_csv(report, csv_file, model_name="mBERT")


if __name__ == "__main__":
    input_dir = 'input_data'

    print(f"Targeting {len(TARGET_DIALECTS)} dialects: {TARGET_DIALECTS}")

    # lopps throgu all files in input directory
    for dialect in TARGET_DIALECTS:
        csv_path = os.path.join(input_dir, f"{dialect}.csv")
        
        if os.path.exists(csv_path):
            print(f"\n--- Starting analysis for: {dialect} ---")
            try:
                run_dialect_analysis(csv_path)
            except Exception as e:
                print(f" Failed to process {dialect}: {e}")
        else:
            print(f" Warning: File not found at {csv_path}. Skipping.")