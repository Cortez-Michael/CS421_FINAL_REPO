import os
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

MODEL_PATH  = "tabularisai/multilingual-emotion-classification"
MODEL_NAME  = "XLM-RoBERTa" 

# regions we are testing
TARGET_DIALECTS    = ['mexican', 'spanish_spain', 'venezuelan', 'argentinian']
# settings to balance the dataset if needed
USE_BALANCED_SAMPLE = False
TARGET_PER_EMOTION  = 10

# maps the labels to the 7 emotions we look at
XLM_LABEL_MAP = {
    # anger
    "anger":       "anger",
    "frustration": "anger",
    # disgust & Surprise
    "disgust":     "disgust",
    # fear
    "fear":        "fear",
    # joy
    "joy":         "joy",
    "gratitude":   "joy",  
    "love":        "joy",
    # sadness
    "sadness":     "sadness",
    "surprise":    "surprise",
    
    # others
    "neutral":     "others",
    "others":      "others",
    "other":       "others",
    "contempt":    "others"           
}

# maps the labels, defaults to itself if not found
def normalise_label(raw_label: str) -> str:
    return XLM_LABEL_MAP.get(raw_label.lower(), raw_label.lower())

# cleans some of the text to improve the model due to noise
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http'  if t.startswith('http')                else t
        new_text.append(t)
    return " ".join(new_text)


# processes the csv and loads a balanced sample if wanted
def load_balanced_sample(csv_name, target_per_emotion=TARGET_PER_EMOTION):
    df = pd.read_csv(csv_name, on_bad_lines='skip', engine='python')

    # checks text column
    text_col = 'text' if 'text' in df.columns else 'tweet'

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
        print(f" Processing ALL {len(df)} lines from {os.path.basename(csv_name)}")
        return df, text_col

    # loops through and grabs the exact amount we need per emotion
    sampled_parts = []
    for emotion in EMOTION_COLS:
        subset = df[df['_label'] == emotion]
        if len(subset) == 0:
            continue
        sampled_parts.append(subset.head(target_per_emotion))

    balanced_df = pd.concat(sampled_parts).reset_index(drop=True)
    return balanced_df, text_col


_model     = None
_tokenizer = None
_config    = None

# loads model once so it doesnt slow down the loop
def _get_model():
    global _model, _tokenizer, _config
    if _model is None:
        print(f"Loading model: {MODEL_PATH}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _config    = AutoConfig.from_pretrained(MODEL_PATH)
        _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()
    return _model, _tokenizer, _config


# runs the actual analysis for one sentence and gets the emotion
def run_analysis(text):
    # gets model and cleans text
    model, tokenizer, config = _get_model()

    processed_text = preprocess(text)

    # turns text into math for the model
    encoded_input = tokenizer(
        processed_text,
        return_tensors='pt',
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        output = model(**encoded_input)

    scores   = softmax(output[0][0].detach().numpy())
    
    # gets winning emotion
    pred_idx = int(np.argmax(scores))
    raw_label  = config.id2label[pred_idx]
    
    # maps to emotion
    emotion    = normalise_label(raw_label)
    confidence = float(np.round(scores[pred_idx], 4))

    # gets shap keywords
    keywords = get_important_keywords(text)

    return {
        "model_type":   MODEL_NAME,
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
            max_length=512,          
        )
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()

    # runs shap
    explainer   = shap.Explainer(predict, tokenizer)
    shap_values = explainer([processed_text])

    # actual emotion picked
    encoded_input = tokenizer(
        processed_text, return_tensors='pt', truncation=True, max_length=512
    )
    with torch.no_grad():
        output = model(**encoded_input)
    scores   = softmax(output[0][0].detach().numpy())
    pred_idx = int(np.argmax(scores))

    # common spanish words we want to ignore
    stop_words = {
        "a", "de", "el", "la", "en", "que", "y", "los", "las",
        "un", "una", "se", "por", "con", "es", "me", "lo",
    }

    word_importance = []
    # find the most important words
    for word, importance in zip(shap_values.data[0], shap_values.values[0]):
        importance = importance[pred_idx]
        if importance > 0:
            clean_word = word.replace('▁', '').replace('##', '').strip()
            is_punct   = all(c in string.punctuation for c in clean_word)
            
            if clean_word and not is_punct and clean_word.lower() not in stop_words:
                word_importance.append((clean_word, importance))

    # most to least important words
    ranked = sorted(word_importance, key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked]


def xlm_roberta_preds(df, text_col, csv_name):
    # gets the dialect name from the file
    dialect = csv_name.split("/")[-1].split(".")[0]
    if dialect == "spanish_spain":
        dialect = "spain"

    # creates folder for results if missing
    out_dir = f"results/{MODEL_NAME}/"
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{out_dir}{dialect}_{MODEL_NAME}_predictions.csv"

    # deletes old file so we dont double up on data
    if os.path.exists(output_path):
        os.remove(output_path)

    predictions = []
    print(f"Starting {MODEL_NAME} inference on {len(df)} rows...")

    # loop through prediction
    for _, row in df.iterrows():
        sentence   = row[text_col]
        true_label = normalise_label(row['_label'])

        # run model
        result = run_analysis(sentence)
        result['true_emotion'] = true_label
        
        # writes this row straight to the csv
        write_result_to_csv(result, output_path)
        # prediction for math later
        predictions.append(result['emotion'])

    return predictions

# calculates precision recall f1 and accuracy
def evaluate_results(y_true, y_pred):
    print(f"\n--- {MODEL_NAME} Evaluation Metrics ---")
    print(classification_report(y_true, y_pred, zero_division=0))
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


# final report for csv
def save_report_to_csv(report_dict, input_path, model_name=MODEL_NAME):
    # flattens the math into rows for a csv
    rows = []
    aggregate_keys = {"accuracy", "macro avg", "weighted avg"}

    # loops through each emotion
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

    # loops through the averages
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

    # handles accuracy row specifically
    if "accuracy" in report_dict:
        df['precision'] = df['precision'].astype(object)
        df['recall']    = df['recall'].astype(object)
        df['support']   = df['support'].astype(object)
        acc_row = pd.DataFrame([{
            "model":     model_name,
            "class":     "accuracy",
            "precision": "",
            "recall":    "",
            "f1_score":  round(report_dict["accuracy"], 4),
            "support":   "",
            "row_type":  "aggregate",
        }])
        df = pd.concat([df, acc_row], ignore_index=True)

    dialect = input_path.split("/")[-1].split(".")[0]
    if dialect == "spanish_spain":
        dialect = "spain"

    out_dir = f"results/{MODEL_NAME}/"
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{out_dir}{dialect}_{model_name}_classification_report.csv"
    
    # saves the final math report
    df.to_csv(output_path, index=False)
    print(f"Report saved to: {output_path}")
    return df


def write_result_to_csv(result_dict, output_path):
    # turns the single prediction into a dataframe row
    df = pd.DataFrame([{
        "model_type":        result_dict["model_type"],
        "text":              result_dict["text"],
        "true_emotion":      result_dict["true_emotion"],
        "predicted_emotion": result_dict["emotion"],
        "confidence":        result_dict["confidence"],
        "top_keywords":      ", ".join(result_dict["top_keywords"]) if result_dict["top_keywords"] else "",
    }])

    # adds it to existing file or makes a new one
    if os.path.exists(output_path):
        df.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df.to_csv(output_path, mode="w", header=True, index=False)


# main function that runs the whole process for one dialect
def run_dialect_analysis(csv_file):
    df, text_col = load_balanced_sample(csv_file)
    gold_labels  = [normalise_label(label) for label in df['_label'].tolist()]
    model_preds  = xlm_roberta_preds(df, text_col, csv_file)
    report       = evaluate_results(gold_labels, model_preds)
    save_report_to_csv(report, csv_file)


if __name__ == "__main__":
    input_dir = 'input_data'
    print(f"Model   : {MODEL_PATH}")
    print(f"Dialects: {TARGET_DIALECTS}")

    # loops through all our dialects and runs them one by one
    for dialect in TARGET_DIALECTS:
        csv_path = os.path.join(input_dir, f"{dialect}.csv")
        
        if os.path.exists(csv_path):
            print(f"\n--- Starting analysis for: {dialect} ---")
            try:
                run_dialect_analysis(csv_path)
            except Exception as e:
                print(f"Failed to process {dialect}: {e}")
        else:
            print(f"Warning: File not found at {csv_path}. Skipping.")