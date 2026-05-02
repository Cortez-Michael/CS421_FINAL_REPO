import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud
from collections import Counter, defaultdict
import spacy

# Controls which specific dialects we parse through in the input data
TARGET_DIALECTS = ["venezuelan", "argentinian", "mexican", "spain"]

# Emotion color palette for word clouds and 
EMOTION_COLORS = {
    "anger":    ["#FF4500", "#FF6347", "#DC143C", "#B22222", "#FF0000"],
    "disgust":  ["#556B2F", "#6B8E23", "#808000", "#9ACD32", "#8FBC8F"],
    "fear":     ["#4B0082", "#6A0DAD", "#800080", "#9932CC", "#BA55D3"],
    "joy":      ["#FFD700", "#FFA500", "#FFEC3D", "#FFB347", "#FFFACD"],
    "sadness":  ["#4169E1", "#1E90FF", "#6495ED", "#87CEEB", "#B0C4DE"],
    "surprise": ["#FF69B4", "#FF1493", "#FF6EB4", "#FFB6C1", "#FFC0CB"],
    "other":    ["#708090", "#A9A9A9", "#C0C0C0", "#D3D3D3", "#DCDCDC"],
}


# spaCy filter 
# Load the Spanish NLP model. We're disabling the parser and NER pipelines 
# here to speed things up since we only need the basic text processing.
# python -m spacy download es_core_news_sm
try:
    nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])
    SPACY_LOADED = True
    print("[SUCCESS] spaCy Spanish model loaded.")
except OSError:
    print("[EROR] spaCy model not found. Run: python -m spacy download es_core_news_sm")
    print("    Falling back to basic filtering.")
    SPACY_LOADED = False

# If token is a meaningful soanish word -> reutnrs True
# This means 
# - not a subword fragment
# - at least 3 chars long
# - exists in spaCy's spanish vocab
# - alphabetical only
def is_valid_word(token: str) -> bool:
   
    token = token.strip()

    if token.startswith("##") or token.startswith("▁"):
        return False

    if not token.isalpha() or len(token) < 3:
        return False

    if SPACY_LOADED:
        doc = nlp(token)
        if not doc:
            return False
        tok = doc[0]
        # Drop stopwords and tokens not in vocabulary
        if tok.is_stop:
            return False
        if not nlp.vocab[token].is_oov is False:
            # Word exists in vocab
            pass
    return True

# filters keywords from raw list
def filter_keywords(raw_list: list) -> list:
    filtered_keywords = []
    for w in raw_list:
        if is_valid_word(w):
            filter_keywords.append(w.lower())
    return filtered_keywords


# parse for key words
def parse_keywords(keyword_str):
    """Parses the comma-separated keyword string into a clean list."""
    if pd.isna(keyword_str) or keyword_str == "":
        return []
    clean_str = str(keyword_str).replace('"', '').replace("'", "")
    return [word.strip() for word in clean_str.split(',') if word.strip()]

# Help create directory and save file
def save_plot(filename, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    print(f"[SUCCESS] Saved: {path}")


# Horizontal bar chart -> one model + one dialect
# reads a single predictions csv and grpahs top key word frequencies.
def plot_keyword_bars_per_dialect(prediction_csv: str, top_n: int = 15):
   
    df = pd.read_csv(prediction_csv)

    if 'top_keywords' not in df.columns or 'emotion' not in df.columns:
        print(f"[ERROR] Missing required columns in {prediction_csv}")
        return

    # Collect filtered keywords grouped by emotion
    emotion_keywords: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        raw   = parse_keywords(row['top_keywords'])
        clean = filter_keywords(raw)
        emotion = str(row['emotion']).lower()
        emotion_keywords[emotion].extend(clean)

    emotions_present = []
    for e in EMOTION_COLORS:
        if emotion_keywords.get(e):
            emotions_present.append(e)
    
    if not emotions_present:
        print(f"[ERROR] No valid keywords found in {os.path.basename(prediction_csv)}")
        return

    model_name = os.path.basename(os.path.dirname(prediction_csv))
    dialect    = os.path.basename(prediction_csv).split('_')[0]
    if dialect == "spanish":
        dialect = "spain"

    n    = len(emotions_present)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4), squeeze=False)
    fig.patch.set_facecolor("#0D0D0D")
    axes = axes.flatten()

    for i, emotion in enumerate(emotions_present):
        ax  = axes[i]
        ax.set_facecolor("#111111")

        freq    = Counter(emotion_keywords[emotion])
        top     = freq.most_common(top_n)
        if not top:
            ax.axis('off')
            continue

        words  = [w for w, _ in reversed(top)]
        counts = [c for _, c in reversed(top)]
        color  = EMOTION_COLORS.get(emotion, ["#AAAAAA"])[0]

        bars = ax.barh(words, counts, color=color, alpha=0.85, height=0.6)

        # value labels on each bar
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(count),
                va='center', ha='left',
                color='white', fontsize=8, fontfamily='monospace'
            )

        ax.set_title(emotion.upper(), color=color, fontsize=12,
                     fontweight='bold', fontfamily='monospace', pad=6)
        ax.tick_params(colors='white', labelsize=8)
        ax.xaxis.label.set_color('white')
        ax.set_xlabel('Frequency', color='#AAAAAA', fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.tick_params(axis='y', labelcolor='white')
        ax.tick_params(axis='x', labelcolor='#888888')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f"Top Keywords by Emotion  ·  {model_name}  ·  {dialect.capitalize()}",
        color='white', fontsize=14, fontweight='bold',
        fontfamily='monospace', y=1.01
    )
    plt.tight_layout()
    save_plot(f"{dialect}_{model_name}_keyword_bars.png", "./plots/keywords_full_words")
    plt.close()



# one model + ALL its dialects
# Aggregates keywords across every dialect for a given model and produces
# one horizontal bar chart per emotion showing cross-dialect keyword frequency.
def plot_keyword_bars_all_dialects(model_name: str, top_n: int = 15):
    
    pattern   = "../results/All_results/*/*_predictions.csv"
    pred_files = glob.glob(pattern)

    emotion_keywords: dict[str, list[str]] = defaultdict(list)

    for file in pred_files:
        if model_name.lower() not in file.lower():
            continue
        dialect = os.path.basename(file).split('_')[0]
        if dialect == "spanish": dialect = "spain"
        if dialect not in TARGET_DIALECTS: continue

        df = pd.read_csv(file)
        
        emo_col = 'predicted_emotion' if 'predicted_emotion' in df.columns else 'emotion'
        
        if 'top_keywords' not in df.columns or emo_col not in df.columns:
            print(f"[ERROR] Warning: Missing expected columns in {file}. Found: {df.columns.tolist()}")
            continue

        for _, row in df.iterrows():
            raw   = parse_keywords(row['top_keywords'])
            clean = filter_keywords(raw)

            emotion = str(row[emo_col]).lower() 
            emotion_keywords[emotion].extend(clean)

    emotions_present = [e for e in EMOTION_COLORS if emotion_keywords.get(e)]
    if not emotions_present:
        print(f"[ERROR] No keywords found for model: {model_name}")
        return

    n    = len(emotions_present)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4), squeeze=False)
    fig.patch.set_facecolor("#0D0D0D")
    axes = axes.flatten()

    for i, emotion in enumerate(emotions_present):
        ax  = axes[i]
        ax.set_facecolor("#111111")

        freq = Counter(emotion_keywords[emotion])
        top  = freq.most_common(top_n)
        if not top:
            ax.axis('off')
            continue

        words  = [w for w, _ in reversed(top)]
        counts = [c for _, c in reversed(top)]
        color  = EMOTION_COLORS.get(emotion, ["#AAAAAA"])[0]

        bars = ax.barh(words, counts, color=color, alpha=0.85, height=0.6)

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(count),
                va='center', ha='left',
                color='white', fontsize=8, fontfamily='monospace'
            )

        ax.set_title(emotion.upper(), color=color, fontsize=12,
                     fontweight='bold', fontfamily='monospace', pad=6)
        ax.tick_params(colors='white', labelsize=8)
        ax.set_xlabel('Frequency', color='#AAAAAA', fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.tick_params(axis='y', labelcolor='white')
        ax.tick_params(axis='x', labelcolor='#888888')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f"Top Keywords by Emotion  ·  {model_name}  ·  All Dialects",
        color='white', fontsize=14, fontweight='bold',
        fontfamily='monospace', y=1.01
    )
    plt.tight_layout()
    save_plot(f"combined_{model_name}_keyword_bars.png", "./plots/keywords_full_words")
    plt.close()


# Generates both bar chart versions for every model/dialect combination and one combined-dialect chart per model.
def generate_all_keyword_bar_plots():

    pred_files = glob.glob("../results/All_results/*/*_predictions.csv")

    # Version 1 — per dialect
    for file in pred_files:
        dialect = os.path.basename(file).split('_')[0]
        if dialect == "spanish":
            dialect = "spain"
        if dialect in TARGET_DIALECTS:
            print(f"Generating keyword bar chart for: {file}")
            plot_keyword_bars_per_dialect(file)

    for model in ["RoBERTuito", "BETO", "XLM-RoBERTa"]:
        print(f"Generating combined keyword bar chart for: {model}")
        plot_keyword_bars_all_dialects(model_name=model)

# Scanns all results, tracks raw tokens, plots side by side comparison of raw/fragmented keywords
def generate_all_comparative_keyword_plots():
    
    agg_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    pred_files = glob.glob("../results/All_results/*/*_predictions.csv")

    for file in pred_files:
        model = os.path.basename(os.path.dirname(file))
        filename = os.path.basename(file)
        dialect = filename.split('_')[0]
        if dialect == "spanish": dialect = "spain"
        if dialect not in TARGET_DIALECTS: continue

        df = pd.read_csv(file)
        emo_col = 'predicted_emotion' if 'predicted_emotion' in df.columns else 'emotion'

        for _, row in df.iterrows():
            emo = str(row[emo_col]).lower()
            # Raw extraction - skip filter_keywords 
            # to see the dirty sub-word fragments
            raw_kws = parse_keywords(row['top_keywords'])
            agg_data[dialect][emo][model].extend(raw_kws)

    # generate plots
    for dialect, emotions in agg_data.items():
        for emotion, models in emotions.items():
            plot_comparative_keyword_grid(dialect, emotion, models)

# Creates a grid of bar charts, one for each model, comparing raw keywords for a specific dialect and emotion.
def plot_comparative_keyword_grid(dialect, emotion, model_data, top_n=10):

    models = sorted(model_data.keys())
    n = len(models)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4), squeeze=False)
    fig.patch.set_facecolor("#0D0D0D")
    axes = axes.flatten()

    for i, model in enumerate(models):
        ax = axes[i]
        ax.set_facecolor("#111111")
        
        # Get raw counts
        freq = Counter(model_data[model])
        top = freq.most_common(top_n)
        
        words = [w for w, _ in reversed(top)]
        counts = [c for _, c in reversed(top)]
        
        # Consistent color based on the emotion
        color = EMOTION_COLORS.get(emotion, ["#AAAAAA"])[0]
        
        bars = ax.barh(words, counts, color=color, alpha=0.85, height=0.6)
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(count), va='center', ha='left', color='white', 
                    fontsize=8, fontfamily='monospace')

        ax.set_title(model, color='white', fontsize=11, fontweight='bold', pad=6)
        ax.tick_params(colors='white', labelsize=8)
        ax.set_xlabel('Frequency', color='#888888', fontsize=8)
        
        for spine in ax.spines.values(): spine.set_edgecolor('#333333')

    for j in range(i + 1, len(axes)): axes[j].axis('off')

    fig.suptitle(f"Keyword Interpretation: {emotion.upper()} ({dialect.upper()})", 
                 color='white', fontsize=14, fontweight='bold', y=1.02)
    
    os.makedirs("./plots/keywords", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"./plots/keywords/{dialect}_{emotion}_comparison.png", 
                facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[SUCCESS] Created comparison: {dialect}_{emotion}_comparison.png")

# Scans all results, adds all full words by dialect, emotion, model and plots a side by side comparison
def generate_comparative_full_word_plots():

    agg_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    pred_files = glob.glob("../results/All_results/*/*_predictions.csv")

    for file in pred_files:
        model = os.path.basename(os.path.dirname(file))
        filename = os.path.basename(file)
        dialect = filename.split('_')[0]
        if dialect == "spanish": 
            
            dialect = "spain"

        if dialect not in TARGET_DIALECTS: 
            continue

        df = pd.read_csv(file)
        emo_col = 'predicted_emotion' if 'predicted_emotion' in df.columns else 'emotion'

        for _, row in df.iterrows():
            emo = str(row[emo_col]).lower()
            
            raw_kws = parse_keywords(row['top_keywords'])
            clean_kws = filter_keywords(raw_kws) 
            
            agg_data[dialect][emo][model].extend(clean_kws)

    for dialect, emotions in agg_data.items():
        for emotion, models in emotions.items():
            plot_comparative_keyword_grid_full_words(dialect, emotion, models)

# Helper to plot the full-word frequency grid.
def plot_comparative_keyword_grid_full_words(dialect, emotion, model_data, top_n=10):
    
    models = sorted(model_data.keys())
    n = len(models)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4), squeeze=False)
    fig.patch.set_facecolor("#0D0D0D")
    axes = axes.flatten()

    for i, model in enumerate(models):
        ax = axes[i]
        ax.set_facecolor("#111111")
        
        freq = Counter(model_data[model])
        top = freq.most_common(top_n)
        
        words = [w for w, _ in reversed(top)]
        counts = [c for _, c in reversed(top)]
        color = EMOTION_COLORS.get(emotion, ["#AAAAAA"])[0]
        
        bars = ax.barh(words, counts, color=color, alpha=0.85, height=0.6)
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(count), va='center', ha='left', color='white', 
                    fontsize=8, fontfamily='monospace')

        ax.set_title(model, color='white', fontsize=11, fontweight='bold', pad=6)
        ax.tick_params(colors='white', labelsize=8)
        ax.set_xlabel('Frequency', color='#888888', fontsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor('#333333')

    for j in range(i + 1, len(axes)): axes[j].axis('off')

    fig.suptitle(f"Top Full-Word Keywords: {emotion.upper()} ({dialect.upper()})", 
                 color='white', fontsize=14, fontweight='bold', y=1.02)
    
    output_dir = "./plots/keywords_full_words"
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dialect}_{emotion}_full_word_comparison.png", 
                facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[SUCCESS] Created comparison: {dialect}_{emotion}_full_word_comparison.png")

if __name__ == "__main__":
    generate_all_keyword_bar_plots()
    # generate_all_comparative_keyword_plots()
    # generate_comparative_full_word_plots()