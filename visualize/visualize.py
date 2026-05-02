import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import glob
import re
from collections import defaultdict, Counter
import numpy as np

# Controls which specific dialects we parse through in the input data 
TARGET_DIALECTS = ['mexican', 'spain', 'venezuelan', 'argentinian']

def get_dialect_from_filename(filename):
    dialect = filename.split('/')[-1].split('_')[0]

    if dialect == "spanish":
        return "spain" 
    return dialect


# Gets all Classifications from CSV, plots F1 scores per class dialect
def plot_f1_by_dialect(results_directory):
    
    all_files = glob.glob(f"{results_directory}/*.csv")
    df_list = []
    
    for filename in all_files:
        temp_df = pd.read_csv(filename)

        dialect = filename.split('/')[-1].split('_')[0] 
        temp_df['dialect'] = dialect
        df_list.append(temp_df[temp_df['row_type'] == 'per_class'])
    
    df = pd.concat(df_list)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='class', y='f1_score', hue='dialect')
    plt.title('F1-Score Performance per Emotion and Dialect')
    plt.ylabel('F1-Score')
    plt.ylim(0, 1.0)
    plt.legend(title='Dialect', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# Creates a heatmap of a specific metric, filtered to only oinclude the 7 target emotions
# 7 emotions = 6 ekman emotions + "other" emother
def plot_metric_heatmap(csv_path, metric='f1_score'):
    """
    Creates a heatmap of a specific metric (e.g., F1) across classes,
    filtered to only include target emotions.
    """
    df = pd.read_csv(csv_path)
    
    target_emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'others', 'other']
    
    df = df[(df['row_type'] == 'per_class') & (df['class'].isin(target_emotions))]
    
    pivot_df = df.pivot(index='class', columns='model', values=metric)
    
    dialect = csv_path.split("/")[-1].split("_")[0]
    if dialect == "spanish":
        dialect = "spain"
    model = csv_path.split("/")[-2]
    plot_file_name = f"{dialect}_{model}_heatmap.png"

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
    plt.title(f'F1 Scores of {dialect} + {model} Emotion Analysis')

    save_plot(plot_file_name, "./plots/heatmaps")

# Runs previouse plot_metric_heatmap() function for all classification reports in results folder
# for target dialects
def generate_all_heatmaps():

    report_files = glob.glob("../results/All_results/*/*classification_report*.csv")

    for file_path in report_files:

        dialect = get_dialect_from_filename(file_path)
        
        if dialect in TARGET_DIALECTS:
            print(f"Generating heatmap for: {dialect} ({file_path})")
            plot_metric_heatmap(file_path, metric='f1_score')
        else:
            continue



# Creates a single grouped bar chart: Models on X-axis, F1 on Y-axis, Emotions as colored bars. Filters by TARGET_DIALECTS and applies a custom model order.
def plot_comparative_f1_bars_single():

    all_files = glob.glob("../results/All_results/*/*classification_report*.csv")
    df_list = []

    target_emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'others']

    for file in all_files:
        # 1. Filter by Dialect
        dialect = get_dialect_from_filename(file)
        if dialect not in TARGET_DIALECTS:
            continue
            
        folder_name = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file)
        
        # 2. Filter for per_class metrics and target emotions
        df = df[df['row_type'] == 'per_class'].copy()
        df = df[df['class'].isin(target_emotions + ['other'])]
        
        df['model'] = folder_name
        df_list.append(df)

    if not df_list:
        print("No valid data found after filtering for target dialects/emotions.")
        return

    full_df = pd.concat(df_list)

    # 3. Calculate the average across the filtered dialects
    avg_df = full_df.groupby(['model', 'class'])['f1_score'].mean().reset_index()

    # --- REORDERING LOGIC ---
    # Apply the same order as your macro F1 graph
    custom_order = ['mBERT', 'XLM-RoBERTa', 'Gemma', 'Gemma_Few_Shot', 'Llama', 'Llama_Few_Shot']
    avg_df['model'] = pd.Categorical(avg_df['model'], categories=custom_order, ordered=True)
    avg_df = avg_df.sort_values('model')
    # ------------------------

    plt.figure(figsize=(14, 7))
    
    # Use 'order' argument in sns.barplot to ensure the categorical order is respected
    sns.barplot(
        data=avg_df, 
        x="model", 
        y="f1_score", 
        hue="class", 
        palette="viridis",
        order=custom_order # Explicitly enforce the order here
    )
    
    plt.title(f'Avg F1-Score: {", ".join(TARGET_DIALECTS).upper()} (Core Emotions)')
    plt.ylabel('Average F1 Score')
    plt.xlabel('Model Architecture')
    plt.ylim(0, 1.05)
    plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_plot("single_grouped_f1_comparison.png", "./plots")
    plt.show()

# Creates a heatmap showing Macro F1 scores for each Model (columns)
# and each Dialect (rows).
def plot_macro_f1_heatmap():

    all_files = glob.glob("../results/All_results/*/*classification_report*.csv")
    df_list = []

    for file in all_files:
        # Extract metadata
        model = os.path.basename(os.path.dirname(file))
        dialect = os.path.basename(file).split('_')[0]
        
        df = pd.read_csv(file)
        # Filter for the macro average row
        macro_row = df[(df['row_type'] == 'aggregate') & (df['class'] == 'macro avg')]
        
        if not macro_row.empty:
            f1 = macro_row['f1_score'].values[0]
            df_list.append({'model': model, 'dialect': dialect, 'macro_f1': f1})

    data = pd.DataFrame(df_list)
    pivot_df = data.pivot(index='dialect', columns='model', values='macro_f1')

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='RdYlGn', vmin=0, vmax=1)
    plt.title('Macro F1 Score: Dialect vs. Model')
    save_plot("macro_f1_heatmap.png", "./plots/")
    plt.show()

# Creates a bar graph showing the average Macro F1 score per model
# across all dialects, filtered to TARGET_DIALECTS only.
def plot_macro_f1_bargraph():
    
    all_files = glob.glob("../results/All_results/*/*classification_report*.csv")
    df_list = []

    for file in all_files:
        dialect = os.path.basename(file).split('_')[0]
        if dialect == "spanish":
            dialect = "spain"
        if dialect not in TARGET_DIALECTS:
            continue

        model = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file)
        macro_row = df[(df['row_type'] == 'aggregate') & (df['class'] == 'macro avg')]

        if not macro_row.empty:
            f1 = macro_row['f1_score'].values[0]
            df_list.append({'model': model, 'macro_f1': f1})

    data = pd.DataFrame(df_list)
    
    # Calculate average and standard deviation
    summary = data.groupby('model')['macro_f1'].agg(['mean', 'std']).reset_index()

    # --- REORDERING LOGIC ---
    custom_order = ['mBERT', 'XLM-RoBERTa', 'Gemma', 'Gemma_Few_Shot', 'Llama', 'Llama_Few_Shot']
    summary['model'] = pd.Categorical(summary['model'], categories=custom_order, ordered=True)
    summary = summary.sort_values('model')
    # ------------------------

    # --- BASELINE LOGIC ---
    # Extract the value for XLM-RoBERTa
    baseline_row = summary[summary['model'] == 'XLM-RoBERTa']
    baseline_score = baseline_row['mean'].values[0] if not baseline_row.empty else 0
    # ----------------------

    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x='model', y='mean', palette='muted', capsize=.1)
    
    # Draw the baseline line
    if baseline_score > 0:
        plt.axhline(y=baseline_score, color='red', linestyle='--', linewidth=2, label=f'SOTA Baseline (XLM-R: {baseline_score:.2f})')
        plt.legend(loc='upper right')

    plt.title('Average Macro F1 Performance Across All Dialects')
    plt.ylabel('Macro F1 Score')
    plt.xlabel('Model')
    plt.ylim(0, 1.0)

    save_plot("average_macro_f1_bar.png", "./plots/")
    plt.show()

def plot_macro_f1_comparison_by_dialect(model_folder_name):
    """
    Creates a bar chart: 
    X-axis: Dialects, Y-axis: Macro F1-score.
    Includes custom dialect ordering and focuses on Macro F1.
    """
    # 1. Search path
    search_path = f"../results/All_results/{model_folder_name}/*_classification_report.csv"
    all_files = glob.glob(search_path)
    
    # --- CUSTOM ORDERING ARRAY ---
    custom_dialect_order = ['mexican', 'spain', 'venezuelan', 'argentinian']
    # -----------------------------
    
    df_list = []

    if not all_files:
        print(f"[!] No files found for folder: {model_folder_name}")
        return

    for file in all_files:
        dialect = get_dialect_from_filename(os.path.basename(file))
        
        if dialect not in TARGET_DIALECTS:
            continue
            
        df = pd.read_csv(file)
        
        # --- TARGET MACRO F1 ---
        # Look for the 'macro avg' row
        macro_row = df[(df['row_type'] == 'aggregate') & (df['class'] == 'macro avg')]
        
        if not macro_row.empty:
            f1 = macro_row['f1_score'].values[0]
            df_list.append({'dialect': dialect, 'macro_f1': f1})

    if not df_list:
        print(f"[!] No valid Macro F1 data found in: {model_folder_name}")
        return

    full_df = pd.DataFrame(df_list)

    # 2. Plot
    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        data=full_df, 
        x='dialect', 
        y='macro_f1', 
        palette='viridis',
        order=custom_dialect_order 
    )
    
    plt.title(f'Macro F1-Score by Dialect: {model_folder_name}')
    plt.ylabel('Macro F1 Score')
    plt.xlabel('Dialect')
    plt.ylim(0, 1.05)
    
    save_plot(f"{model_folder_name}_macro_f1_comparison.png", "./plots/comparison")
    plt.close()

# 
def plot_weighted_f1_bargraph():
    """
    Creates a bar graph showing the average WEIGHTED F1 score per model
    across all dialects, filtered to TARGET_DIALECTS only.
    Includes a reference line for the XLM-RoBERTa baseline.
    """
    all_files = glob.glob("../results/All_results/*/*classification_report*.csv")
    df_list = []

    for file in all_files:
        dialect = get_dialect_from_filename(os.path.basename(file))
        if dialect not in TARGET_DIALECTS:
            continue

        model = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file)
        
        # weighted average F1
        weighted_row = df[(df['row_type'] == 'aggregate') & (df['class'] == 'weighted avg')]

        if not weighted_row.empty:
            f1 = weighted_row['f1_score'].values[0]
            df_list.append({'model': model, 'weighted_f1': f1})

    data = pd.DataFrame(df_list)
    
    summary = data.groupby('model')['weighted_f1'].agg(['mean', 'std']).reset_index()

    custom_order = ['mBERT', 'XLM-RoBERTa', 'Gemma', 'Gemma_Few_Shot', 'Llama', 'Llama_Few_Shot']
    summary['model'] = pd.Categorical(summary['model'], categories=custom_order, ordered=True)
    summary = summary.sort_values('model')

    baseline_row = summary[summary['model'] == 'XLM-RoBERTa']
    baseline_score = baseline_row['mean'].values[0] if not baseline_row.empty else 0

    plt.figure(figsize=(10, 6))
    sns.barplot(data=summary, x='model', y='mean', palette='muted', capsize=.1)
    
    # Draw the baseline line
    if baseline_score > 0:
        plt.axhline(y=baseline_score, color='red', linestyle='--', linewidth=2, label=f'SOTA Baseline (XLM-R: {baseline_score:.2f})')
        plt.legend(loc='upper right')

    plt.title('Average Weighted F1 Performance Across All Dialects')
    plt.ylabel('Weighted F1 Score')
    plt.xlabel('Model')
    plt.ylim(0, 1.0)

    save_plot("average_weighted_f1_bar.png", "./plots/")
    plt.close()

def save_plot(filename, folder_name):
    """
    Saves the current plot to a specific folder.
    Creates the folder if it doesn't exist.
    """
    # 1. Ensure the directory exists
    os.makedirs(folder_name, exist_ok=True)
    
    # 2. Define the full path
    full_path = os.path.join(folder_name, filename)
    
    # 3. Save the plot
    # bbox_inches='tight' removes extra white space around the plot
    plt.savefig(full_path, bbox_inches='tight', dpi=300)
    print(f"[SUCCESS] Plot saved to: {full_path}")
    
    # 4. Clear the plot to free up memory
    plt.close()

# Parses the comma-separated keyword string into a clean list.
def parse_keywords(keyword_str):
    if pd.isna(keyword_str) or keyword_str == "":
        return []
    # Remove quotes and split by comma
    clean_str = str(keyword_str).replace('"', '').replace("'", "")
    return [word.strip() for word in clean_str.split(',') if word.strip()]

# Creates graph comparing Average F1 scores amongst the 4 core dialects, 
def plot_f1_comparison_by_dialect(model_folder_name):
    """
    Creates a bar chart: 
    X-axis: Dialects, Y-axis: Weighted F1-score.
    Includes custom dialect ordering.
    """
    search_path = f"../results/All_results/{model_folder_name}/*_classification_report.csv"
    all_files = glob.glob(search_path)
    

    # list to rearrange the X-axis
    custom_dialect_order = ['mexican', 'spain', 'venezuelan', 'argentinian']

    
    df_list = []

    if not all_files:
        print(f"[ERROR] No files found for folder: {model_folder_name}")
        return

    for file in all_files:
        dialect = get_dialect_from_filename(os.path.basename(file))
        
        if dialect not in TARGET_DIALECTS:
            continue
            
        df = pd.read_csv(file)
        
        weighted_row = df[(df['row_type'] == 'aggregate') & (df['class'] == 'weighted avg')]
        
        if not weighted_row.empty:
            f1 = weighted_row['f1_score'].values[0]
            df_list.append({'dialect': dialect, 'weighted_f1': f1})

    if not df_list:
        print(f"[ERROR] No valid weighted F1 data found in: {model_folder_name}")
        return

    full_df = pd.DataFrame(df_list)

    plt.figure(figsize=(10, 6))
    
    sns.barplot(
        data=full_df, 
        x='dialect', 
        y='weighted_f1', 
        palette='viridis',
        order=custom_dialect_order # uses custom order
    )
    
    plt.title(f'Weighted F1-Score by Dialect: {model_folder_name}')
    plt.ylabel('Weighted F1 Score')
    plt.xlabel('Dialect')
    plt.ylim(0, 1.05)
    
    save_plot(f"{model_folder_name}_weighted_f1_comparison.png", "./plots/comparison")
    plt.close()

################ Keyword Cleanliness (Tokenization Quality) ##################
# Plots the percentage of 'clean' keywords (length >= 3) per model.
def plot_keyword_cleanliness():
    pred_files = glob.glob("../results/All_results/*/*_predictions.csv")
    data = []

    for file in pred_files:
        model = os.path.basename(os.path.dirname(file))
        df = pd.read_csv(file)
        
        all_tokens = []
        for kw_str in df['top_keywords'].dropna():
            all_tokens.extend(parse_keywords(kw_str))
            
        if not all_tokens: continue
            
        clean_tokens = []

        for t in all_tokens:
            if len(str(t)) >= 3:
                clean_tokens.append(t)
        score = (len(clean_tokens) / len(all_tokens)) * 100
        data.append({'model': model, 'cleanliness_score': score})

    df_clean = pd.DataFrame(data).groupby('model')['cleanliness_score'].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_clean, x='model', y='cleanliness_score', palette='viridis')
    plt.title('Tokenization Quality: % of "Clean" Keywords (len >= 3)')
    plt.ylabel('Percentage of Clean Tokens')
    plt.ylim(0, 100)
    save_plot("keyword_cleanliness.png", "./plots/analysis")
    plt.close()


####################### Macro F1 bar graph per Language, showing all models ###################

def plot_macro_f1_for_dialect(target_dialect):
    if target_dialect == "spain":
        target_dialect = "spanish_spain"
    
    search_pattern = f"../results/All_results/*/*{target_dialect}*_classification_report.csv"
    all_files = glob.glob(search_pattern)
    print(f"Files found in Gemma_Few_Shot folder: {all_files}")
    
    df_list = []

    model_map = {
        'mBERT': 'mBERT',
        'XLM-RoBERTa': 'XLM-RoBERTa',
        'Gemma': 'Gemma',
        'Gemma_Few_Shot': 'Gemma_Few_Shot',
        'Llama': 'Llama',
        'Llama_Few_Shot': 'Llama_Few_Shot'
    }

    for file in all_files:
        folder_name = os.path.basename(os.path.dirname(file))
        
        matched_model = model_map.get(folder_name)
        
        matched_model = None

        for key, name in model_map.items():
            if key.lower() in folder_name.lower():
                matched_model = name
                break
        
        if matched_model:
            df = pd.read_csv(file)
            macro_row = df[(df['row_type'] == 'aggregate') & (df['class'] == 'macro avg')]
            if not macro_row.empty:
                df_list.append({'model': matched_model, 'macro_f1': macro_row['f1_score'].values[0]})

    if not df_list:
        print(f"[ERROR] No data found for dialect: {target_dialect}")
        return

    data = pd.DataFrame(df_list)
    
    custom_order = list(model_map.values())
    
    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(
        data=data, x='model', y='macro_f1', 
        order=custom_order, 
        palette='viridis', 
        hue='model', 
        legend=False,
        width=0.7 
    )
    
    baseline_score = data[data['model'] == 'XLM-RoBERTa']['macro_f1'].mean()
    if not pd.isna(baseline_score):
        plt.axhline(y=baseline_score, color='red', linestyle='--', linewidth=2, label=f'XLM-R Baseline ({baseline_score:.2f})')
        plt.legend(loc='upper right')

    plt.title(f'Macro F1 Performance: {target_dialect.upper()}')
    plt.ylabel('Macro F1 Score')
    plt.xlabel('Model')
    plt.ylim(0, 1.0)
    
    save_plot(f"{target_dialect}_macro_f1_comparison.png", "./plots/comparison")
    plt.close()

# Loops through all target dialects and generates their specific plots.
def generate_all_dialect_macro_f1_plots():

    for dialect in TARGET_DIALECTS:
        
        plot_macro_f1_for_dialect(dialect)


########################## Confusion Matrix ##########################
        
# Generates and saves a normalized confusion matrix heatmap testing model performance against ground truth for specific dialect sets.
def plot_confusion_matrix(prediction_csv, ground_truth_csv, model_name, dialect):

    preds = pd.read_csv(prediction_csv)
    gold = pd.read_csv(ground_truth_csv)
    
    if len(preds) != len(gold):
        print(f"[ERROR] Mismatch in row count: {len(preds)} preds vs {len(gold)} gold. Skipping.")
        return

    gold_emotions = gold['emotion']
    
    pred_col = 'predicted_emotion' 
    
    labels = sorted(gold['emotion'].unique())
    
    cm = confusion_matrix(gold_emotions, preds[pred_col], labels=labels, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.title(f'Confusion Matrix: {model_name} ({dialect})')
    plt.ylabel('Actual Emotion')
    plt.xlabel('Predicted Emotion')
    
    save_plot(f"{dialect}_{model_name}_confusion_matrix.png", "./plots/confusion")
    plt.close()

# Loops through all folders to find predictions and corresponding ground truth.
def generate_all_confusion_matrices():

    pred_files = glob.glob("../results/All_results/*/*_predictions.csv")

    for pred_file in pred_files:

        model = os.path.basename(os.path.dirname(pred_file))
        dialect = get_dialect_from_filename(os.path.basename(pred_file))
        
        input_dialect = "spanish_spain" if dialect == "spain" else dialect
        ground_truth_file = f"../input_data/{input_dialect}.csv"
        
        if os.path.exists(ground_truth_file):
            print(f"Generating CM for: {model} ({dialect})")
            plot_confusion_matrix(pred_file, ground_truth_file, model, dialect)
        else:
            print(f"[ERROR] Could not find ground truth for {dialect}, skipping.")

# Generates a pie chart showing the distribution for the golden label, 
# then the distrubtion of models in model_name_list
def plot_label_distribution_comparison(model_name_list):

    gold_df_list = []
    for dialect in TARGET_DIALECTS:
        fname = "spanish_spain.csv" if dialect == "spain" else f"{dialect}.csv"
        path = f"../input_data/{fname}"
        if os.path.exists(path):
            df = pd.read_csv(path)
            gold_df_list.append(df[['emotion']])
    
    gold_df = pd.concat(gold_df_list)
    gold_counts = gold_df['emotion'].value_counts()

    num_plots = 1 + len(model_name_list)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    
    def style_pie(ax, counts, title):
        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
        ax.set_title(title)

    style_pie(axes[0], gold_counts, "Ground Truth Distribution")

    for i, model in enumerate(model_name_list):
        pred_df_list = []
        for dialect in TARGET_DIALECTS:
            search_pattern = f"../results/All_results/{model}/{dialect}_*_predictions.csv"
            files = glob.glob(search_pattern)
            
            if files:
                df = pd.read_csv(files[0])
                col_name = 'predicted_emotion' if 'predicted_emotion' in df.columns else 'emotion'
                pred_df_list.append(df[[col_name]].rename(columns={col_name: 'predicted_emotion'}))
        
        if pred_df_list:
            pred_df = pd.concat(pred_df_list)
            pred_counts = pred_df['predicted_emotion'].value_counts()
            style_pie(axes[i+1], pred_counts, f"Model: {model}")
        else:
            print(f"[ERROR] No prediction data found for model folder: {model}")

    plt.tight_layout()
    save_plot("label_distribution_comparison.png", "./plots/analysis")
    plt.show()

# Generates and saves a grouped F1-score comparison plot by emotion and dialect for a specified model
def plot_f1_comparison_by_dialect(model_folder_name, custom_dialect_order=None):

    search_path = f"../results/All_results/{model_folder_name}/*_classification_report.csv"
    all_files = glob.glob(search_path)
    
    if custom_dialect_order is None:
        custom_dialect_order = ['mexican', 'spain', 'venezuelan', 'argentinian']
    
    df_list = []
    target_emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'others', 'other']

    if not all_files:
        print(f"[ERROR] No files found for folder: {model_folder_name}")
        return

    for file in all_files:
        dialect = get_dialect_from_filename(os.path.basename(file))
        
        if dialect not in TARGET_DIALECTS:
            continue
            
        df = pd.read_csv(file)
        
        df = df[(df['row_type'] == 'per_class') & (df['class'].isin(target_emotions))].copy()
        df['dialect'] = dialect
        df_list.append(df)

    if not df_list:
        print(f"[ERROR] No valid per-class data found in: {model_folder_name}")
        return

    full_df = pd.concat(df_list)

    # 2. Plot
    plt.figure(figsize=(14, 7))
    
    sns.barplot(
        data=full_df, 
        x='dialect', 
        y='f1_score', 
        hue='class', 
        palette='viridis',
        order=custom_dialect_order
    )
    
    plt.title(f'F1-Score by Emotion across Dialects: {model_folder_name}')
    plt.ylabel('F1 Score')
    plt.xlabel('Dialect')
    plt.ylim(0, 1.05)
    plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    save_plot(f"{model_folder_name}_f1_comparison.png", "./plots/comparison")
    plt.close()


############################# EXECUTE ##################################


plot_comparative_f1_bars_single()
plot_macro_f1_bargraph()
generate_all_dialect_macro_f1_plots()

generate_all_confusion_matrices()
plot_weighted_f1_bargraph()

plot_label_distribution_comparison(['Gemma_Few_Shot', 'Gemma', 'XLM-RoBERTa'])

plot_f1_comparison_by_dialect("Gemma")
plot_f1_comparison_by_dialect("XLM-RoBERTa")
my_order = ['mexican', 'spain', 'venezuelan', 'argentinian']
plot_f1_comparison_by_dialect("Gemma_Few_Shot", custom_dialect_order=my_order)
plot_f1_comparison_by_dialect("XLM-RoBERTa", custom_dialect_order=my_order)
plot_macro_f1_comparison_by_dialect("Gemma_Few_Shot")
plot_macro_f1_comparison_by_dialect("XLM-RoBERTa")

plot_macro_f1_bargraph()

plot_f1_comparison_by_dialect("RoBERTuito")
plot_f1_comparison_by_dialect("BETO")
plot_f1_comparison_by_dialect("Gemma")
plot_f1_comparison_by_dialect("Llama")
