
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from math import pi

# Define paths (Relative to script execution or absolute within workspace)
base_dir = r"c:\Users\UJA\Desktop\HiTZ GreenAI TFM\Metrics llms"
files = {
    'Latxa 3.1 8b': os.path.join(base_dir, "metrics latxa 3.1 8b - Hoja 1.csv"),
    'Qwen 3 8b': os.path.join(base_dir, "metrics qwen 3 8b - Hoja 1.csv"),
    'Gemma 2 9b': os.path.join(base_dir, "metrics gemma 2 9b - Hoja 1.csv")
}

output_dir = os.path.join(base_dir, "bar_plots")
os.makedirs(output_dir, exist_ok=True)

# Define column mappings for each file to standardize names
mappings = {
    'Latxa 3.1 8b': {
        'Modelo': 'Method',
        'media acc (lm harness)': 'Mean accuracy',
        'GPU inference(GB) (Cargar modelo con la misma config de bnb, si no no funciona)': 'GPU inference (GB)',
        'Disk size (GB)': 'Disk size (GB)',
        'VRAM (durante ft) gb': 'GPU training (GB)',
        'Carbon emitted (kg CO2 eq.)': 'Carbon emitted (kg CO2 eq.)',
        'Token/s': 'Throughput (Token/s)',
        'Eficiencia(tok/kWh)': 'Efficiency (tok/kWh)',
        'Tiempo entrenamiento LoRa (min)': 'Training duration (min)'
    },
    'Qwen 3 8b': {
        'Modelo': 'Method',
        'media acc (lm harness)': 'Mean accuracy',
        'GPU inference(GB)': 'GPU inference (GB)',
        'Tamaño modelo en disco(du -sh)': 'Disk size (GB)',
        'VRAM (durante ft)': 'GPU training (GB)',
        'Emisiones de CO₂e (g/query)': 'Carbon emitted (kg CO2 eq.)', 
        'Token/s': 'Throughput (Token/s)',
        'Eficiencia(tok/kWh)': 'Efficiency (tok/kWh)',
        'Tiempo entrenamiento LoRa(min)': 'Training duration (min)'
    },
    'Gemma 2 9b': {
        'Modelo': 'Method',
        'media acc (lm harness)': 'Mean accuracy',
        'GPU inference(GB)': 'GPU inference (GB)',
        'Tamaño modelo en disco(du -sh)': 'Disk size (GB)',
        'VRAM (durante ft)': 'GPU training (GB)',
        'Emisiones de CO₂e (g/query)': 'Carbon emitted (kg CO2 eq.)',
        'Token/s': 'Throughput (Token/s)',
        'Eficiencia(tok/kWh)': 'Efficiency (tok/kWh)',
        'Tiempo entrenamiento LoRa': 'Training duration (min)'
    }
}

standard_columns = [
    'Method', 'Model', 
    'Mean accuracy', 'GPU inference (GB)', 'Disk size (GB)', 
    'GPU training (GB)', 'Carbon emitted (kg CO2 eq.)', 
    'Throughput (Token/s)', 'Efficiency (tok/kWh)', 'Training duration (min)'
]

dfs = []

for model_name, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
        
    # Read CSV
    df = pd.read_csv(filepath)
    
    # Rename columns
    df = df.rename(columns=mappings[model_name])
    
    # Add Model column
    df['Model'] = model_name
    
    # Keep only relevant columns if they exist
    cols_to_keep = [c for c in standard_columns if c in df.columns]
    df = df[cols_to_keep]
    
    # Clean data: Replace commas with dots and convert to numeric
    for col in df.columns:
        if col not in ['Method', 'Model']:
            if df[col].dtype == object:
                # Replace comma with dot, remove quotes if any
                df[col] = df[col].astype(str).str.replace(',', '.').str.replace('"', '')
                # Force to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    dfs.append(df)

if not dfs:
    print("No data loaded.")
    exit()

# Combine all data
final_df = pd.concat(dfs, ignore_index=True)

# Data Cleaning on Method Names
# "int4" -> "fp4" in Method names
final_df['Method'] = final_df['Method'].str.replace('int4', 'fp4', regex=False)
final_df['Method'] = final_df['Method'].str.strip()

print("Processed Data Head:")
print(final_df.head())
print(final_df['Method'].unique())

# --- PLOTTING SETUP ---
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Tahoma']
plt.rcParams['figure.dpi'] = 300

# Colors for models
model_palette = {
    'Latxa 3.1 8b': '#1f77b4', # Blue
    'Qwen 3 8b': '#d62728',    # Red
    'Gemma 2 9b': '#2ca02c'    # Green
}

# Metrics to plot
metrics = [
    'Mean accuracy', 'GPU inference (GB)', 'Disk size (GB)', 
    'GPU training (GB)', 'Carbon emitted (kg CO2 eq.)', 
    'Throughput (Token/s)', 'Efficiency (tok/kWh)', 'Training duration (min)'
]

# Order of methods
# Order of methods
method_order = [
    'Base model (BF16)', 
    'LoRA', 
    'QLoRA_int8', 'QLoRA_fp4', 'QLoRA_nf4DQ',
    'LoRA-Q_int8', 'LoRA-Q_fp4', 'LoRA-Q_nf4DQ',
    'postquant_int8', 'postquant_fp4', 'postquant_nf4DQ'
]

# Filter method_order to only include methods present in data
present_methods = final_df['Method'].unique()
plot_order = [m for m in method_order if m in present_methods]

# --- 1. GROUPED BAR CHARTS (8 Plots) ---
for metric in metrics:
    plt.figure(figsize=(10, 6))
    
    # Determine order for this specific metric
    current_plot_order = list(plot_order)
    
    # User Request: Include 'LoRA' ONLY for 'Mean accuracy'
    if metric == 'Mean accuracy':
        pass # Keep LoRA if present
    else:
        if 'LoRA' in current_plot_order:
            current_plot_order.remove('LoRA')

    # User Request: Hide Base model if it has 0 or NaN values for this metric
    base_name = 'Base model (BF16)'
    if base_name in final_df['Method'].values:
        base_data = final_df[final_df['Method'] == base_name][metric]
        has_base_data = base_data.fillna(0).gt(0).any()
        
        if not has_base_data and base_name in current_plot_order:
            current_plot_order.remove(base_name)

    # Create barplot
    try:
        if not current_plot_order:
            print(f"Skipping {metric} - no methods to plot.")
            continue

        ax = sns.barplot(
            data=final_df,
            x='Method',
            y=metric,
            hue='Model',
            palette=model_palette,
            order=current_plot_order,
            edgecolor='black',
            linewidth=0.5
        )

        # Add data labels
        for container in ax.containers:
            fmt = '%.2f'
            if final_df[metric].max() > 1000:
                fmt = '%.0f'
            elif final_df[metric].max() < 1:
                fmt = '%.3f'
            ax.bar_label(container, fmt=fmt, fontsize=8, padding=3, rotation=90)
        
        # Customize
        plt.title(f'Comparison of {metric} across Models', fontsize=14, pad=20, fontweight='bold')
        plt.xlabel('Quantization / Fine-tuning Method', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Model Family', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.margins(y=0.2) # Add margin for labels
        plt.tight_layout()
        
        # Save
        safe_metric_name = metric.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
        # Special case for Mean accuracy if needed, but standard naming is fine if it overwrites the target
        plt.savefig(os.path.join(output_dir, f'bar_{safe_metric_name}.png'))
    except Exception as e:
        print(f"Could not plot {metric}: {e}")
    plt.close()


print(f"Done. Plots in {output_dir}")

