
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Define paths (Relative to script execution or absolute within workspace)
base_dir = r"c:\Users\UJA\Desktop\HiTZ GreenAI TFM\Metrics llms"
files = {
    'Latxa 3.1 8b': os.path.join(base_dir, "metrics latxa 3.1 8b - Hoja 1.csv"),
    'Qwen 3 8b': os.path.join(base_dir, "metrics qwen 3 8b - Hoja 1.csv"),
    'Gemma 2 9b': os.path.join(base_dir, "metrics gemma 2 9b - Hoja 1.csv")
}

output_dir = os.path.join(base_dir, "scotter_plots")
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
    df = pd.read_csv(filepath)
    df = df.rename(columns=mappings[model_name])
    df['Model'] = model_name
    cols_to_keep = [c for c in standard_columns if c in df.columns]
    df = df[cols_to_keep]
    for col in df.columns:
        if col not in ['Method', 'Model']:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.').str.replace('"', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
    dfs.append(df)

if not dfs:
    print("No data loaded.")
    exit()

final_df = pd.concat(dfs, ignore_index=True)
final_df['Method'] = final_df['Method'].str.replace('int4', 'fp4', regex=False).str.strip()

# PLOTTING SETUP
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Tahoma']
plt.rcParams['figure.dpi'] = 300

model_palette = {
    'Latxa 3.1 8b': '#1f77b4', 
    'Qwen 3 8b': '#d62728',    
    'Gemma 2 9b': '#2ca02c'    
}

metrics = [
    'Mean accuracy', 'GPU inference (GB)', 'Disk size (GB)', 
    'GPU training (GB)', 'Carbon emitted (kg CO2 eq.)', 
    'Throughput (Token/s)', 'Efficiency (tok/kWh)', 'Training duration (min)'
]

method_order = [
    'base_model (BF16)', 
    'lora', 
    'qlora_int8', 'qlora_fp4', 'qlora_nf4DQ',
    'postquant_int8', 'postquant_fp4', 'postquant_nf4DQ'
]

present_methods = final_df['Method'].unique()
plot_order = [m for m in method_order if m in present_methods]

# --- GROUPED SCATTER PLOTS ---
for metric in metrics:
    plt.figure(figsize=(10, 6))
    
    # Dynamic logic: Hide base_model if 0 or NaN
    current_plot_order = list(plot_order)
    base_data = final_df[final_df['Method'] == 'base_model (BF16)'][metric]
    has_base_data = base_data.fillna(0).gt(0).any()
    
    if not has_base_data and 'base_model (BF16)' in current_plot_order:
        current_plot_order.remove('base_model (BF16)')
    
    try:
        # Using sns.scatterplot (Categorical X)
        # We can use stripplot for better handling of categorical data, but scatterplot usually works too.
        # Stripplot is specifically designed for "one dimension categorical, one numerical".
        # We'll use Stripplot but look like a scatter.
        ax = sns.stripplot(
            data=final_df,
            x='Method',
            y=metric,
            hue='Model',
            palette=model_palette,
            order=current_plot_order,
            size=10,        # Large dots
            jitter=False,   # No jitter since we want exact values aligned
            alpha=0.8,
            linewidth=1,
            edgecolor='black' # Border around dots for "professional" look
        )
        
        # Add grid for easier reading of values
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Labels
        # Optional: Add text labels next to dots? Might be too cluttered. 
        # User asked for labels "encima de cada barra" for bars. 
        # For scatter, labels can be tricky. I'll add them if space permits or just keep it clean.
        # Let's add them slightly offset.
        
        for i, row in final_df.iterrows():
            if row['Method'] not in current_plot_order: continue
            if pd.isna(row[metric]) or row[metric] == 0: continue
            
            # Find the x-position. Stripplot places categories at integer indices 0, 1, 2...
            # This is hard to map back perfectly without manual handling.
            # Instead, let's just rely on the dots. If user wants labels, I can try `text`.
            pass

        # Customize
        plt.title(f'Comparison of {metric} across Models (Scatter)', fontsize=14, pad=15, fontweight='bold')
        plt.xlabel('Quantization / Fine-tuning Method', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Model Family', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        safe_metric_name = metric.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
        plt.savefig(os.path.join(output_dir, f'scatter_{safe_metric_name}.png'))
    except Exception as e:
        print(f"Could not plot {metric}: {e}")
    plt.close()

print(f"Done. Plots in {output_dir}")
