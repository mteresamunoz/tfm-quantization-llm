import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from math import pi
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate advanced trade-off plots for LLM metrics')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing metrics CSV files')
    parser.add_argument('--output_dir', type=str, default='./results/plots',
                        help='Output directory for generated plots')
    return parser.parse_args()


def load_and_process_data(data_dir):
    """Load and standardize metrics from CSV files."""
    
    # Define input files
    files = {
        'Latxa 3.1 8b': os.path.join(data_dir, "metrics_latxa_3.1_8b.csv"),
        'Qwen 3 8b': os.path.join(data_dir, "metrics_qwen_3_8b.csv"),
        'Gemma 2 9b': os.path.join(data_dir, "metrics_gemma_2_9b.csv")
    }
    
    # Define column mappings to standardize names
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
            print(f"Warning: File not found: {filepath}")
            continue
        
        df = pd.read_csv(filepath)
        df = df.rename(columns=mappings[model_name])
        df['Model'] = model_name
        cols_to_keep = [c for c in standard_columns if c in df.columns]
        df = df[cols_to_keep]
        
        # Convert string numbers to float
        for col in df.columns:
            if col not in ['Method', 'Model']:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '.').str.replace('"', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        dfs.append(df)
    
    final_df = pd.concat(dfs, ignore_index=True)
    final_df['Method'] = final_df['Method'].str.replace('int4', 'fp4', regex=False).str.strip()
    # Exclude LoRA as it's baseline-only comparison
    final_df = final_df[final_df['Method'] != 'LoRA']
    
    return final_df


def plot_tradeoffs(df, output_dir, model_palette):
    """Generate trade-off scatter plots."""
    
    tradeoffs = [
        ('GPU inference (GB)', 'Mean accuracy', 'Memory Footprint vs Accuracy'),
        ('Carbon emitted (kg CO2 eq.)', 'Mean accuracy', 'Carbon Footprint vs Accuracy'),
        ('Training duration (min)', 'Throughput (Token/s)', 'Training Cost vs Inference Speed')
    ]
    
    for x_col, y_col, title in tradeoffs:
        plt.figure(figsize=(10, 6))
        
        plot_data = df.dropna(subset=[x_col, y_col])
        plot_data = plot_data[(plot_data[x_col] > 0) & (plot_data[y_col] > 0)]
        
        sns.scatterplot(
            data=plot_data,
            x=x_col,
            y=y_col,
            hue='Model',
            style='Method',
            palette=model_palette,
            s=150,
            alpha=0.9,
            edgecolor='black'
        )
        
        plt.title(f'Trade-off: {title}', fontsize=14, fontweight='bold', pad=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'tradeoff_{x_col.split()[0]}_vs_{y_col.split()[0]}.png'))
        plt.close()
    
    print(f"Trade-off plots saved to {output_dir}")


def create_radar_chart(data, labels, processed_metrics, title, filename):
    """Generate radar chart for multi-metric comparison."""
    
    categories = labels
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], categories, size=9)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["", "", "", ""], color="grey", size=7)
    plt.ylim(0, 1.05)
    
    linestyles = ['-', '--', '-.', ':']
    
    for i, (idx, row) in enumerate(data.iterrows()):
        values = row[processed_metrics].values.flatten().tolist()
        values += values[:1]
        values = [v if not pd.isna(v) else 0 for v in values]
        
        ax.plot(angles, values, linewidth=2, linestyle=linestyles[i % len(linestyles)], label=row['Method'])
        ax.fill(angles, values, alpha=0.05)
    
    plt.title(title, size=14, color='black', y=1.1, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.2, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_radar_charts(df, output_dir):
    """Generate normalized radar charts for each model."""
    
    radar_metrics = {
        'Mean accuracy': 'max',
        'Throughput (Token/s)': 'max',
        'GPU inference (GB)': 'min',
        'Carbon emitted (kg CO2 eq.)': 'min'
    }
    
    radar_df = df.copy()
    processed_metrics = []
    labels = []
    
    for metric, direction in radar_metrics.items():
        if metric not in radar_df.columns:
            continue
        
        clean_series = radar_df[metric].dropna()
        if clean_series.empty:
            continue
        
        min_v = clean_series.min()
        max_v = clean_series.max()
        
        new_col = f'{metric}_norm'
        processed_metrics.append(new_col)
        
        if direction == 'max':
            radar_df[new_col] = (radar_df[metric] - min_v) / (max_v - min_v) if max_v > min_v else 0.5
        else:
            radar_df[new_col] = (max_v - radar_df[metric]) / (max_v - min_v) if max_v > min_v else 0.5
        
        labels.append(metric)
    
    for model in df['Model'].unique():
        model_data = radar_df[radar_df['Model'] == model].copy()
        
        key_methods = ['Base model (BF16)', 'QLoRA_nf4DQ', 'QLoRA_fp4']
        subset = model_data[model_data['Method'].isin(key_methods)]
        
        if not subset.empty:
            filename = os.path.join(output_dir, f'radar_{model.replace(" ", "_")}.png')
            create_radar_chart(subset, labels, processed_metrics, model, filename)
    
    print(f"Radar charts saved to {output_dir}")


def plot_pareto(df, x_col, y_col, output_dir, model_palette, filename):
    """Generate Pareto frontier visualization."""
    
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        data=df, x=x_col, y=y_col, hue='Model', style='Method',
        palette=model_palette, s=150, alpha=0.9
    )
    
    plt.title(f'Trade-off: {y_col} vs {x_col}', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup plotting style
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['figure.dpi'] = 300
    
    model_palette = {
        'Latxa 3.1 8b': '#1f77b4',
        'Qwen 3 8b': '#d62728',
        'Gemma 2 9b': '#2ca02c'
    }
    
    # Load data
    print("Loading and processing data...")
    df = load_and_process_data(args.data_dir)
    
    # Generate plots
    print("Generating trade-off scatter plots...")
    plot_tradeoffs(df, args.output_dir, model_palette)
    
    print("Generating radar charts...")
    plot_radar_charts(df, args.output_dir)
    
    print("Generating Pareto frontier plots...")
    plot_pareto(df, 'GPU inference (GB)', 'Mean accuracy', args.output_dir, model_palette, 'pareto_acc_gpu.png')
    plot_pareto(df, 'Carbon emitted (kg CO2 eq.)', 'Mean accuracy', args.output_dir, model_palette, 'pareto_acc_carbon.png')
    
    print(f"All plots generated successfully in {args.output_dir}")


if __name__ == "__main__":
    main()
