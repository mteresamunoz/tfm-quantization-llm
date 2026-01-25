import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate scatter plots for LLM metrics comparison')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing metrics CSV files')
    parser.add_argument('--output_dir', type=str, default='./results/scatter_plots',
                        help='Output directory for generated plots')
    parser.add_argument('--add_labels', action='store_true',
                        help='Add value labels to scatter points')
    return parser.parse_args()


def load_and_process_data(data_dir):
    """Load and standardize metrics from CSV files."""
    
    files = {
        'Latxa 3.1 8b': os.path.join(data_dir, "metrics_latxa_3.1_8b.csv"),
        'Qwen 3 8b': os.path.join(data_dir, "metrics_qwen_3_8b.csv"),
        'Gemma 2 9b': os.path.join(data_dir, "metrics_gemma_2_9b.csv")
    }
    
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
        
        # Clean numeric data
        for col in df.columns:
            if col not in ['Method', 'Model']:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '.').str.replace('"', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No data files found or loaded successfully")
    
    final_df = pd.concat(dfs, ignore_index=True)
    
    # Standardize method names
    final_df['Method'] = final_df['Method'].str.replace('int4', 'fp4', regex=False)
    final_df['Method'] = final_df['Method'].str.strip()
    
    print("Processed Data Summary:")
    print(f"Total rows: {len(final_df)}")
    print(f"Methods: {final_df['Method'].unique()}")
    print(f"Models: {final_df['Model'].unique()}")
    
    return final_df


def plot_scatter_charts(df, output_dir, model_palette, add_labels=False):
    """Generate scatter plots for each metric."""
    
    metrics = [
        'Mean accuracy', 'GPU inference (GB)', 'Disk size (GB)',
        'GPU training (GB)', 'Carbon emitted (kg CO2 eq.)',
        'Throughput (Token/s)', 'Efficiency (tok/kWh)', 'Training duration (min)'
    ]
    
    method_order = [
        'Base model (BF16)',
        'LoRA',
        'QLoRA_int8', 'QLoRA_fp4', 'QLoRA_nf4DQ',
        'LoRA-Q_int8', 'LoRA-Q_fp4', 'LoRA-Q_nf4DQ',
        'postquant_int8', 'postquant_fp4', 'postquant_nf4DQ'
    ]
    
    present_methods = df['Method'].unique()
    plot_order = [m for m in method_order if m in present_methods]
    
    for metric in metrics:
        if metric not in df.columns:
            print(f"Skipping {metric} - column not found")
            continue
        
        plt.figure(figsize=(10, 6))
        
        current_plot_order = list(plot_order)
        
        # Hide Base model if it has 0 or NaN values
        base_name = 'Base model (BF16)'
        if base_name in df['Method'].values:
            base_data = df[df['Method'] == base_name][metric]
            has_base_data = base_data.fillna(0).gt(0).any()
            
            if not has_base_data and base_name in current_plot_order:
                current_plot_order.remove(base_name)
        
        if not current_plot_order:
            print(f"Skipping {metric} - no methods to plot")
            plt.close()
            continue
        
        try:
            # Use stripplot for categorical x-axis with scatter-like appearance
            ax = sns.stripplot(
                data=df,
                x='Method',
                y=metric,
                hue='Model',
                palette=model_palette,
                order=current_plot_order,
                size=10,
                jitter=False,
                alpha=0.8,
                linewidth=1,
                edgecolor='black'
            )
            
            # Add grid
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Optional: Add value labels
            if add_labels:
                for i, method in enumerate(current_plot_order):
                    method_data = df[df['Method'] == method]
                    for _, row in method_data.iterrows():
                        if pd.notna(row[metric]) and row[metric] > 0:
                            # Determine format
                            if df[metric].max() > 1000:
                                label = f'{row[metric]:.0f}'
                            elif df[metric].max() < 1:
                                label = f'{row[metric]:.3f}'
                            else:
                                label = f'{row[metric]:.2f}'
                            
                            plt.text(i, row[metric], label, 
                                   ha='center', va='bottom', 
                                   fontsize=7, rotation=0)
            
            plt.title(f'Comparison of {metric} across Models', fontsize=14, pad=15, fontweight='bold')
            plt.xlabel('Quantization / Fine-tuning Method', fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title='Model Family', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            safe_metric_name = metric.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')
            plt.savefig(os.path.join(output_dir, f'scatter_{safe_metric_name}.png'))
            print(f"Generated: scatter_{safe_metric_name}.png")
            
        except Exception as e:
            print(f"Error plotting {metric}: {e}")
        
        plt.close()


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup plotting style
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Tahoma']
    plt.rcParams['figure.dpi'] = 300
    
    model_palette = {
        'Latxa 3.1 8b': '#1f77b4',
        'Qwen 3 8b': '#d62728',
        'Gemma 2 9b': '#2ca02c'
    }
    
    print("Loading and processing data...")
    df = load_and_process_data(args.data_dir)
    
    print("Generating scatter plots...")
    plot_scatter_charts(df, args.output_dir, model_palette, args.add_labels)
    
    print(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
