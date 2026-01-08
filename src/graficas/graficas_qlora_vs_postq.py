import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('/gaueko1/users/mmartin/tfm-quantization-llm/footprint_metricas/inference_emissions.csv')

# Filter ONLY QLoRA and PostQuant models
qlora_postquant_models = ['qlora8', 'qlora4', 'qlora_nf4', 'post_fp8', 'post_q4', 'post_q8', 'post_nf4']
df_filtered = df[df['model'].isin(qlora_postquant_models)].copy()

print("Filtered models:")
print(df_filtered['model'].tolist())

# General configuration
plt.style.use('default')
fig_size = (10, 6)

# Helper function to create sorted bar chart
def create_sorted_bar_chart(df, x_col, y_col, title, ylabel, filename):
    # Sort data from highest to lowest
    sorted_df = df.sort_values(y_col, ascending=False)
    
    plt.figure(figsize=fig_size)
    bars = plt.bar(range(len(sorted_df)), sorted_df[y_col])
    plt.xlabel('Model')
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Add model labels rotated
    plt.xticks(range(len(sorted_df)), sorted_df[x_col], rotation=45, ha='right')
    
    # Add value labels on top of bars
    max_y = max(sorted_df[y_col])
    for bar, value in zip(bars, sorted_df[y_col]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max_y, 
                f'{value:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# FIGURE 1: Time per query (ms) - HIGHEST TO LOWEST
create_sorted_bar_chart(df_filtered, 'model', 'latency_ms', 'Time per Query (ms) - QLoRA vs PostQuant', 'Latency (ms)', 'qlora_postquant_fig1_latency.png')

# FIGURE 2: Emissions per query (μg CO₂) - HIGHEST TO LOWEST
create_sorted_bar_chart(df_filtered, 'model', 'emissions_per_query_ug', 'Emissions per Query (μg CO₂) - QLoRA vs PostQuant', 'Emissions (μg CO₂)', 'qlora_postquant_fig2_emissions.png')

# FIGURE 3: Throughput (tokens/second) - HIGHEST TO LOWEST
create_sorted_bar_chart(df_filtered, 'model', 'tokens_per_sec', 'Throughput (tokens/s) - QLoRA vs PostQuant', 'Throughput (tokens/s)', 'qlora_postquant_fig3_throughput.png')

# FIGURE 4: Efficiency (tok/kWh) - HIGHEST TO LOWEST
create_sorted_bar_chart(df_filtered, 'model', 'tok_per_kwh', 'Energy Efficiency (tok/kWh) - QLoRA vs PostQuant', 'Energy Efficiency (tok/kWh)', 'qlora_postquant_fig4_efficiency.png')

# FIGURE 5: VRAM peak (GB) - HIGHEST TO LOWEST
create_sorted_bar_chart(df_filtered, 'model', 'vram_peak_gb', 'Peak VRAM Usage (GB) - QLoRA vs PostQuant', 'VRAM Peak (GB)', 'qlora_postquant_fig5_vram.png')

print("QLoRA vs PostQuant figures generated successfully - sorted from HIGHEST to LOWEST")
