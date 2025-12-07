import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_and_merge_csvs(model_name, dataset_configs, results_root="results/winobias"):
    """
    Load CSVs for all dataset configs, merge type1 and type2 for pro/anti stereotype.
    Returns a dict: { 'pro': df, 'anti': df } with F1 computed across male+female counts.
    """
    data = {'pro': [], 'anti': []}
    
    for config in dataset_configs:
        stereotype = 'pro' if 'pro' in config else 'anti'
        results_dir = os.path.join(results_root, model_name.replace('/', '_'), config)
        occ_path = os.path.join(results_dir, f"per_occupation_confusion_{config}.csv")
        if not os.path.exists(occ_path):
            print(f"Warning: {occ_path} does not exist, skipping.")
            continue
        df = pd.read_csv(occ_path)
        data[stereotype].append(df)
    
    merged_data = {}
    for stereotype in ['pro', 'anti']:
        if not data[stereotype]:
            continue
        df = pd.concat(data[stereotype])
        df = df.groupby('Occupation', as_index=False).sum()
        # Compute overall F1 across male + female
        df['F1'] = df.apply(
            lambda row: (2 * row['Male_Correct'] + 2 * row['Female_Correct']) /
                        (row['Male_Total'] + row['Female_Total'] + row['Male_Total'] + row['Female_Total']) * 100
                        if (row['Male_Total'] + row['Female_Total']) > 0 else 0,
            axis=1
        )
        merged_data[stereotype] = df
    return merged_data

def plot_f1_scores(merged_data, model_name, save_path=None):
    """
    Plot horizontal bar chart per occupation with pro/anti stereotype F1 scores.
    Adds a final bar for overall pro/anti F1.
    """
    occupations = merged_data['pro']['Occupation'].tolist()
    y = np.arange(len(occupations))
    height = 0.4

    # Compute overall F1 for pro-stereotype
# Compute overall accuracy for pro-stereotype
    overall_pro = merged_data['pro'][['Male_Correct','Male_Total','Female_Correct','Female_Total']].sum()
    overall_pro_f1_score = (overall_pro['Male_Correct'] + overall_pro['Female_Correct']) / \
                        (overall_pro['Male_Total'] + overall_pro['Female_Total']) * 100 if (overall_pro['Male_Total'] + overall_pro['Female_Total']) > 0 else 0

    # Compute overall accuracy for anti-stereotype
    overall_anti = merged_data['anti'][['Male_Correct','Male_Total','Female_Correct','Female_Total']].sum()
    overall_anti_f1_score = (overall_anti['Male_Correct'] + overall_anti['Female_Correct']) / \
                        (overall_anti['Male_Total'] + overall_anti['Female_Total']) * 100 if (overall_anti['Male_Total'] + overall_anti['Female_Total']) > 0 else 0

    
    fig, ax = plt.subplots(figsize=(12, max(6, len(occupations)*0.3)))

    bars_pro = ax.barh(y - height/2, merged_data['pro']['F1'], height, label='Pro-stereotype', color='skyblue')
    bars_anti = ax.barh(y + height/2, merged_data['anti']['F1'], height, label='Anti-stereotype', color='salmon')

    # Add F1 score labels on bars
    for i, (bar, f1) in enumerate(zip(bars_pro, merged_data['pro']['F1'])):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{f1:.2f}', va='center', ha='left', fontsize=7)
    
    for i, (bar, f1) in enumerate(zip(bars_anti, merged_data['anti']['F1'])):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{f1:.2f}', va='center', ha='left', fontsize=7)

    # Add overall bars at the end
    bar_overall_pro = ax.barh(len(occupations) - height/2, overall_pro_f1_score, height, color='skyblue')
    bar_overall_anti = ax.barh(len(occupations) + height/2, overall_anti_f1_score, height, color='salmon')
    
    # Add labels for overall bars
    ax.text(bar_overall_pro[0].get_width() + 0.5, bar_overall_pro[0].get_y() + bar_overall_pro[0].get_height()/2,
            f'{overall_pro_f1_score:.2f}', va='center', ha='left', fontsize=7)
    ax.text(bar_overall_anti[0].get_width() + 0.5, bar_overall_anti[0].get_y() + bar_overall_anti[0].get_height()/2,
            f'{overall_anti_f1_score:.2f}', va='center', ha='left', fontsize=7)

    ax.set_yticks(list(y) + [len(occupations)])
    ax.set_yticklabels(occupations + ['Overall'])
    ax.set_xlabel("F1-score (%)")
    ax.set_title(f"{model_name} - WinoBias F1-scores per Occupation and Stereotype")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    # plt.show()

def get_model_display_name(model_name):
    """
    Translate model names to display labels.
    """
    name_mapping = {
        'google_flan-t5-base': 'FLAN-T5-Base (Baseline)',
        'meta-llama_Llama-3.2-1B-Instruct_base': 'LLaMA 3.2-1B (Baseline)',
        # Add patterns for your models
    }
    
    # Check direct mapping first
    if model_name in name_mapping:
        return name_mapping[model_name]
    
    # Parse pattern: pretraining_finetuned_text_flant5_base_50bal_lora_5e-05 or 8imbal_lora_5e-04
    # Extract: model type, balanced/imbalanced number, LoRA/full, learning rate
    match = re.search(r'finetuned_(?:text_)?flant5_base_(\d+)(bal|imbal)_(lora|full)(?:_([0-9e\-]+))?', model_name)
    if match:
        num = match.group(1)
        bal_type = 'bal' if match.group(2) == 'bal' else 'imb'
        lr = match.group(4) if match.group(4) else '5e-4'
        return f'{bal_type}={num}, LR={lr}'
    
    # Parse LLaMA pattern: finedtuned_llama32_200_imb or pretraining_finedtuned_llama32_20bal
    match = re.search(r'llama32_(\d+)(?:bal|_imb)', model_name)
    if match:
        bal_num = match.group(1)
        return f'LLaMA 3.2-1B LoRA (bal={bal_num})'
    
    return model_name

def compute_global_metrics(model_name, dataset_configs, results_root="results/winobias"):
    """
    Compute global metrics for a model:
    - Overall F1 (neutral)
    - Pro-stereotype F1
    - Anti-stereotype F1
    - Disparity (|Pro - Anti|)
    """
    merged_data = load_and_merge_csvs(model_name, dataset_configs, results_root)
    
    if 'pro' not in merged_data or 'anti' not in merged_data:
        return None
    
    # Compute overall pro F1
    overall_pro = merged_data['pro'][['Male_Correct','Male_Total','Female_Correct','Female_Total']].sum()
    pro_f1 = (overall_pro['Male_Correct'] + overall_pro['Female_Correct']) / \
             (overall_pro['Male_Total'] + overall_pro['Female_Total']) * 100 \
             if (overall_pro['Male_Total'] + overall_pro['Female_Total']) > 0 else 0
    
    # Compute overall anti F1
    overall_anti = merged_data['anti'][['Male_Correct','Male_Total','Female_Correct','Female_Total']].sum()
    anti_f1 = (overall_anti['Male_Correct'] + overall_anti['Female_Correct']) / \
              (overall_anti['Male_Total'] + overall_anti['Female_Total']) * 100 \
              if (overall_anti['Male_Total'] + overall_anti['Female_Total']) > 0 else 0
    
    # Global F1 (average of pro and anti)
    global_f1 = (pro_f1 + anti_f1) / 2
    
    # Disparity
    disparity = abs(pro_f1 - anti_f1)
    
    return {
        'model_name': model_name,
        'display_name': get_model_display_name(model_name),
        'global_f1': global_f1,
        'pro_f1': pro_f1,
        'anti_f1': anti_f1,
        'disparity': disparity
    }

def plot_comparison_bars(models_list, dataset_configs, results_root="results/winobias", save_path=None):
    """
    Create a horizontal bar plot comparing all experiments with 4 bars per model:
    - Global F1 (neutral color)
    - Pro-stereotype F1
    - Anti-stereotype F1
    - Disparity
    """
    metrics_list = []
    for model in models_list:
        metrics = compute_global_metrics(model, dataset_configs, results_root)
        if metrics:
            metrics_list.append(metrics)
    
    if not metrics_list:
        print("No metrics computed for comparison plot")
        return
    
    # Reverse order so first model appears at top
    metrics_list = metrics_list[::-1]
    
    # Prepare data
    display_names = [m['display_name'] for m in metrics_list]
    global_f1s = [m['global_f1'] for m in metrics_list]
    pro_f1s = [m['pro_f1'] for m in metrics_list]
    anti_f1s = [m['anti_f1'] for m in metrics_list]
    disparities = [m['disparity'] for m in metrics_list]
    
    y = np.arange(len(display_names))
    height = 0.2
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(display_names)*0.6)))
    
    bars1 = ax.barh(y - 1.5*height, global_f1s, height, label='Global F1', color='gray', alpha=0.7)
    bars2 = ax.barh(y - 0.5*height, pro_f1s, height, label='Pro-Stereotype F1', color='skyblue')
    bars3 = ax.barh(y + 0.5*height, anti_f1s, height, label='Anti-Stereotype F1', color='salmon')
    bars4 = ax.barh(y + 1.5*height, disparities, height, label='Disparity (|Pro-Anti|)', color='orange')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            width_val = bar.get_width()
            ax.text(width_val + 0.5, bar.get_y() + bar.get_height()/2.,
                   f'{width_val:.2f}',
                   ha='left', va='center', fontsize=7)
    
    ax.set_xlabel('F1 Score (%)')
    ax.set_title('FLAN-T5-Base Fine-tuned Models: F1 Scores and Bias Disparity')
    ax.set_yticks(y)
    ax.set_yticklabels(display_names)
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def generate_latex_table(models_list, dataset_configs, results_root="results/winobias", save_path=None):
    """
    Generate a LaTeX table organized by proportion, learning rate, and sample number.
    """
    metrics_list = []
    for model in models_list:
        metrics = compute_global_metrics(model, dataset_configs, results_root)
        if metrics:
            # Check if baseline model
            if 'google_flan-t5-base' in model or 'Baseline' in metrics['display_name']:
                metrics['bal_num'] = -1  # Sort first
                metrics['bal_type'] = 'baseline'
                metrics['lr'] = ''
                metrics['is_baseline'] = True
            else:
                # Extract balanced/imbalanced number and learning rate
                match = re.search(r'(\d+)(bal|imbal).*?(?:_([0-9e\-]+))?$', model)
                bal_num = int(match.group(1)) if match else 0
                bal_type = match.group(2) if match and match.group(2) else 'bal'
                lr = match.group(3) if match and match.group(3) else '5e-4'
                
                metrics['bal_num'] = bal_num
                metrics['bal_type'] = bal_type
                metrics['lr'] = lr
                metrics['is_baseline'] = False
            metrics_list.append(metrics)
    
    # Sort: baseline first (-1), then by bal_type, lr, bal_num
    # Custom LR ordering: 5e-04, 5e-05, 2e-05
    lr_order = {'5e-04': 0, '5e-4': 0, '5e-05': 1, '2e-05': 2}
    metrics_list.sort(key=lambda x: (
        0 if x['bal_num'] == -1 else 1,  # Baseline first
        x['bal_type'],                    # Then by type (bal/imbal)
        lr_order.get(x['lr'], 999),       # Then by custom LR order
        x['bal_num']                      # Then by sample number
    ))
    
    # Generate LaTeX table
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Model Performance by Proportion, Learning Rate and Sample Number}")
    latex.append("\\label{tab:model_comparison}")
    
    latex.append("\\begin{tabular}{|c|c|c|cccc|}")
    latex.append("\\hline")
    
    # Header row
    latex.append("Type & LR & Samples & Global & Pro & Anti & Disp. \\\\")
    latex.append("\\hline")
    
    # Track previous values for multirow
    bal_type_rows = {}
    lr_rows = {}
    
    # First pass: count rows per bal_type and lr
    for m in metrics_list:
        key = m['bal_type']
        bal_type_rows[key] = bal_type_rows.get(key, 0) + 1
        lr_key = (m['bal_type'], m['lr'])
        lr_rows[lr_key] = lr_rows.get(lr_key, 0) + 1
    
    # Second pass: generate rows
    bal_type_counter = {}
    lr_counter = {}
    
    for m in metrics_list:
        row = []
        
        # Handle baseline model separately
        if m.get('is_baseline', False):
            row.append("Base")  # Type column
            row.append("-")  # LR column
            row.append("-")  # Samples column
        else:
            # Type column (with multirow)
            bal_type_key = m['bal_type']
            if bal_type_key not in bal_type_counter:
                bal_type_counter[bal_type_key] = 0
            
            if bal_type_counter[bal_type_key] == 0:
                type_label = "BALANCED" if m['bal_type'] == 'bal' else "IMBALANCED"
                row.append(f"\\multirow{{{bal_type_rows[bal_type_key]}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{type_label}}}}}")
            else:
                row.append("")
            bal_type_counter[bal_type_key] += 1
            
            # Learning rate column (with multirow)
            lr_key = (m['bal_type'], m['lr'])
            if lr_key not in lr_counter:
                lr_counter[lr_key] = 0
            
            if lr_counter[lr_key] == 0:
                row.append(f"\\multirow{{{lr_rows[lr_key]}}}{{*}}{{{m['lr']}}}")
            else:
                row.append("")
            lr_counter[lr_key] += 1
            
            # Sample number
            row.append(str(m['bal_num']))
        
        # Metrics
        row.extend([
            f"{m['global_f1']:.2f}",
            f"{m['pro_f1']:.2f}",
            f"{m['anti_f1']:.2f}",
            f"{m['disparity']:.2f}"
        ])
        
        latex.append(" & ".join(row) + " \\\\")
        
        # Add hline for baseline or between different bal_types, cline between LRs
        if m.get('is_baseline', False):
            latex.append("\\hline")
        elif not m.get('is_baseline', False):
            bal_type_key = m['bal_type']
            lr_key = (m['bal_type'], m['lr'])
            if bal_type_counter.get(bal_type_key, 0) == bal_type_rows.get(bal_type_key, 0):
                latex.append("\\hline")
            elif lr_counter.get(lr_key, 0) == lr_rows.get(lr_key, 0):
                latex.append("\\cline{2-7}")
    
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    latex_str = "\n".join(latex)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(latex_str)
        print(f"LaTeX table saved to: {save_path}")
    
    print("\n" + latex_str + "\n")
    return latex_str

def plot_pareto_frontier(models_list, dataset_configs, results_root="results/winobias", save_path=None):
    """
    Create a Pareto frontier plot using Plotly with:
    - X-axis: Disparity (lower is better)
    - Y-axis: Global F1 (higher is better)
    - 4 symbols for model types (text_bal, text_imbal, triples_bal, triples_imbal)
    - 9 colors for LR x dataset_size combinations
    - Pareto frontier line showing non-dominated solutions
    """
    # Define symbols for each model type
    symbol_map = {
        'text_bal': 'circle',
        'text_imbal': 'square',
        'triples_bal': 'diamond',
        'triples_imbal': 'cross'
    }
    
    # Define colors for 9 combinations (3 LRs x 3 dataset sizes)
    # BALANCED_CONFIGS (actual sizes):
    #   5e-04: [1, 3, 4]
    #   5e-05: [5, 20, 50]
    #   2e-05: [20, 50, 100]
    # IMBALANCED_CONFIGS (actual sizes):
    #   5e-04: [8]
    #   5e-05: [40]
    #   2e-05: [200]
    # But for display, balanced sizes are doubled: [2, 6, 8], [10, 40, 100], [40, 100, 200]
    
    # Mapping from actual size to display size (for balanced only)
    balanced_size_mapping = {
        1: 2, 3: 6, 4: 8,           # 5e-04 sizes
        5: 10, 20: 40, 50: 100,     # 5e-05 sizes
        100: 200                     # 2e-05 sizes (20->40, 50->100 handled above)
    }
    
    color_map = {
        ('5e-04', 2): '#1f77b4',    # blue (size 2)
        ('5e-04', 6): '#ff7f0e',    # orange (size 6)
        ('5e-04', 8): '#2ca02c',    # green (size 8 - balanced 4->8 or imbalanced 8)
        ('5e-05', 10): '#d62728',   # red (size 10)
        ('5e-05', 40): '#9467bd',   # purple (size 40 - balanced 20->40 or imbalanced 40)
        ('5e-05', 100): '#8c564b',  # brown (size 100)
        ('2e-05', 40): '#e377c2',   # pink (size 40)
        ('2e-05', 100): '#7f7f7f',  # gray (size 100)
        ('2e-05', 200): '#bcbd22',  # yellow-green (size 200 - balanced 100->200 or imbalanced 200)
    }
    
    # Collect metrics
    plot_data = []
    for model in models_list:
        metrics = compute_global_metrics(model, dataset_configs, results_root)
        if metrics and 'Baseline' not in metrics['display_name']:
            # Determine model type
            has_text = '_text_' in model
            # Check for imbal carefully (not just 'bal' substring)
            has_imbal = 'imbal' in model and not model.replace('imbal', '').endswith('bal')
            
            if has_text and has_imbal:
                model_type = 'text_imbal'
            elif has_text:
                model_type = 'text_bal'
            elif has_imbal:
                model_type = 'triples_imbal'
            else:
                model_type = 'triples_bal'
            
            # Extract LR and dataset size
            match = re.search(r'(\d+)(bal|imbal).*?(?:_([0-9e\-]+))?$', model)
            if match:
                bal_num = int(match.group(1))
                lr = match.group(3) if match.group(3) else '5e-4'
                
                # Normalize LR
                if lr == '5e-4':
                    lr = '5e-04'
                
                # For balanced models, double the size for display
                display_size = bal_num
                if model_type in ['text_bal', 'triples_bal']:
                    display_size = balanced_size_mapping.get(bal_num, bal_num * 2)
                
                plot_data.append({
                    'model_name': model,
                    'display_name': metrics['display_name'],
                    'disparity': metrics['disparity'],
                    'global_f1': metrics['global_f1'],
                    'model_type': model_type,
                    'lr': lr,
                    'bal_num': bal_num,
                    'display_size': display_size
                })
    
    # Compute Pareto frontier
    # A point is on the Pareto frontier if no other point dominates it
    # Point A dominates point B if: A has lower disparity AND higher F1
    pareto_points = []
    for i, point in enumerate(plot_data):
        is_dominated = False
        for j, other in enumerate(plot_data):
            if i != j:
                # other dominates point if it has lower disparity AND higher/equal F1
                # OR equal disparity AND higher F1
                if (other['disparity'] < point['disparity'] and other['global_f1'] >= point['global_f1']) or \
                   (other['disparity'] <= point['disparity'] and other['global_f1'] > point['global_f1']):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_points.append(point)
    
    # Sort Pareto points by disparity for drawing the frontier line
    pareto_points_sorted = sorted(pareto_points, key=lambda x: x['disparity'])
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add Pareto frontier line
    if len(pareto_points_sorted) > 1:
        fig.add_trace(go.Scatter(
            x=[p['disparity'] for p in pareto_points_sorted],
            y=[p['global_f1'] for p in pareto_points_sorted],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Pareto Frontier',
            showlegend=True
        ))
    
    # Add traces for each data point
    for data_point in plot_data:
        model_type = data_point['model_type']
        lr = data_point['lr']
        display_size = data_point['display_size']
        
        symbol = symbol_map[model_type]
        color = color_map.get((lr, display_size), '#000000')
        
        # Check if this point is on the Pareto frontier
        is_pareto = data_point in pareto_points
        marker_size = 18 if is_pareto else 14
        
        fig.add_trace(go.Scatter(
            x=[data_point['disparity']],
            y=[data_point['global_f1']],
            mode='markers',
            marker=dict(
                symbol=symbol,
                size=marker_size,
                color=color,
                line=dict(width=2 if is_pareto else 1, color='red' if is_pareto else 'black')
            ),
            name=data_point['display_name'],
            hovertemplate='<b>%{fullData.name}</b><br>Disparity: %{x:.2f}<br>Global F1: %{y:.2f}<extra></extra>',
            showlegend=False
        ))
    
    # Add baseline if exists
    for model in models_list:
        metrics = compute_global_metrics(model, dataset_configs, results_root)
        if metrics and 'Baseline' in metrics['display_name']:
            fig.add_trace(go.Scatter(
                x=[metrics['disparity']],
                y=[metrics['global_f1']],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=20,
                    color='gold',
                    line=dict(width=2, color='black')
                ),
                name='Baseline',
                text=metrics['display_name'],
                hovertemplate='<b>%{text}</b><br>Disparity: %{x:.2f}<br>Global F1: %{y:.2f}<extra></extra>',
                showlegend=True
            ))
    
    # Create custom legend annotations
    # Symbol legend
    symbol_legend_items = [
        ('circle', 'Text + Balanced', 0.02, 0.98),
        ('square', 'Text + Imbalanced', 0.02, 0.93),
        ('diamond', 'Triples + Balanced', 0.02, 0.88),
        ('cross', 'Triples + Imbalanced', 0.02, 0.83)
    ]
    
    # Color legend - organized by LR and dataset size
    color_legend_items = [
        ('#1f77b4', '5e-04, Size=2', 0.02, 0.73),
        ('#ff7f0e', '5e-04, Size=6', 0.02, 0.68),
        ('#2ca02c', '5e-04, Size=8', 0.02, 0.63),
        ('#d62728', '5e-05, Size=10', 0.02, 0.58),
        ('#9467bd', '5e-05, Size=40', 0.02, 0.53),
        ('#8c564b', '5e-05, Size=100', 0.02, 0.48),
        ('#e377c2', '2e-05, Size=40', 0.02, 0.43),
        ('#7f7f7f', '2e-05, Size=100', 0.02, 0.38),
        ('#bcbd22', '2e-05, Size=200', 0.02, 0.33),
    ]
    
    
    # Add invisible traces for symbol legend (to appear in legend)
    for symbol, label, _, _ in symbol_legend_items:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(symbol=symbol, size=10, color='gray', line=dict(width=1, color='black')),
            name=label,
            showlegend=True
        ))
    
    # Add invisible traces for color legend (to appear in legend)
    for color, label, _, _ in color_legend_items:
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(symbol='circle', size=10, color=color, line=dict(width=1, color='black')),
            name=label,
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title='Pareto Frontier: Global F1 vs Bias Disparity',
        xaxis_title='Bias Disparity (lower is better)',
        yaxis_title='Global F1 Score (higher is better)',
        hovermode='closest',
        width=1200,
        height=800,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.02,
            font=dict(size=12)
        ),
        plot_bgcolor='rgba(240,240,240,0.5)',
        xaxis=dict(gridcolor='white', showgrid=True),
        yaxis=dict(gridcolor='white', showgrid=True)
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Pareto frontier plot saved to: {save_path}")
    
    fig.show()
    return fig

def print_s_score_rankings(models_list, dataset_configs, lambda_val=1.0, results_root="results/winobias"):
    """
    Calculate and print S = Global F1 - lambda * Disparity for all models.
    Sort by S score (higher is better).
    
    Args:
        models_list: List of model names to evaluate
        dataset_configs: Dataset configurations to use
        lambda_val: Weight parameter for disparity penalty (default=1.0)
        results_root: Root directory for results
    """
    print(f"\n{'='*80}")
    print(f"S-Score Rankings (S = Global F1 - {lambda_val} * Disparity)")
    print(f"{'='*80}\n")
    
    # Mapping from actual size to display size (for balanced only)
    balanced_size_mapping = {
        1: 2, 3: 6, 4: 8,           # 5e-04 sizes
        5: 10, 20: 40, 50: 100,     # 5e-05 sizes
        100: 200                     # 2e-05 sizes (20->40, 50->100 handled above)
    }
    
    # Collect metrics for all models
    rankings = []
    for model in models_list:
        metrics = compute_global_metrics(model, dataset_configs, results_root)
        if metrics and 'Baseline' not in metrics['display_name']:
            # Determine model characteristics
            has_text = '_text_' in model
            has_imbal = 'imbal' in model and not model.replace('imbal', '').endswith('bal')
            
            if has_text and has_imbal:
                model_type = 'Text + Imbalanced'
                is_balanced = False
            elif has_text:
                model_type = 'Text + Balanced'
                is_balanced = True
            elif has_imbal:
                model_type = 'Triples + Imbalanced'
                is_balanced = False
            else:
                model_type = 'Triples + Balanced'
                is_balanced = True
            
            # Extract LR and dataset size
            match = re.search(r'(\d+)(bal|imbal).*?(?:_([0-9e\-]+))?$', model)
            if match:
                dataset_size = int(match.group(1))
                lr = match.group(3) if match.group(3) else '5e-4'
                
                # Normalize LR
                if lr == '5e-4':
                    lr = '5e-04'
                
                # For balanced models, double the size for display
                display_size = dataset_size
                if is_balanced:
                    display_size = balanced_size_mapping.get(dataset_size, dataset_size * 2)
                
                # Calculate S score
                s_score = metrics['global_f1'] - lambda_val * metrics['disparity']
                
                rankings.append({
                    'model_type': model_type,
                    'lr': lr,
                    'dataset_size': display_size,
                    'global_f1': metrics['global_f1'],
                    'disparity': metrics['disparity'],
                    's_score': s_score,
                    'display_name': metrics['display_name']
                })
    
    # Sort by S score (descending - higher is better)
    rankings.sort(key=lambda x: x['s_score'], reverse=True)
    
    # Print header
    print(f"{'Rank':<6} {'Type':<25} {'LR':<10} {'Size':<8} {'F1':<8} {'Disp':<8} {'S-Score':<10}")
    print(f"{'-'*80}")
    
    # Print rankings
    for i, entry in enumerate(rankings, 1):
        print(f"{i:<6} {entry['model_type']:<25} {entry['lr']:<10} {entry['dataset_size']:<8} "
              f"{entry['global_f1']:<8.2f} {entry['disparity']:<8.2f} {entry['s_score']:<10.2f}")
    
    print(f"\n{'='*80}\n")
    
    # Print baseline for reference
    for model in models_list:
        metrics = compute_global_metrics(model, dataset_configs, results_root)
        if metrics and 'Baseline' in metrics['display_name']:
            baseline_s = metrics['global_f1'] - lambda_val * metrics['disparity']
            print(f"Baseline: F1={metrics['global_f1']:.2f}, Disparity={metrics['disparity']:.2f}, S-Score={baseline_s:.2f}\n")
            break
    
    return rankings

if __name__ == "__main__":
    dataset_configs = ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]
    '''
    t5_models = ["google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl"]
    t5_models = ["google/flan-t5-base", "pretraining_finetuned_flant5_base_100bal_full", "pretraining_finetuned_flant5_base_100bal_lora", "pretraining_finetuned_flant5_base_20bal_lora", "pretraining_finetuned_flant5_base_20bal_full", "pretraining_finetuned_flant5_base_5bal_full"]
    t5_models = ["google_flan-t5-base", "pretraining_finetuned_1000c4_flant5_base_5bal_full"]
    t5_models = ["google_flan-t5-base", "pretraining_finetuned_text_flant5_base_5bal_full", "pretraining_finetuned_text_flant5_base_3bal_lora","pretraining_finetuned_text_flant5_base_4bal_lora", "pretraining_finetuned_text_flant5_base_5bal_lora", "pretraining_finetuned_text_flant5_base_6bal_lora", "pretraining_finetuned_text_flant5_base_7bal_lora", "pretraining_finetuned_text_flant5_base_10bal_lora"]
    t5_models = ["google_flan-t5-base", "pretraining_finetuned_text_flant5_base_50bal_lora_5e-05", "pretraining_finetuned_text_flant5_base_50bal_lora_2e-05"]
    t5_models = ["google_flan-t5-base", "pretraining_finetuned_text_flant5_base_20bal_lora_2e-05", "pretraining_finetuned_text_flant5_base_50bal_lora_2e-05", "pretraining_finetuned_text_flant5_base_100bal_lora_2e-05", "pretraining_finetuned_text_flant5_base_200bal_lora_2e-05"]
    
    # t5_models = ["google_flan-t5-base", "pretraining_finetuned_flant5_base_1bal_lora_5e-04","pretraining_finetuned_flant5_base_3bal_lora_5e-04","pretraining_finetuned_flant5_base_4bal_lora_5e-04"]
    
    # 2e-05 text
    t5_models = ["google_flan-t5-base", "pretraining_finetuned_text_flant5_base_20bal_lora_2e-05", "pretraining_finetuned_text_flant5_base_50bal_lora_2e-05", "pretraining_finetuned_text_flant5_base_100bal_lora_2e-05", "pretraining_finetuned_text_flant5_base_200bal_lora_2e-05"]
    # 5e-05 text
    t5_models = ["google_flan-t5-base", "pretraining_finetuned_text_flant5_base_5bal_lora_5e-05", "pretraining_finetuned_text_flant5_base_10bal_lora_5e-05","pretraining_finetuned_text_flant5_base_15bal_lora_5e-05","pretraining_finetuned_text_flant5_base_20bal_lora_5e-05","pretraining_finetuned_text_flant5_base_50bal_lora_5e-05", "pretraining_finetuned_text_flant5_base_100bal_lora_5e-05"]
    # 5e-04 text
    t5_models = ["google_flan-t5-base", "pretraining_finetuned_text_flant5_base_1bal_lora_5e-04", "pretraining_finetuned_text_flant5_base_2bal_lora_5e-04","pretraining_finetuned_text_flant5_base_3bal_lora_5e-04","pretraining_finetuned_text_flant5_base_4bal_lora_5e-04","pretraining_finetuned_text_flant5_base_5bal_lora_5e-04", "pretraining_finetuned_text_flant5_base_6bal_lora_5e-04"]
    '''    

    #==================================================================================================================
    base_model = ["google_flan-t5-base"]
    #==================================================================================================================
    # TABELA 1 - TEXT BALANCED
    models_2e_05 = ["pretraining_finetuned_text_flant5_base_20bal_lora_2e-05", "pretraining_finetuned_text_flant5_base_50bal_lora_2e-05", "pretraining_finetuned_text_flant5_base_100bal_lora_2e-05", "pretraining_finetuned_text_flant5_base_200bal_lora_2e-05"]
    models_5e_05 = ["pretraining_finetuned_text_flant5_base_5bal_lora_5e-05","pretraining_finetuned_text_flant5_base_20bal_lora_5e-05","pretraining_finetuned_text_flant5_base_50bal_lora_5e-05", "pretraining_finetuned_text_flant5_base_100bal_lora_5e-05"]
    models_5e_04 = ["pretraining_finetuned_text_flant5_base_1bal_lora_5e-04","pretraining_finetuned_text_flant5_base_3bal_lora_5e-04","pretraining_finetuned_text_flant5_base_4bal_lora_5e-04","pretraining_finetuned_text_flant5_base_5bal_lora_5e-04"]
    table1_models = base_model + models_2e_05 + models_5e_05 + models_5e_04
    # TABELA 1.1 - TEXT BALANCED (No extra sizes)
    models_2e_05 = ["pretraining_finetuned_text_flant5_base_20bal_lora_2e-05", "pretraining_finetuned_text_flant5_base_50bal_lora_2e-05", "pretraining_finetuned_text_flant5_base_100bal_lora_2e-05"]
    models_5e_05 = ["pretraining_finetuned_text_flant5_base_5bal_lora_5e-05","pretraining_finetuned_text_flant5_base_20bal_lora_5e-05","pretraining_finetuned_text_flant5_base_50bal_lora_5e-05"]
    models_5e_04 = ["pretraining_finetuned_text_flant5_base_1bal_lora_5e-04","pretraining_finetuned_text_flant5_base_3bal_lora_5e-04","pretraining_finetuned_text_flant5_base_4bal_lora_5e-04"]
    table1_1_models = base_model + models_2e_05 + models_5e_05 + models_5e_04
    #==================================================================================================================
    # TABELA 2 - TRIPLES BALANCED
    models_2e_05 = ["pretraining_finetuned_flant5_base_20bal_lora_2e-05", "pretraining_finetuned_flant5_base_50bal_lora_2e-05", "pretraining_finetuned_flant5_base_100bal_lora_2e-05"]
    models_5e_05 = ["pretraining_finetuned_flant5_base_5bal_lora_5e-05","pretraining_finetuned_flant5_base_20bal_lora_5e-05","pretraining_finetuned_flant5_base_50bal_lora_5e-05"]
    models_5e_04 = ["pretraining_finetuned_flant5_base_1bal_lora_5e-04","pretraining_finetuned_flant5_base_3bal_lora_5e-04","pretraining_finetuned_flant5_base_4bal_lora_5e-04"]
    table2_models = base_model + models_2e_05 + models_5e_05 + models_5e_04
    #==================================================================================================================
    # TABELA 3 - TEXT BALANCED+IMBALANCED
    models_2e_05 = ["pretraining_finetuned_text_flant5_base_100bal_lora_2e-05", "pretraining_finetuned_text_flant5_base_200imbal_lora_2e-05"]
    models_5e_05 = ["pretraining_finetuned_text_flant5_base_20bal_lora_5e-05", "pretraining_finetuned_text_flant5_base_40imbal_lora_5e-05"]
    models_5e_04 = ["pretraining_finetuned_text_flant5_base_4bal_lora_5e-04", "pretraining_finetuned_text_flant5_base_8imbal_lora_5e-04"]
    table3_models = base_model + models_2e_05 + models_5e_05 + models_5e_04
    #==================================================================================================================
    # TABELA 4 - TRIPLES BALANCED+IMBALANCED
    models_2e_05 = ["pretraining_finetuned_flant5_base_100bal_lora_2e-05", "pretraining_finetuned_flant5_base_200imbal_lora_2e-05"]
    models_5e_05 = ["pretraining_finetuned_flant5_base_20bal_lora_5e-05", "pretraining_finetuned_flant5_base_40imbal_lora_5e-05"]
    models_5e_04 = ["pretraining_finetuned_flant5_base_4bal_lora_5e-04", "pretraining_finetuned_flant5_base_8imbal_lora_5e-04"]
    table4_models = base_model + models_2e_05 + models_5e_05 + models_5e_04
    #==================================================================================================================

    #IMBALANCED
    # t5_models1 = ["google_flan-t5-base", "pretraining_finetuned_text_flant5_base_8imbal_lora_5e-04", "pretraining_finetuned_text_flant5_base_40imbal_lora_5e-05", "pretraining_finetuned_text_flant5_base_100imbal_lora_5e-05", "pretraining_finetuned_text_flant5_base_200imbal_lora_2e-05"]
    # t5_models2 = ["google_flan-t5-base", "pretraining_finetuned_text_flant5_base_4bal_lora_5e-04", "pretraining_finetuned_text_flant5_base_20bal_lora_5e-05", "pretraining_finetuned_text_flant5_base_50bal_lora_5e-05", "pretraining_finetuned_text_flant5_base_100bal_lora_2e-05"]
    
    t5_models1 = ["google_flan-t5-base", "pretraining_finetuned_flant5_base_8imbal_lora_5e-04", "pretraining_finetuned_flant5_base_40imbal_lora_5e-05", "pretraining_finetuned_flant5_base_100imbal_lora_5e-05", "pretraining_finetuned_flant5_base_200imbal_lora_2e-05"]
    t5_models2 = ["google_flan-t5-base", "pretraining_finetuned_flant5_base_4bal_lora_5e-04", "pretraining_finetuned_flant5_base_20bal_lora_5e-05", "pretraining_finetuned_flant5_base_50bal_lora_5e-05", "pretraining_finetuned_flant5_base_100bal_lora_2e-05"]
    
    t5_models = t5_models1 + t5_models2[1:] 
    # t5_models = ["google_flan-t5-base", "pretraining_finetuned_flant5_base_5bal_lora_5e-05","pretraining_finetuned_flant5_base_20bal_lora_5e-05","pretraining_finetuned_flant5_base_50bal_lora_5e-05"]
    # t5_models = ["google_flan-t5-base", "pretraining_finetuned_flant5_base_20bal_lora_2e-05","pretraining_finetuned_flant5_base_50bal_lora_2e-05","pretraining_finetuned_flant5_base_100bal_lora_2e-05"]


    # falcon_models = ["tiiuae/Falcon-H1-1.5B-Deep-Instruct"]
    llama_models = ["meta-llama_Llama-3.2-1B-Instruct_base", "finedtuned_llama32_200_imb", "pretraining_finedtuned_llama32_20bal", "pretraining_finedtuned_llama32_100bal"]

    # results_root="winobias_fewshot"
    # results_root="winobias_c4"
    results_root="winobias"
    
    # Interleave table1_1_models and table2_models
    interleaved = [item for pair in zip(table1_1_models, table2_models) for item in pair]
    # Add any remaining items if lists have different lengths
    if len(table1_1_models) > len(table2_models):
        interleaved.extend(table1_1_models[len(table2_models):])
    elif len(table2_models) > len(table1_1_models):
        interleaved.extend(table2_models[len(table1_1_models):])
    TABLES = [interleaved]


    # Generate individual plots per model
    # for model_name in t5_models:
    '''
    for table in TABLES:
        for model_name in table:
            merged_data = load_and_merge_csvs(model_name, dataset_configs, results_root=f"results/{results_root}")
            plot_f1_scores(merged_data, model_name, save_path=f"results/{results_root}/{model_name.replace('/', '_')}/{model_name.replace('/', '_')}_f1_plot.png")

        # Generate comparison bar plot
        plot_comparison_bars(table, dataset_configs, results_root=f"results/{results_root}", 
                            save_path=f"results/{results_root}/comparison_bar_plot.png")
        
        # Generate LaTeX table
        generate_latex_table(table, dataset_configs, results_root=f"results/{results_root}",
                            save_path=f"results/{results_root}/comparison_table.tex")
    '''
    # Generate Pareto frontier plot with all models from table1, table2, table3, table4
    all_models = list(set(table1_1_models + table2_models[1:] + table3_models[1:] + table4_models[1:]))
    plot_pareto_frontier(all_models, dataset_configs, results_root=f"results/{results_root}",
                        save_path=f"results/{results_root}/pareto_frontier.html")
    
    # Print S-Score rankings for different lambda values
    print("\n" + "="*80)
    print("S-SCORE ANALYSIS FOR DIFFERENT LAMBDA VALUES")
    print("="*80)
    
    for lambda_val in [0.5, 1.0, 2.0]:
        print_s_score_rankings(all_models, dataset_configs, lambda_val=lambda_val, 
                              results_root=f"results/{results_root}")

    # for model_name in llama_models:
    #     merged_data = load_and_merge_csvs(model_name, dataset_configs)
    #     plot_f1_scores(merged_data, model_name, save_path=f"results/winobias/{model_name.replace('/', '_')}/{model_name.replace('/', '_')}_f1_plot.png")