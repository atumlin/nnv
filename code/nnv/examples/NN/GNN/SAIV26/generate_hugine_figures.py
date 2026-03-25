#!/usr/bin/env python3
"""Generate verification figures and LaTeX tables for HuGINE experiments.

Reads CSV output from run_hugine_full_experiments.m and produces:
  - Stacked area plots (verified/unknown/violated vs epsilon)
  - Timing bar charts
  - LaTeX timing table
  - LaTeX edge comparison table

Usage:
    python generate_hugine_figures.py results/hugine_*/hugine_results.csv -o figures/
    python generate_hugine_figures.py results.csv -o figures/ --latex
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── Data Loading ─────────────────────────────────────────────────────────

def load_csv(csv_path):
    """Load HuGINE results CSV into list of dicts."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = {
                'task': row['Task'].upper(),
                'mode': row['Mode'],
                'grid': row['Grid'],
                'node_eps': float(row['Node_Epsilon']),
                'edge_eps': row['Edge_Epsilon'],
                'n_total': int(row['Total_Graphs']),
                'n_safe': int(row['Safe_Graphs']),
                'avg_time': float(row['Avg_Time_s']),
                'verified': int(row['Total_Verified']),
                'unknown': int(row['Total_Unknown']),
                'violated': int(row['Total_Violated']),
                'total_nodes': int(row['Total_Voltage_Nodes']),
                'pct_verified': float(row['Pct_Verified']),
                'mean_width': float(row['Mean_Bound_Width']),
                'max_width': float(row['Max_Bound_Width']),
            }
            if r['edge_eps'] != 'N/A':
                r['edge_eps'] = float(r['edge_eps'])
            else:
                r['edge_eps'] = None
            results.append(r)
    return results


def filter_results(results, task=None, mode=None, grid=None, edge_eps=None):
    """Filter results by criteria."""
    out = results
    if task:
        out = [r for r in out if r['task'] == task.upper()]
    if mode:
        out = [r for r in out if r['mode'] == mode]
    if grid:
        out = [r for r in out if r['grid'] == grid]
    if edge_eps is not None:
        if edge_eps == 'none':
            out = [r for r in out if r['edge_eps'] is None]
        else:
            out = [r for r in out if r['edge_eps'] == edge_eps]
    return out


# ── Stacked Area Plots ──────────────────────────────────────────────────

GRID_ORDER = ['ieee24', 'ieee39', 'ieee118']
GRID_LABELS = {
    'ieee24': 'IEEE-24\n$V \\in [0.95, 1.05]$ p.u.',
    'ieee39': 'IEEE-39\n$V \\in [0.94, 1.06]$ p.u.',
    'ieee118': 'IEEE-118\n$V \\in [0.94, 1.09]$ p.u.',
}
COLORS = {'verified': '#2ecc71', 'unknown': '#f39c12', 'violated': '#e74c3c'}

# Global font sizes for publication-quality figures
FONTSIZE_TITLE = 16
FONTSIZE_AXIS_LABEL = 14
FONTSIZE_TICK = 12
FONTSIZE_LEGEND = 12


def format_eps(eps):
    """Format epsilon for axis label."""
    exp = np.log10(eps)
    if abs(exp - round(exp)) < 1e-9:
        return f'$10^{{{int(round(exp))}}}$'
    return f'{eps:.0e}'


def generate_stacked_area(results, task, output_dir):
    """Generate 1x3 stacked area plot for a task."""
    node_only = filter_results(results, task=task, mode='node_only')
    if not node_only:
        return

    grids_present = sorted(set(r['grid'] for r in node_only),
                           key=lambda g: GRID_ORDER.index(g) if g in GRID_ORDER else 99)

    fig, axes = plt.subplots(1, len(grids_present), figsize=(5 * len(grids_present), 4),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, grid in zip(axes, grids_present):
        grid_data = sorted(filter_results(node_only, grid=grid), key=lambda r: r['node_eps'])
        if not grid_data:
            continue

        eps_vals = [r['node_eps'] for r in grid_data]
        pct_v = [r['pct_verified'] for r in grid_data]
        pct_u = [100 * r['unknown'] / max(1, r['total_nodes']) for r in grid_data]
        pct_x = [100 * r['violated'] / max(1, r['total_nodes']) for r in grid_data]

        x = range(len(eps_vals))
        ax.fill_between(x, 0, pct_v, color=COLORS['verified'], alpha=0.8, label='Verified')
        ax.fill_between(x, pct_v, [v + u for v, u in zip(pct_v, pct_u)],
                        color=COLORS['unknown'], alpha=0.8, label='Unknown')
        ax.fill_between(x, [v + u for v, u in zip(pct_v, pct_u)],
                        [v + u + vx for v, u, vx in zip(pct_v, pct_u, pct_x)],
                        color=COLORS['violated'], alpha=0.8, label='Violated')

        ax.set_xticks(x)
        ax.set_xticklabels([format_eps(e) for e in eps_vals], fontsize=FONTSIZE_TICK)
        ax.set_xlabel('Perturbation Level ($\\epsilon$)', fontsize=FONTSIZE_AXIS_LABEL)
        ax.set_ylim(0, 100)
        ax.set_title(GRID_LABELS.get(grid, grid), fontweight='bold', fontsize=FONTSIZE_TITLE)
        ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)

    axes[0].set_ylabel('Percentage (%)', fontsize=FONTSIZE_AXIS_LABEL)

    # Horizontal legend below x-axis labels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=FONTSIZE_LEGEND, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()

    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'hugine_stacked_{task.lower()}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: hugine_stacked_{task.lower()}.{{pdf,png}}')


# ── Timing Bar Charts ───────────────────────────────────────────────────

def generate_timing_bars(results, task, output_dir):
    """Generate grouped bar chart of timing per grid."""
    node_only = filter_results(results, task=task, mode='node_only')
    if not node_only:
        return

    grids_present = sorted(set(r['grid'] for r in node_only),
                           key=lambda g: GRID_ORDER.index(g) if g in GRID_ORDER else 99)
    eps_vals = sorted(set(r['node_eps'] for r in node_only))

    fig, ax = plt.subplots(figsize=(8, 5))
    n_grids = len(grids_present)
    n_eps = len(eps_vals)
    width = 0.8 / n_grids
    grid_colors = ['#3498db', '#e67e22', '#27ae60']

    for gi, grid in enumerate(grids_present):
        times = []
        for eps in eps_vals:
            matches = [r for r in node_only if r['grid'] == grid and r['node_eps'] == eps]
            times.append(matches[0]['avg_time'] if matches else 0)

        x = np.arange(n_eps) + gi * width - (n_grids - 1) * width / 2
        ax.bar(x, times, width, label=grid.upper().replace('IEEE', 'IEEE-'),
               color=grid_colors[gi % len(grid_colors)], alpha=0.85)

    ax.set_xticks(np.arange(n_eps))
    ax.set_xticklabels([format_eps(e) for e in eps_vals], fontsize=FONTSIZE_TICK)
    ax.set_xlabel('Perturbation Level ($\\epsilon$)', fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_ylabel('Time (s/graph)', fontsize=FONTSIZE_AXIS_LABEL)
    ax.set_yscale('log')
    ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)

    # Horizontal legend below x-axis labels
    ax.legend(loc='lower center', ncol=n_grids, fontsize=FONTSIZE_LEGEND,
              bbox_to_anchor=(0.5, -0.22))

    fig.tight_layout()

    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'hugine_timing_{task.lower()}.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: hugine_timing_{task.lower()}.{{pdf,png}}')


# ── Edge Comparison Plot ────────────────────────────────────────────────

def generate_edge_comparison(results, output_dir):
    """Generate node vs node+edge comparison bar chart (PF only)."""
    node_only = filter_results(results, task='pf', mode='node_only')
    node_edge = filter_results(results, task='pf', mode='node_edge')
    if not node_only or not node_edge:
        return

    grids_present = sorted(set(r['grid'] for r in node_only),
                           key=lambda g: GRID_ORDER.index(g) if g in GRID_ORDER else 99)
    eps_vals = sorted(set(r['node_eps'] for r in node_only))
    edge_eps_vals = sorted(set(r['edge_eps'] for r in node_edge if r['edge_eps'] is not None))

    fig, axes = plt.subplots(1, len(grids_present), figsize=(5 * len(grids_present), 4),
                             sharey=True, squeeze=False)
    axes = axes[0]

    for ax, grid in zip(axes, grids_present):
        n_eps = len(eps_vals)
        n_bars = 1 + len(edge_eps_vals)  # node-only + each edge eps
        width = 0.8 / n_bars
        colors = ['#3498db', '#e67e22', '#e74c3c']

        # Node-only bars
        pcts_node = []
        for eps in eps_vals:
            m = [r for r in node_only if r['grid'] == grid and r['node_eps'] == eps]
            pcts_node.append(m[0]['pct_verified'] if m else 0)

        x = np.arange(n_eps)
        ax.bar(x - (n_bars - 1) * width / 2, pcts_node, width,
               label='Node only', color=colors[0], alpha=0.85)

        # Edge perturbation bars
        for ei, edge_eps in enumerate(edge_eps_vals):
            pcts_edge = []
            for eps in eps_vals:
                m = [r for r in node_edge if r['grid'] == grid
                     and r['node_eps'] == eps and r['edge_eps'] == edge_eps]
                pcts_edge.append(m[0]['pct_verified'] if m else 0)

            offset = (ei + 1) * width - (n_bars - 1) * width / 2
            ax.bar(x + offset, pcts_edge, width,
                   label=f'+Edge {format_eps(edge_eps)}', color=colors[ei + 1], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([format_eps(e) for e in eps_vals], fontsize=FONTSIZE_TICK)
        ax.set_xlabel('Perturbation Level ($\\epsilon$)', fontsize=FONTSIZE_AXIS_LABEL)
        ax.set_title(GRID_LABELS.get(grid, grid).split('\n')[0], fontweight='bold',
                     fontsize=FONTSIZE_TITLE)
        ax.tick_params(axis='y', labelsize=FONTSIZE_TICK)

    axes[0].set_ylabel('Verified (%)', fontsize=FONTSIZE_AXIS_LABEL)

    # Horizontal legend below x-axis labels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels),
               fontsize=FONTSIZE_LEGEND, bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()

    for ext in ['pdf', 'png']:
        path = os.path.join(output_dir, f'hugine_edge_comparison.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: hugine_edge_comparison.{{pdf,png}}')


# ── LaTeX Tables ────────────────────────────────────────────────────────

def write_timing_latex(results, output_dir):
    """Write LaTeX timing table (s/graph)."""
    tasks = sorted(set(r['task'] for r in results if r['mode'] == 'node_only'))
    grids = sorted(set(r['grid'] for r in results if r['mode'] == 'node_only'),
                   key=lambda g: GRID_ORDER.index(g) if g in GRID_ORDER else 99)
    eps_vals = sorted(set(r['node_eps'] for r in results if r['mode'] == 'node_only'))

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{HuGINE verification timing (s/graph). IEEE-118 uses subgraph verification; IEEE-24/39 use full-graph with approx-star.}')
    lines.append(r'\label{tab:hugine_timing}')
    lines.append(r'\setlength{\tabcolsep}{14pt}')

    n_eps = len(eps_vals)
    col_spec = 'l l ' + ' '.join(['c'] * n_eps)
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    # Header
    lines.append(r'& & \multicolumn{' + str(n_eps) + r'}{c}{Time (s/graph) per $\epsilon_{\mathrm{node}}$} \\')
    lines.append(r'\cmidrule(lr){3-' + str(2 + n_eps) + '}')
    eps_headers = ' & '.join([f'$10^{{{int(np.log10(e))}}}$' for e in eps_vals])
    lines.append(r'\textbf{Task} & \textbf{System} & ' + eps_headers + r' \\')
    lines.append(r'\midrule')

    for ti, task in enumerate(tasks):
        if ti > 0:
            lines.append(r'\midrule')
        for gi, grid in enumerate(grids):
            prefix = r'\multirow{' + str(len(grids)) + r'}{*}{\textit{' + task + '}}' if gi == 0 else ''
            grid_label = grid.upper().replace('IEEE', 'IEEE-')

            vals = []
            for eps in eps_vals:
                m = [r for r in results if r['task'] == task.upper()
                     and r['mode'] == 'node_only' and r['grid'] == grid and r['node_eps'] == eps]
                if m:
                    t = m[0]['avg_time']
                    if t < 1:
                        vals.append(f'\\phantom{{0}}{t:.2f}')
                    elif t < 100:
                        vals.append(f'\\phantom{{0}}{t:.2f}')
                    else:
                        vals.append(f'{t:.2f}')
                else:
                    vals.append('--')

            lines.append(f'{prefix} & {grid_label} & ' + ' & '.join(vals) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    path = os.path.join(output_dir, 'hugine_timing_table.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved: hugine_timing_table.tex')


def write_edge_comparison_latex(results, output_dir):
    """Write LaTeX edge comparison table (PF only)."""
    node_only = filter_results(results, task='pf', mode='node_only')
    node_edge = filter_results(results, task='pf', mode='node_edge')
    if not node_only or not node_edge:
        return

    grids = sorted(set(r['grid'] for r in node_only),
                   key=lambda g: GRID_ORDER.index(g) if g in GRID_ORDER else 99)
    eps_vals = sorted(set(r['node_eps'] for r in node_only))
    edge_eps_vals = sorted(set(r['edge_eps'] for r in node_edge if r['edge_eps'] is not None))

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{Impact of edge perturbations on HuGINE verification (PF task). Robustness shows percentage of verified safe outputs; $\Delta = (\text{Node{+}Edge}) - (\text{Node-only})$. Time overhead in seconds per graph and multiplicative factor.}')
    lines.append(r'\label{tab:hugine_edge}')
    lines.append(r'\setlength{\tabcolsep}{4pt}')

    # Columns: System | eps_node | Node (%) | +Edge e1 (%) | Delta1 | +Edge e2 (%) | Delta2 | Node time | +Edge e1 time | +Edge e2 time
    n_edge = len(edge_eps_vals)
    robustness_cols = 1 + 2 * n_edge  # Node + (Edge + Delta) per edge_eps
    time_cols = 1 + n_edge  # Node + each edge_eps

    col_spec = 'l l ' + ' '.join(['c'] * (robustness_cols + time_cols))
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    # Header row 1
    lines.append(r'& & \multicolumn{' + str(robustness_cols) + r'}{c}{Robustness (\%)} & \multicolumn{' + str(time_cols) + r'}{c}{Time (s/graph)} \\')
    lines.append(r'\cmidrule(lr){3-' + str(2 + robustness_cols) + r'} \cmidrule(lr){' + str(3 + robustness_cols) + '-' + str(2 + robustness_cols + time_cols) + '}')

    # Header row 2
    header_parts = [r'\textbf{System}', r'$\epsilon_{\mathrm{node}}$', 'Node']
    for ee in edge_eps_vals:
        exp = int(np.log10(ee))
        header_parts.append(f'+Edge $10^{{{exp}}}$')
        header_parts.append(r'$\Delta$')
    header_parts.append('Node')
    for ee in edge_eps_vals:
        exp = int(np.log10(ee))
        header_parts.append(f'+Edge $10^{{{exp}}}$')
    lines.append(' & '.join(header_parts) + r' \\')
    lines.append(r'\midrule')

    for gi, grid in enumerate(grids):
        if gi > 0:
            lines.append(r'\midrule')

        grid_label = grid.upper().replace('IEEE', 'IEEE-')
        n_rows = len(eps_vals)

        for ei, eps in enumerate(eps_vals):
            prefix = r'\multirow{' + str(n_rows) + '}{*}{' + grid_label + '}' if ei == 0 else ''

            # Node-only
            m_node = [r for r in node_only if r['grid'] == grid and r['node_eps'] == eps]
            node_pct = m_node[0]['pct_verified'] if m_node else 0
            node_time = m_node[0]['avg_time'] if m_node else 0

            parts = [prefix, f'{eps:.0e}', f'{node_pct:.1f}']

            # Edge perturbation results
            for ee in edge_eps_vals:
                m_edge = [r for r in node_edge if r['grid'] == grid
                          and r['node_eps'] == eps and r['edge_eps'] == ee]
                if m_edge:
                    edge_pct = m_edge[0]['pct_verified']
                    edge_time = m_edge[0]['avg_time']
                    delta = edge_pct - node_pct
                    parts.append(f'{edge_pct:.1f}')
                    parts.append(f'${delta:+.1f}$')
                else:
                    parts.extend(['--', '--'])

            # Timing columns
            parts.append(f'{node_time:.2f}')
            for ee in edge_eps_vals:
                m_edge = [r for r in node_edge if r['grid'] == grid
                          and r['node_eps'] == eps and r['edge_eps'] == ee]
                if m_edge:
                    parts.append(f'{m_edge[0]["avg_time"]:.2f}')
                else:
                    parts.append('--')

            lines.append(' & '.join(parts) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    path = os.path.join(output_dir, 'hugine_edge_comparison_table.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved: hugine_edge_comparison_table.tex')


def write_robustness_latex(results, output_dir):
    """Write LaTeX robustness table (verified %)."""
    tasks = sorted(set(r['task'] for r in results if r['mode'] == 'node_only'))
    grids = sorted(set(r['grid'] for r in results if r['mode'] == 'node_only'),
                   key=lambda g: GRID_ORDER.index(g) if g in GRID_ORDER else 99)
    eps_vals = sorted(set(r['node_eps'] for r in results if r['mode'] == 'node_only'))

    lines = []
    lines.append(r'\begin{table}[t]')
    lines.append(r'\centering')
    lines.append(r'\caption{HuGINE verification robustness (\% verified safe) for node-only perturbation.}')
    lines.append(r'\label{tab:hugine_robustness}')
    lines.append(r'\setlength{\tabcolsep}{10pt}')

    n_eps = len(eps_vals)
    col_spec = 'l l ' + ' '.join(['c'] * n_eps)
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    eps_headers = ' & '.join([f'$10^{{{int(np.log10(e))}}}$' for e in eps_vals])
    lines.append(r'& & \multicolumn{' + str(n_eps) + r'}{c}{Verified (\%) per $\epsilon_{\mathrm{node}}$} \\')
    lines.append(r'\cmidrule(lr){3-' + str(2 + n_eps) + '}')
    lines.append(r'\textbf{Task} & \textbf{System} & ' + eps_headers + r' \\')
    lines.append(r'\midrule')

    for ti, task in enumerate(tasks):
        if ti > 0:
            lines.append(r'\midrule')
        for gi, grid in enumerate(grids):
            prefix = r'\multirow{' + str(len(grids)) + r'}{*}{\textit{' + task + '}}' if gi == 0 else ''
            grid_label = grid.upper().replace('IEEE', 'IEEE-')

            vals = []
            for eps in eps_vals:
                m = [r for r in results if r['task'] == task.upper()
                     and r['mode'] == 'node_only' and r['grid'] == grid and r['node_eps'] == eps]
                if m:
                    vals.append(f'{m[0]["pct_verified"]:.1f}')
                else:
                    vals.append('--')

            lines.append(f'{prefix} & {grid_label} & ' + ' & '.join(vals) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    path = os.path.join(output_dir, 'hugine_robustness_table.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  Saved: hugine_robustness_table.tex')


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate HuGINE verification figures and tables')
    parser.add_argument('csv_files', nargs='+', help='CSV result files from run_hugine_full_experiments.m')
    parser.add_argument('-o', '--output', default='results/figures/hugine', help='Output directory')
    parser.add_argument('--latex', action='store_true', help='Generate LaTeX tables')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load all CSVs
    all_results = []
    for csv_path in args.csv_files:
        if not os.path.isfile(csv_path):
            print(f'WARNING: {csv_path} not found, skipping')
            continue
        results = load_csv(csv_path)
        all_results.extend(results)
        print(f'Loaded {len(results)} entries from {csv_path}')

    if not all_results:
        print('No results loaded. Exiting.')
        sys.exit(1)

    print(f'\nTotal entries: {len(all_results)}')
    tasks = sorted(set(r['task'] for r in all_results))
    print(f'Tasks: {tasks}')

    if not args.no_figures:
        print('\nGenerating figures...')
        for task in tasks:
            generate_stacked_area(all_results, task, args.output)
            generate_timing_bars(all_results, task, args.output)

        generate_edge_comparison(all_results, args.output)

    if args.latex:
        print('\nGenerating LaTeX tables...')
        write_timing_latex(all_results, args.output)
        write_robustness_latex(all_results, args.output)
        write_edge_comparison_latex(all_results, args.output)

    print(f'\nAll outputs saved to: {args.output}')


if __name__ == '__main__':
    main()
