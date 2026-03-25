#!/usr/bin/env python3
"""
Figure generation for GNNV PF/OPF verification results.

Parses results from log files and generates publication-quality
stacked area plots showing verified/unknown/violated breakdown
across epsilon values for each grid size.

Author: Anne Tumlin
Date: 03/17/2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import argparse
from pathlib import Path


# Publication style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']


def parse_log_file(log_path):
    """Parse a verification log file and extract results.

    Returns a list of dicts with keys:
        task, grid, arch, eps, total, verified, unknown, violated, avg_time
    """
    results = []
    with open(log_path, 'r') as f:
        lines = f.readlines()

    header_pattern = re.compile(
        r'--- (\w+) (IEEE\d+) (.+?), eps=([\d.]+) \((\d+) graphs, (\d+) safe\) ---'
    )
    summary_pattern = re.compile(
        r'Summary: verified=(\d+)/(\d+) \(([\d.]+)%\), '
        r'unknown=(\d+), violated=(\d+), avg_time=([\d.]+)s'
    )

    # Track epsilon headers inline (handles multiple run_all_experiments in one log)
    actual_eps_list = None
    eps_counter = {}

    current_header = None
    for line in lines:
        # Update actual_eps_list when we see a new header
        eps_header_match = re.search(r'Node epsilon values:\s*\[(.*?)\]', line)
        if eps_header_match:
            actual_eps_list = [float(x) for x in eps_header_match.group(1).split()]
            eps_counter = {}  # reset counter for new run block

        hm = header_pattern.search(line)
        if hm:
            parsed_eps = float(hm.group(4))
            task_key = hm.group(1).lower()
            grid_key = hm.group(2).lower()
            arch_key = hm.group(3).strip()

            # Resolve truncated eps=0.000 using current actual epsilon list
            if parsed_eps == 0.0 and actual_eps_list is not None:
                counter_key = (task_key, grid_key, arch_key)
                count = eps_counter.get(counter_key, 0)
                small_eps = [e for e in actual_eps_list if e < 0.0005]
                if count < len(small_eps):
                    parsed_eps = small_eps[count]
                eps_counter[counter_key] = count + 1

            current_header = {
                'task': task_key,
                'grid': grid_key,
                'arch': arch_key,
                'eps': parsed_eps,
                'num_graphs': int(hm.group(5)),
            }
            continue

        sm = summary_pattern.search(line)
        if sm and current_header is not None:
            total = int(sm.group(2))
            verified = int(sm.group(1))
            unknown = int(sm.group(4))
            violated = int(sm.group(5))
            avg_time = float(sm.group(6))

            results.append({
                **current_header,
                'total': total,
                'verified': verified,
                'unknown': unknown,
                'violated': violated,
                'verified_pct': 100.0 * verified / total if total > 0 else 0,
                'unknown_pct': 100.0 * unknown / total if total > 0 else 0,
                'violated_pct': 100.0 * violated / total if total > 0 else 0,
                'avg_time': avg_time,
            })
            current_header = None

    return results


def filter_results(results, task=None, arch=None, grid=None):
    """Filter results by task, architecture, and/or grid."""
    out = results
    if task:
        out = [r for r in out if r['task'] == task.lower()]
    if arch:
        out = [r for r in out if r['arch'] == arch]
    if grid:
        out = [r for r in out if r['grid'] == grid.lower()]
    return out


def generate_stacked_area_figure(results, task, arch, output_dir, grids=None):
    """Generate a 1x3 stacked area figure for a given task and architecture.

    Each subplot is one grid (IEEE-24, IEEE-39, IEEE-118).
    Y-axis: percentage breakdown (verified/unknown/violated).
    X-axis: epsilon values.
    """
    if grids is None:
        grids = ['ieee24', 'ieee39', 'ieee118']

    grid_labels = {
        'ieee24': 'IEEE-24\n$V \\in [0.95, 1.05]$ p.u.',
        'ieee39': 'IEEE-39\n$V \\in [0.94, 1.06]$ p.u.',
        'ieee118': 'IEEE-118\n$V \\in [0.94, 1.09]$ p.u.',
    }

    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # green, orange, red
    labels = ['Verified', 'Unknown', 'Violated']

    fig, axes = plt.subplots(1, len(grids), figsize=(4.5 * len(grids), 4.5))
    if len(grids) == 1:
        axes = [axes]

    # Expected epsilons for padding missing data points
    expected_eps = [1e-5, 1e-4, 1e-3, 1e-2]

    for ax, grid in zip(axes, grids):
        data = filter_results(results, task=task, arch=arch, grid=grid)
        if not data:
            ax.set_title(f'{grid_labels.get(grid, grid)}\n(no data)')
            continue

        data = sorted(data, key=lambda r: r['eps'])
        # Filter to only expected epsilons
        data = [r for r in data if any(abs(r['eps'] - e) / max(e, 1e-15) < 0.01 for e in expected_eps)]
        epsilons = [r['eps'] for r in data]
        verified = [r['verified_pct'] for r in data]
        unknown = [r['unknown_pct'] for r in data]
        violated = [r['violated_pct'] for r in data]

        # Pad missing epsilons with 100% unknown (intractable)
        for eps in expected_eps:
            if not any(abs(e - eps) / max(eps, 1e-15) < 0.01 for e in epsilons):
                epsilons.append(eps)
                verified.append(0.0)
                unknown.append(100.0)
                violated.append(0.0)
        # Re-sort after padding
        order = np.argsort(epsilons)
        epsilons = [epsilons[i] for i in order]
        verified = [verified[i] for i in order]
        unknown = [unknown[i] for i in order]
        violated = [violated[i] for i in order]

        x = np.arange(len(epsilons))

        # Stack bottom-up: verified, unknown, violated
        y1 = np.array(verified)
        y2 = y1 + np.array(unknown)
        y3 = y2 + np.array(violated)

        ax.fill_between(x, 0, y1, color=colors[0], alpha=0.9)
        ax.fill_between(x, y1, y2, color=colors[1], alpha=0.9)
        ax.fill_between(x, y2, y3, color=colors[2], alpha=0.9)

        # White edge lines
        ax.plot(x, y1, color='white', linewidth=1.5, alpha=0.7)
        ax.plot(x, y2, color='white', linewidth=1.5, alpha=0.7)

        # Formatting
        ax.set_title(grid_labels.get(grid, grid), fontweight='bold')
        ax.set_xlabel(r'$\epsilon_{\mathrm{node}}$', fontweight='bold')
        ax.set_xticks(x)
        def fmt_eps(e):
            if e <= 0:
                return '0'
            log_val = np.log10(e)
            if abs(log_val - round(log_val)) < 1e-9:
                return f'$10^{{{int(round(log_val))}}}$'
            return f'{e:g}'
        ax.set_xticklabels([fmt_eps(e) for e in epsilons])
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.3, len(epsilons) - 0.7)
        ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)

    axes[0].set_ylabel('Percentage (%)', fontweight='bold')

    # Single legend
    patches = [
        mpatches.Patch(color=colors[0], label=labels[0]),
        mpatches.Patch(color=colors[1], label=labels[1]),
        mpatches.Patch(color=colors[2], label=labels[2]),
    ]
    fig.legend(handles=patches, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=True)

    arch_display = arch.replace('GINE Conv', 'GINE-Conv').replace('GINE Linear', 'GINE-Linear')
    task_display = task.upper()

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save
    tag = f'{task}_{arch.lower().replace(" ", "_")}'
    for ext in ['pdf', 'png']:
        fpath = output_dir / f'verification_stacked_{tag}.{ext}'
        plt.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f'  Saved: {fpath.name}')
    plt.close()


def generate_timing_figure(results, task, arch, output_dir, grids=None):
    """Generate a grouped bar chart of per-graph timing across grids and epsilons."""
    if grids is None:
        grids = ['ieee24', 'ieee39', 'ieee118']

    grid_labels = {
        'ieee24': 'IEEE-24',
        'ieee39': 'IEEE-39',
        'ieee118': 'IEEE-118',
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))

    all_eps = sorted(set(r['eps'] for r in results
                         if r['task'] == task.lower() and r['arch'] == arch))
    n_grids = len(grids)
    n_eps = len(all_eps)
    width = 0.8 / n_grids
    x = np.arange(n_eps)

    colors_bar = ['#3498db', '#e67e22', '#27ae60']

    for gi, grid in enumerate(grids):
        data = filter_results(results, task=task, arch=arch, grid=grid)
        data = sorted(data, key=lambda r: r['eps'])
        times = []
        for eps in all_eps:
            match = [r for r in data if abs(r['eps'] - eps) < 1e-9]
            times.append(match[0]['avg_time'] if match else 0)

        offset = (gi - n_grids / 2 + 0.5) * width
        ax.bar(x + offset, times, width, label=grid_labels.get(grid, grid),
               color=colors_bar[gi % len(colors_bar)])

    ax.set_ylabel('Time (s/graph)', fontweight='bold')
    ax.set_xlabel(r'$\epsilon_{\mathrm{node}}$', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{e}' for e in all_eps])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.9)

    plt.tight_layout()

    tag = f'{task}_{arch.lower().replace(" ", "_")}'
    for ext in ['pdf', 'png']:
        fpath = output_dir / f'verification_timing_{tag}.{ext}'
        plt.savefig(fpath, dpi=300, bbox_inches='tight')
        print(f'  Saved: {fpath.name}')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate verification figures from log files.')
    parser.add_argument('log_files', nargs='+',
                        help='Log file(s) to parse (e.g., results/logs/pf_*.log)')
    parser.add_argument('--output', '-o', default='results/figures',
                        help='Output directory for figures')
    parser.add_argument('--task', '-t', default=None,
                        help='Filter by task (pf, opf). Default: all tasks found.')
    parser.add_argument('--arch', '-a', default='GINE Conv',
                        help='Architecture to plot (default: "GINE Conv")')
    parser.add_argument('--no-timing', action='store_true',
                        help='Skip timing figure generation')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse all log files
    all_results = []
    for lf in args.log_files:
        path = Path(lf)
        if not path.exists():
            print(f'Warning: {lf} not found, skipping.')
            continue
        print(f'Parsing: {path.name}')
        parsed = parse_log_file(path)
        all_results.extend(parsed)
        print(f'  Found {len(parsed)} configs')

    if not all_results:
        print('No results found. Exiting.')
        return

    # Determine tasks to plot
    tasks_found = sorted(set(r['task'] for r in all_results))
    if args.task:
        tasks_to_plot = [args.task.lower()]
    else:
        tasks_to_plot = tasks_found

    arch = args.arch

    print(f'\nGenerating figures for arch="{arch}", tasks={tasks_to_plot}')

    for task in tasks_to_plot:
        task_data = filter_results(all_results, task=task, arch=arch)
        if not task_data:
            print(f'  No data for task={task}, arch={arch}. Skipping.')
            continue

        print(f'\n--- {task.upper()} {arch} ---')
        generate_stacked_area_figure(all_results, task, arch, output_dir)

        if not args.no_timing:
            generate_timing_figure(all_results, task, arch, output_dir)

    print(f'\nAll figures saved to: {output_dir}')


if __name__ == '__main__':
    main()