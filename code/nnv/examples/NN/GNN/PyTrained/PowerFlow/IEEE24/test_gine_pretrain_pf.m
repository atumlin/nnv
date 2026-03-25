% test_gine_pretrain_pf.m - Test Python-trained Hu et al. GIN+E model via gnn2nnv
%
% Demonstrates the Python -> MATLAB import pipeline for Hu et al. GIN+E:
%   1. Load .mat exported from Python training
%   2. Import with gnn2nnv() (requires HuGINEConvLayer)
%   3. Verify evaluate() matches Python predictions
%   4. Run reachability analysis (node perturbation)
%   5. Run subgraph reachability for a target node
%
% Hu et al. ICLR 2020 "Strategies for Pre-training Graph Neural Networks"
%
% Author: Anne Tumlin
% Date: 03/17/2026

%% Setup
clear; clc;

modelPath = fullfile(fileparts(mfilename('fullpath')), 'models', 'gine_pretrain_pf_ieee24.mat');
fprintf('=== Python-trained Hu et al. GIN+E (HuGINEConvLayer) Test ===\n');
fprintf('Model: %s\n\n', modelPath);

if ~isfile(modelPath)
    fprintf('Model file not found. Copy from gnn_training/outputs/ieee24_pf/\n');
    return;
end

%% Import model
[gnn, test_data, norm_stats] = gnn2nnv(modelPath);

fprintf('Layers: %d\n', gnn.numLayers);
fprintf('Input: %d nodes x %d features\n', size(test_data.X, 1), size(test_data.X, 2));
fprintf('Edges: %d (including self-loops), Edge features: %d\n\n', ...
    size(test_data.adj_list, 1), size(test_data.E, 2));

%% Evaluate
Y = gnn.evaluate(test_data.X);
fprintf('Output: %d nodes x %d features\n\n', size(Y, 1), size(Y, 2));

%% Create perturbation and run full-graph reachability
X = test_data.X;
epsilon = 0.01;
eps_matrix = epsilon * ones(size(X));

GS_in = GraphStar(X, -eps_matrix, eps_matrix);

fprintf('=== Full-Graph Reachability Analysis ===\n');
reachOpts = struct('reachMethod', 'approx-star');
t_start = tic;
GS_out = gnn.reach(GS_in, reachOpts);
total_time = toc(t_start);

fprintf('Completed in %.4f seconds\n', total_time);

%% Verify soundness
[lb_out, ub_out] = GS_out.getRanges();
Y_center = gnn.evaluate(GS_in.V(:,:,1));
tol = 1e-6;

assert(all(Y_center(:) >= lb_out(:) - tol), 'Center should be >= lower bound');
assert(all(Y_center(:) <= ub_out(:) + tol), 'Center should be <= upper bound');
fprintf('Soundness check PASSED\n');

%% Report bounds
bound_widths = ub_out - lb_out;
fprintf('\nOutput bounds:\n');
fprintf('  Mean width: %.6f\n', mean(bound_widths(:)));
fprintf('  Max width:  %.6f\n', max(bound_widths(:)));
fprintf('  Min width:  %.6f\n', min(bound_widths(:)));

%% Subgraph reachability for a few nodes
fprintf('\n=== Subgraph Reachability (per-node) ===\n');
target_nodes = [1; 5; 10; 15; 20];
target_nodes = target_nodes(target_nodes <= size(X, 1));

t_sub_start = tic;
[node_results, sg_info] = gnn.reachSubgraph(GS_in, target_nodes, reachOpts);
total_sub_time = toc(t_sub_start);

fprintf('Total time for %d nodes: %.4f seconds\n\n', length(target_nodes), total_sub_time);

for ti = 1:length(target_nodes)
    t = target_nodes(ti);
    gs_sub = node_results{ti};
    t_local = sg_info(ti).target_local_idx;

    % Compare with full-graph bounds
    fprintf('Node %2d: subgraph=%2d nodes, %2d edges, time=%.4fs', ...
        t, sg_info(ti).n_sub_nodes, sg_info(ti).n_sub_edges, sg_info(ti).time);

    max_bound_diff = 0;
    for fi = 1:GS_out.numFeatures
        [full_lb, full_ub] = GS_out.getRange(t, fi);
        [sub_lb, sub_ub] = gs_sub.getRange(t_local, fi);
        max_bound_diff = max(max_bound_diff, max(abs(full_lb - sub_lb), abs(full_ub - sub_ub)));
    end
    fprintf(', max bound diff=%.2e\n', max_bound_diff);
end

fprintf('\n=== Complete ===\n');
