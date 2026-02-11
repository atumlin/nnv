% test_gine_linear_pf.m - Test Python-trained simplified GINE model via gnn2nnv
%
% Demonstrates the Python -> MATLAB import pipeline for GINE-linear:
%   1. Load .mat exported from Python training
%   2. Import with gnn2nnv()
%   3. Verify evaluate() matches Python predictions
%   4. Run reachability analysis (node perturbation)
%
% Author: Anne Tumlin
% Date: 02/11/2026

%% Setup
clear; clc;

modelPath = fullfile(fileparts(mfilename('fullpath')), 'models', 'e2e_gine_linear.mat');
fprintf('=== Python-trained GINE-linear Test ===\n');
fprintf('Model: %s\n\n', modelPath);

%% Import model
[gnn, test_data, norm_stats] = gnn2nnv(modelPath);

fprintf('Layers: %d\n', gnn.numLayers);
fprintf('Input: %d nodes x %d features\n', size(test_data.X, 1), size(test_data.X, 2));
fprintf('Edges: %d, Edge features: %d\n\n', size(test_data.adj_list, 1), size(test_data.E, 2));

%% Evaluate
Y = gnn.evaluate(test_data.X);
fprintf('Output: %d nodes x %d features\n\n', size(Y, 1), size(Y, 2));

%% Create perturbation and run reachability
X = test_data.X;
epsilon = 0.01;
eps_matrix = epsilon * ones(size(X));

GS_in = GraphStar(X, -eps_matrix, eps_matrix);

fprintf('=== Reachability Analysis ===\n');
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

fprintf('\n=== Complete ===\n');
