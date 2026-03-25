% test_sage_verification.m - End-to-end SAGEConv verification test
%
% Tests: model import, evaluate vs Python, reachability soundness,
%        bound tightness scaling
%
% Uses the real trained sage_pf_ieee24.mat model.

% Shared setup
mat_path = fullfile(fileparts(mfilename('fullpath')), ...
    '..', '..', '..', '..', 'examples', 'NN', 'GNN', 'PowerFlow', ...
    'Comparison', 'models', 'ieee24', 'sage_pf_ieee24.mat');

% Verify model file exists
assert(isfile(mat_path), 'sage_pf_ieee24.mat not found at expected path');

[gnn, test_data, norm_stats] = gnn2nnv(mat_path);
X = test_data.X;
numNodes = size(X, 1);
numFeatures = size(X, 2);

%% 1) Import and evaluate against Python predictions
model = load(mat_path);
py_pred = double(model.python_predictions);
nnv_pred = gnn.evaluate(X);

max_diff = max(abs(nnv_pred(:) - py_pred(:)));
fprintf('Max NNV vs Python difference: %.2e\n', max_diff);
assert(max_diff < 1e-5, ...
    sprintf('NNV evaluate should match Python predictions (diff: %.2e)', max_diff));

% Verify layer structure
assert(gnn.numLayers == 6, 'Should have 6 layers (3 SAGEConv + 3 ReLU)');

%% 2) Reachability soundness with small perturbation
rng(42);
epsilon = 0.001;
perturb_features = [1, 2];

range_per_col = max(X) - min(X);
eps_matrix = zeros(numNodes, numFeatures);
for f = perturb_features
    eps_matrix(:, f) = range_per_col(f) * epsilon;
end

GS_in = GraphStar(X, -eps_matrix, eps_matrix);

reachOpts = struct('reachMethod', 'approx-star');
GS_out = gnn.reach(GS_in, reachOpts);

% Center matches evaluate
center_out = GS_out.V(:, :, 1);
expected_center = gnn.evaluate(X);
center_diff = max(abs(center_out(:) - expected_center(:)));
fprintf('Center propagation diff: %.2e\n', center_diff);
assert(center_diff < 1e-8, 'Center of output set should match evaluate');

% Output bounds are finite
[lb_out, ub_out] = GS_out.getRanges();
assert(all(isfinite(lb_out(:))), 'Lower bounds should be finite');
assert(all(isfinite(ub_out(:))), 'Upper bounds should be finite');
assert(all(lb_out(:) <= ub_out(:)), 'Lower bounds should be <= upper bounds');

% 100 random samples within bounds
num_samples = 100;
failures = 0;
tol = 1e-6;
for s = 1:num_samples
    alpha = rand(GS_in.numPred, 1) .* (GS_in.pred_ub - GS_in.pred_lb) + GS_in.pred_lb;
    X_sample = GS_in.V(:, :, 1);
    for k = 1:GS_in.numPred
        X_sample = X_sample + alpha(k) * GS_in.V(:, :, k+1);
    end
    Y_sample = gnn.evaluate(X_sample);
    if ~(all(Y_sample(:) >= lb_out(:) - tol) && all(Y_sample(:) <= ub_out(:) + tol))
        failures = failures + 1;
    end
end
fprintf('Soundness: %d/%d samples within bounds\n', num_samples - failures, num_samples);
assert(failures == 0, sprintf('%d/%d samples FAILED containment check', failures, num_samples));

%% 3) Bound tightness: widths should scale with epsilon
perturb_features3 = [1, 2];
range_per_col3 = max(X) - min(X);
reachOpts3 = struct('reachMethod', 'approx-star');

eps_small = 0.001;
eps_large = 0.01;

% Small epsilon
eps_mat_small = zeros(numNodes, numFeatures);
for f = perturb_features3
    eps_mat_small(:, f) = range_per_col3(f) * eps_small;
end
GS_in_small = GraphStar(X, -eps_mat_small, eps_mat_small);
GS_out_small = gnn.reach(GS_in_small, reachOpts3);
[lb_s, ub_s] = GS_out_small.getRanges();
widths_small = ub_s - lb_s;

% Large epsilon
eps_mat_large = zeros(numNodes, numFeatures);
for f = perturb_features3
    eps_mat_large(:, f) = range_per_col3(f) * eps_large;
end
GS_in_large = GraphStar(X, -eps_mat_large, eps_mat_large);
GS_out_large = gnn.reach(GS_in_large, reachOpts3);
[lb_l, ub_l] = GS_out_large.getRanges();
widths_large = ub_l - lb_l;

% Larger perturbation should give wider (or equal) bounds
mean_small = mean(widths_small(:));
mean_large = mean(widths_large(:));
fprintf('Mean bound width: eps=%.3f -> %.6f, eps=%.3f -> %.6f\n', ...
    eps_small, mean_small, eps_large, mean_large);
assert(mean_large > mean_small, 'Larger epsilon should produce wider bounds');

% For linear layers + ReLU, scaling should be roughly proportional
% (exact for linear, approximate for ReLU)
ratio = mean_large / mean_small;
expected_ratio = eps_large / eps_small;  % 10x
fprintf('Width ratio: %.2f (expected ~%.1f for linear)\n', ratio, expected_ratio);
assert(ratio > 5 && ratio < 20, ...
    sprintf('Width ratio %.2f should be roughly proportional to epsilon ratio %.1f', ...
    ratio, expected_ratio));

disp('All SAGEConv verification tests passed!');
