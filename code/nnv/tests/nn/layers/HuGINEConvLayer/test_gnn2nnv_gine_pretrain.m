% test_gnn2nnv_gine_pretrain.m - Import pipeline validation for gine_pretrain
%
% Tests that gnn2nnv correctly imports a Python-trained HuGINEConvLayer
% model and produces sound reachability results.
%
% Requires: gine_pretrain_pf_ieee24.mat in outputs/ieee24_pf/

addpath(genpath('../../../../engine'));

fprintf('=== test_gnn2nnv_gine_pretrain ===\n');

%% Find the model file
mat_path = '';
search_paths = {
    '../../../../examples/NN/GNN/PyTrained/PowerFlow/IEEE24/gine_pretrain_pf_ieee24.mat',
    '../../../../../../../gnn_training/outputs/ieee24_pf/gine_pretrain_pf_ieee24.mat',
};

% Also check environment variable
if isempty(mat_path)
    for i = 1:length(search_paths)
        if isfile(search_paths{i})
            mat_path = search_paths{i};
            break;
        end
    end
end

if isempty(mat_path)
    fprintf('  SKIPPED: gine_pretrain_pf_ieee24.mat not found.\n');
    fprintf('  Run train_gine_pretrain.py first, then copy .mat to test location.\n');
    return;
end

fprintf('  Using model: %s\n', mat_path);

%% 1) Import Python-trained model
fprintf('\nTest 1: Import model...\n');

[gnn, test_data, norm_stats] = gnn2nnv(mat_path);

% Check layer types
has_hugine = false;
has_relu = false;
for i = 1:gnn.numLayers
    if isa(gnn.Layers{i}, 'HuGINEConvLayer')
        has_hugine = true;
    end
    if isa(gnn.Layers{i}, 'ReluLayer')
        has_relu = true;
    end
end

assert(has_hugine, 'Test 1 FAIL: No HuGINEConvLayer found in imported GNN');
assert(has_relu, 'Test 1 FAIL: No interleaved ReluLayer found');
assert(~isempty(gnn.adj_list), 'Test 1 FAIL: adj_list should be set');
assert(~isempty(gnn.E), 'Test 1 FAIL: edge features should be set');

fprintf('  Test 1 PASS: %d layers imported (%d HuGINEConvLayer + ReluLayer)\n', gnn.numLayers, gnn.numLayers);

%% 2) Cross-validation: NNV evaluate matches Python predictions
fprintf('\nTest 2: Cross-validation with Python predictions...\n');

X_test = test_data.X;
nnv_pred = gnn.evaluate(X_test);

model_raw = load(mat_path);
if isfield(model_raw, 'python_predictions')
    py_pred = model_raw.python_predictions;
    if iscell(py_pred)
        py_pred = double(py_pred{1});
    else
        py_pred = double(py_pred);
    end
    max_diff = max(abs(nnv_pred(:) - py_pred(:)));
    fprintf('  Max difference (NNV vs Python): %.2e\n', max_diff);
    assert(max_diff < 1e-5, sprintf('Test 2 FAIL: max diff %.2e exceeds threshold', max_diff));
    fprintf('  Test 2 PASS.\n');
else
    fprintf('  Test 2 SKIPPED: no python_predictions in .mat\n');
end

%% 3) Reachability on real data
fprintf('\nTest 3: Reachability on real data...\n');

eps_val = 0.01;
eps_matrix = eps_val * ones(size(X_test));
GS_in = GraphStar(X_test, -eps_matrix, eps_matrix);

reachOpts = struct('reachMethod', 'approx-star');
GS_out = gnn.reach(GS_in, reachOpts);

[lb, ub] = GS_out.getRanges();
assert(all(isfinite(lb(:))), 'Test 3 FAIL: lower bounds contain NaN/Inf');
assert(all(isfinite(ub(:))), 'Test 3 FAIL: upper bounds contain NaN/Inf');

% Check that unperturbed evaluation is within bounds
Y_center = gnn.evaluate(X_test);
tol = 1e-6;
assert(all(Y_center(:) >= lb(:) - tol), 'Test 3 FAIL: center output below lower bound');
assert(all(Y_center(:) <= ub(:) + tol), 'Test 3 FAIL: center output above upper bound');

fprintf('  Test 3 PASS: bounds are finite and contain center evaluation\n');

%% 4) Soundness on real data: 100 random samples
fprintf('\nTest 4: Real-data soundness (100 random samples)...\n');

rng(42);
num_samples = 100;
violations = 0;
min_margin = inf;

for s = 1:num_samples
    alpha = rand(GS_in.numPred, 1) .* (GS_in.pred_ub - GS_in.pred_lb) + GS_in.pred_lb;
    X_s = GS_in.V(:, :, 1);
    for k = 1:GS_in.numPred
        X_s = X_s + alpha(k) * GS_in.V(:, :, k+1);
    end
    Y_s = gnn.evaluate(X_s);

    lb_margin = min(Y_s(:) - lb(:));
    ub_margin = min(ub(:) - Y_s(:));
    margin = min(lb_margin, ub_margin);
    min_margin = min(min_margin, margin);

    if margin < -tol
        violations = violations + 1;
    end
end

fprintf('  %d violations / %d samples, worst margin=%.2e\n', violations, num_samples, min_margin);
assert(violations == 0, 'Test 4 FAIL: soundness violation on real data');
fprintf('  Test 4 PASS.\n');

%% Summary
fprintf('\n=== All gnn2nnv gine_pretrain import tests PASSED ===\n');
