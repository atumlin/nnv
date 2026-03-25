% test_HuGINEConvLayer.m - Unit tests for HuGINEConvLayer class
%
% Tests: constructor, evaluate, reach + soundness, edge perturbation,
%        random sample containment, precision
%
% Hu et al. ICLR 2020 "Strategies for Pre-training Graph Neural Networks"

% Shared setup (before any %% sections)
rng(42);
F_in = 4; hidden = 6; F_out = 3; E_in = 2;

W1 = randn(F_in, hidden) * 0.1; b1 = randn(hidden, 1) * 0.1;
W2 = randn(hidden, F_out) * 0.1; b2 = randn(F_out, 1) * 0.1;
W_edge = randn(E_in, F_in) * 0.1; b_edge = randn(F_in, 1) * 0.1;

L = HuGINEConvLayer('test_hu_gine', W1, b1, W2, b2, W_edge, b_edge);

% Small graph: 4 nodes, 5 edges + 4 self-loop edges (with zero edge feats)
numNodes = 4;
base_edges = 5;
adj_list = [1 2; 1 3; 2 3; 3 4; 4 1; ...   % real edges
            1 1; 2 2; 3 3; 4 4];            % self-loops
numEdges = size(adj_list, 1);
E_base = randn(base_edges, E_in) * 0.1;
E = [E_base; zeros(numNodes, E_in)];  % self-loop edges have zero features

NF = randn(numNodes, F_in) * 0.5;
LB = -0.1 * ones(numNodes, F_in);
UB = 0.1 * ones(numNodes, F_in);
GS_in = GraphStar(NF, LB, UB);

%% 1) Constructor test
assert(L.InputSize == F_in, 'InputSize should be F_in');
assert(L.HiddenSize == hidden, 'HiddenSize should match');
assert(L.OutputSize == F_out, 'OutputSize should be F_out');
assert(L.EdgeInputSize == E_in, 'EdgeInputSize should be E_in');
assert(strcmp(L.Name, 'test_hu_gine'), 'Name should match');
assert(isequal(L.MLPWeights1, W1), 'MLP weights 1 should match');
assert(isequal(L.MLPWeights2, W2), 'MLP weights 2 should match');
assert(isequal(L.EdgeProjWeights, W_edge), 'Edge proj weights should match');
assert(isequal(L.EdgeProjBias, b_edge), 'Edge proj bias should match');

%% 2) Evaluate test - manual computation verification
X = randn(numNodes, F_in) * 0.5;
Y = L.evaluate(X, E, adj_list);
assert(size(Y, 1) == numNodes, 'Output should have same number of nodes');
assert(size(Y, 2) == F_out, 'Output should have F_out features');

% Verify computation manually (Hu et al. architecture)
src_nodes = adj_list(:, 1);
dst_nodes = adj_list(:, 2);

% Message passing (NO ReLU - key difference from GINEConvLayer)
E_proj = E * W_edge + b_edge';
X_src = X(src_nodes, :);
edge_msg = X_src + E_proj;  % NO ReLU here

agg = zeros(numNodes, F_in);
for e = 1:numEdges
    agg(dst_nodes(e), :) = agg(dst_nodes(e), :) + edge_msg(e, :);
end

% MLP: Linear -> ReLU -> Linear
H = max(0, agg * W1 + b1');  % MLP layer 1 + ReLU
expected_Y = H * W2 + b2';  % MLP layer 2

assert(max(abs(Y - expected_Y), [], 'all') < 1e-10, 'Evaluate should match manual computation');

%% 3) Reach and center soundness (node-only perturbation)
GS_out = L.reach(GS_in, E, adj_list, 'approx-star');
assert(isa(GS_out, 'GraphStar'), 'Output should be GraphStar');
assert(GS_out.numNodes == numNodes, 'Output should have same number of nodes');
assert(GS_out.numFeatures == F_out, 'Output should have F_out features');

% Soundness: center evaluation should be within reach bounds
% Note: approx-star ReLU relaxation can shift the GraphStar center,
% so we check containment rather than exact center match.
[lb_out, ub_out] = GS_out.getRanges();
Y_center = L.evaluate(GS_in.V(:,:,1), E, adj_list);
tol = 1e-6;
assert(all(Y_center(:) >= lb_out(:) - tol), 'Center output should be >= lower bound');
assert(all(Y_center(:) <= ub_out(:) + tol), 'Center output should be <= upper bound');

%% 4) Random sample containment - node-only (100 samples)
rng(42);
GS_out4 = L.reach(GS_in, E, adj_list, 'approx-star');
[lb_out4, ub_out4] = GS_out4.getRanges();
tol4 = 1e-6;

num_samples = 100;
all_in_bounds = true;
for s = 1:num_samples
    % Random node input within GraphStar bounds
    alpha = rand(GS_in.numPred, 1) .* (GS_in.pred_ub - GS_in.pred_lb) + GS_in.pred_lb;
    X_s = GS_in.V(:, :, 1);
    for k = 1:GS_in.numPred
        X_s = X_s + alpha(k) * GS_in.V(:, :, k+1);
    end
    % Evaluate and check containment
    Y_s = L.evaluate(X_s, E, adj_list);
    if ~(all(Y_s(:) >= lb_out4(:) - tol4) && all(Y_s(:) <= ub_out4(:) + tol4))
        all_in_bounds = false;
        fprintf('Sample %d FAILED containment check (node-only)\n', s);
        fprintf('  Max lb violation: %.2e\n', max(lb_out4(:) - Y_s(:)));
        fprintf('  Max ub violation: %.2e\n', max(Y_s(:) - ub_out4(:)));
        break;
    end
end
assert(all_in_bounds, 'All random samples should be within output bounds (node-only)');

%% 5) Edge perturbation reach and center soundness
E_lb = E - 0.05;
E_ub = E + 0.05;
E_star = ImageStar(E_lb, E_ub);

GS_out_edge = L.reach(GS_in, E_star, adj_list, 'approx-star');
assert(isa(GS_out_edge, 'GraphStar'), 'Output should be GraphStar with edge perturbation');
assert(GS_out_edge.numNodes == numNodes, 'Output should have same number of nodes');

% Soundness: center evaluation should be within reach bounds
[lb_out_edge, ub_out_edge] = GS_out_edge.getRanges();
Y_center_edge = L.evaluate(GS_in.V(:,:,1), E, adj_list);
tol_edge = 1e-6;
assert(all(Y_center_edge(:) >= lb_out_edge(:) - tol_edge), 'Center output should be >= lower bound (edge perturbation)');
assert(all(Y_center_edge(:) <= ub_out_edge(:) + tol_edge), 'Center output should be <= upper bound (edge perturbation)');

%% 6) Edge perturbation random sample containment (50 samples)
rng(42);
E_lb6 = E - 0.05;
E_ub6 = E + 0.05;
E_star6 = ImageStar(E_lb6, E_ub6);

GS_out6 = L.reach(GS_in, E_star6, adj_list, 'approx-star');
[lb_out6, ub_out6] = GS_out6.getRanges();
tol6 = 1e-6;

num_samples = 50;
all_in_bounds = true;
for s = 1:num_samples
    % Random node input within GraphStar bounds
    alpha_n = rand(GS_in.numPred, 1) .* (GS_in.pred_ub - GS_in.pred_lb) + GS_in.pred_lb;
    X_s = GS_in.V(:, :, 1);
    for k = 1:GS_in.numPred
        X_s = X_s + alpha_n(k) * GS_in.V(:, :, k+1);
    end
    % Random edge input within perturbation bounds
    E_s = E_lb6 + rand(size(E)) .* (E_ub6 - E_lb6);
    % Evaluate and check containment
    Y_s = L.evaluate(X_s, E_s, adj_list);
    if ~(all(Y_s(:) >= lb_out6(:) - tol6) && all(Y_s(:) <= ub_out6(:) + tol6))
        all_in_bounds = false;
        fprintf('Sample %d FAILED containment check (edge perturbation)\n', s);
        fprintf('  Max lb violation: %.2e\n', max(lb_out6(:) - Y_s(:)));
        fprintf('  Max ub violation: %.2e\n', max(Y_s(:) - ub_out6(:)));
        break;
    end
end
assert(all_in_bounds, 'All random samples should be within output bounds (edge perturbation)');

%% 7) Precision change test
L_prec = HuGINEConvLayer('prec_test', W1, b1, W2, b2, W_edge, b_edge);
L_prec.changeParamsPrecision('single');
assert(isa(L_prec.MLPWeights1, 'single'), 'MLP weights 1 should be single');
assert(isa(L_prec.MLPWeights2, 'single'), 'MLP weights 2 should be single');
assert(isa(L_prec.EdgeProjWeights, 'single'), 'Edge proj weights should be single');

L_prec.changeParamsPrecision('double');
assert(isa(L_prec.MLPWeights1, 'double'), 'MLP weights 1 should be double');

%% 8) Constructor variants test
L_noname = HuGINEConvLayer(W1, b1, W2, b2, W_edge, b_edge);
assert(strcmp(L_noname.Name, 'hu_gine_conv_layer'), 'Default name should be hu_gine_conv_layer');
assert(L_noname.InputSize == F_in, 'InputSize should match with 6-arg constructor');

disp('All HuGINEConvLayer tests passed!');
