% test_HuGINEConvLayer_soundness_sampling.m
% Large-scale sampling-based soundness tests for HuGINEConvLayer
%
% Validates that the over-approximation is sound (no sample escapes bounds)
% across varying perturbation sizes, corner cases, and multi-layer networks.

addpath(genpath('../../../../engine'));

fprintf('=== test_HuGINEConvLayer_soundness_sampling ===\n');

%% Shared setup
rng(42);
F_in = 4; hidden = 6; F_out = 3; E_in = 2;
numNodes = 4;

W1 = randn(F_in, hidden) * 0.1; b1 = randn(hidden, 1) * 0.1;
W2 = randn(hidden, F_out) * 0.1; b2 = randn(F_out, 1) * 0.1;
W_edge = randn(E_in, F_in) * 0.1; b_edge = randn(F_in, 1) * 0.1;

L = HuGINEConvLayer('hugine_stress', W1, b1, W2, b2, W_edge, b_edge);

% Graph with self-loops
adj_list = [1 2; 1 3; 2 3; 3 4; 4 1; 1 1; 2 2; 3 3; 4 4];
numEdges = size(adj_list, 1);
E = [randn(5, E_in) * 0.1; zeros(numNodes, E_in)];
NF = randn(numNodes, F_in) * 0.5;

reachOpts = struct('reachMethod', 'approx-star');

%% 1) Stress test: 1000 random samples, node-only, varying perturbation sizes
fprintf('\nTest 1: Node-only stress test (1000 samples per eps)...\n');

eps_values = [0.01, 0.05, 0.1, 0.2];
tol = 1e-6;

for ei = 1:length(eps_values)
    eps_val = eps_values(ei);
    LB = -eps_val * ones(numNodes, F_in);
    UB = eps_val * ones(numNodes, F_in);
    GS_in = GraphStar(NF, LB, UB);

    GS_out = L.reach(GS_in, E, adj_list, 'approx-star');
    [lb, ub] = GS_out.getRanges();

    violations = 0;
    min_margin = inf;
    num_samples = 1000;

    for s = 1:num_samples
        alpha = rand(GS_in.numPred, 1) .* (GS_in.pred_ub - GS_in.pred_lb) + GS_in.pred_lb;
        X_s = GS_in.V(:, :, 1);
        for k = 1:GS_in.numPred
            X_s = X_s + alpha(k) * GS_in.V(:, :, k+1);
        end
        Y_s = L.evaluate(X_s, E, adj_list);

        lb_margin = min(Y_s(:) - lb(:));
        ub_margin = min(ub(:) - Y_s(:));
        margin = min(lb_margin, ub_margin);
        min_margin = min(min_margin, margin);

        if margin < -tol
            violations = violations + 1;
        end
    end

    fprintf('  eps=%.2f: %d violations / %d samples, worst margin=%.2e\n', ...
        eps_val, violations, num_samples, min_margin);
    assert(violations == 0, sprintf('Soundness violation at eps=%.2f', eps_val));
end

%% 2) Stress test: 500 random samples, node+edge perturbation
fprintf('\nTest 2: Node+edge perturbation stress test (500 samples)...\n');

eps_node = 0.05;
eps_edge = 0.05;
LB_n = -eps_node * ones(numNodes, F_in);
UB_n = eps_node * ones(numNodes, F_in);
GS_in2 = GraphStar(NF, LB_n, UB_n);

E_lb = E - eps_edge;
E_ub = E + eps_edge;
E_star = ImageStar(E_lb, E_ub);

GS_out2 = L.reach(GS_in2, E_star, adj_list, 'approx-star');
[lb2, ub2] = GS_out2.getRanges();

violations2 = 0;
min_margin2 = inf;
num_samples2 = 500;

for s = 1:num_samples2
    alpha = rand(GS_in2.numPred, 1) .* (GS_in2.pred_ub - GS_in2.pred_lb) + GS_in2.pred_lb;
    X_s = GS_in2.V(:, :, 1);
    for k = 1:GS_in2.numPred
        X_s = X_s + alpha(k) * GS_in2.V(:, :, k+1);
    end
    E_s = E_lb + rand(size(E)) .* (E_ub - E_lb);
    Y_s = L.evaluate(X_s, E_s, adj_list);

    lb_margin = min(Y_s(:) - lb2(:));
    ub_margin = min(ub2(:) - Y_s(:));
    margin = min(lb_margin, ub_margin);
    min_margin2 = min(min_margin2, margin);

    if margin < -tol
        violations2 = violations2 + 1;
    end
end

fprintf('  %d violations / %d samples, worst margin=%.2e\n', violations2, num_samples2, min_margin2);
assert(violations2 == 0, 'Soundness violation in node+edge perturbation');

%% 3) Corner case testing - predicate vertices
fprintf('\nTest 3: Predicate vertex testing...\n');

eps_corner = 0.05;
LB_c = -eps_corner * ones(numNodes, F_in);
UB_c = eps_corner * ones(numNodes, F_in);
GS_in3 = GraphStar(NF, LB_c, UB_c);

GS_out3 = L.reach(GS_in3, E, adj_list, 'approx-star');
[lb3, ub3] = GS_out3.getRanges();

numPred = GS_in3.numPred;
% For small numPred, test all 2^p corners; for large, sample 1024
if numPred <= 10
    num_corners = 2^numPred;
    corner_violations = 0;
    for c = 0:(num_corners - 1)
        bits = dec2bin(c, numPred) - '0';
        alpha = GS_in3.pred_lb + bits(:) .* (GS_in3.pred_ub - GS_in3.pred_lb);
        X_s = GS_in3.V(:, :, 1);
        for k = 1:numPred
            X_s = X_s + alpha(k) * GS_in3.V(:, :, k+1);
        end
        Y_s = L.evaluate(X_s, E, adj_list);
        if ~(all(Y_s(:) >= lb3(:) - tol) && all(Y_s(:) <= ub3(:) + tol))
            corner_violations = corner_violations + 1;
        end
    end
    fprintf('  Tested %d corners: %d violations\n', num_corners, corner_violations);
    assert(corner_violations == 0, 'Corner vertex soundness violation');
else
    num_corners = 1024;
    corner_violations = 0;
    for c = 1:num_corners
        bits = randi([0, 1], numPred, 1);
        alpha = GS_in3.pred_lb + bits .* (GS_in3.pred_ub - GS_in3.pred_lb);
        X_s = GS_in3.V(:, :, 1);
        for k = 1:numPred
            X_s = X_s + alpha(k) * GS_in3.V(:, :, k+1);
        end
        Y_s = L.evaluate(X_s, E, adj_list);
        if ~(all(Y_s(:) >= lb3(:) - tol) && all(Y_s(:) <= ub3(:) + tol))
            corner_violations = corner_violations + 1;
        end
    end
    fprintf('  Sampled %d random corners: %d violations\n', num_corners, corner_violations);
    assert(corner_violations == 0, 'Random corner vertex soundness violation');
end

%% 4) Multi-layer soundness (2- and 3-layer GNN)
fprintf('\nTest 4: Multi-layer network soundness...\n');

% Build 2-layer network: HuGINEConv -> ReLU -> HuGINEConv
W1b = randn(F_out, hidden) * 0.1; b1b = randn(hidden, 1) * 0.1;
W2b = randn(hidden, F_out) * 0.1; b2b = randn(F_out, 1) * 0.1;
W_edge_b = randn(E_in, F_out) * 0.1; b_edge_b = randn(F_out, 1) * 0.1;

L2 = HuGINEConvLayer('hugine2', W1b, b1b, W2b, b2b, W_edge_b, b_edge_b);

eps_ml = 0.05;
LB_ml = -eps_ml * ones(numNodes, F_in);
UB_ml = eps_ml * ones(numNodes, F_in);
GS_in_ml = GraphStar(NF, LB_ml, UB_ml);

% 2-layer GNN
gnn2 = GNN({L, ReluLayer(), L2});
gnn2.adj_list = adj_list;
gnn2.E = E;
gnn2.edge_weights = ones(numEdges, 1);

GS_out_ml = gnn2.reach(GS_in_ml, reachOpts);
[lb_ml, ub_ml] = GS_out_ml.getRanges();

violations_ml = 0;
num_samples_ml = 500;
min_margin_ml = inf;

for s = 1:num_samples_ml
    alpha = rand(GS_in_ml.numPred, 1) .* (GS_in_ml.pred_ub - GS_in_ml.pred_lb) + GS_in_ml.pred_lb;
    X_s = GS_in_ml.V(:, :, 1);
    for k = 1:GS_in_ml.numPred
        X_s = X_s + alpha(k) * GS_in_ml.V(:, :, k+1);
    end
    Y_s = gnn2.evaluate(X_s);
    margin = min(min(Y_s(:) - lb_ml(:)), min(ub_ml(:) - Y_s(:)));
    min_margin_ml = min(min_margin_ml, margin);
    if margin < -tol
        violations_ml = violations_ml + 1;
    end
end

fprintf('  2-layer GNN: %d violations / %d samples, worst margin=%.2e\n', ...
    violations_ml, num_samples_ml, min_margin_ml);
assert(violations_ml == 0, '2-layer network soundness violation');

%% 5) Comparison with GINEConvLayer over-approximation tightness
fprintf('\nTest 5: Tightness comparison vs GINEConvLayer...\n');

% Build equivalent GINEConvLayer (eps=0, same weights)
% Note: GINEConvLayer has ReLU in message, HuGINEConvLayer does not.
% They compute DIFFERENT functions, so we cannot compare evaluate() output.
% Instead we compare how TIGHT the reach bounds are on equivalent problems.
L_gine = GINEConvLayer('gine_ref', W1, b1, W2, b2, W_edge, b_edge, 0.0);

eps_tight = 0.05;
LB_t = -eps_tight * ones(numNodes, F_in);
UB_t = eps_tight * ones(numNodes, F_in);
GS_in_t = GraphStar(NF, LB_t, UB_t);

GS_out_hu = L.reach(GS_in_t, E, adj_list, 'approx-star');
[lb_hu, ub_hu] = GS_out_hu.getRanges();
widths_hu = ub_hu - lb_hu;

% GINEConvLayer needs adj_list WITHOUT self-loops (it has (1+eps)*x)
adj_no_self = adj_list(1:5, :);  % first 5 edges are real
E_no_self = E(1:5, :);
GS_out_gine = L_gine.reach(GS_in_t, E_no_self, adj_no_self, 'approx-star');
[lb_gine, ub_gine] = GS_out_gine.getRanges();
widths_gine = ub_gine - lb_gine;

mean_width_hu = mean(widths_hu(:));
mean_width_gine = mean(widths_gine(:));
ratio = mean_width_gine / mean_width_hu;

fprintf('  HuGINEConvLayer mean bound width: %.6f\n', mean_width_hu);
fprintf('  GINEConvLayer   mean bound width: %.6f\n', mean_width_gine);
fprintf('  Ratio (GINE/Hu): %.2f (>1 means Hu is tighter)\n', ratio);

% We expect Hu to be at least as tight (fewer ReLU passes)
% But don't assert since they compute different functions
fprintf('  (Informational only - architectures compute different functions)\n');

%% Summary
fprintf('\n=== All HuGINEConvLayer soundness sampling tests PASSED ===\n');
