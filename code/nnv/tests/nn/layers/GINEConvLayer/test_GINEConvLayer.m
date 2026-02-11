% test_GINEConvLayer.m - Unit tests for GINEConvLayer class
%
% Tests: constructor, evaluate, reach + soundness, edge perturbation, precision

% Shared setup (before any %% sections)
F_in = 4; hidden = 6; F_out = 3; E_in = 2;

W1 = rand(F_in, hidden); b1 = rand(hidden, 1);
W2 = rand(hidden, F_out); b2 = rand(F_out, 1);
W_edge = rand(E_in, F_in); b_edge = rand(F_in, 1);

L = GINEConvLayer('test_gine_conv', W1, b1, W2, b2, W_edge, b_edge);

numNodes = 4;
numEdges = 5;
X = rand(numNodes, F_in);
E = rand(numEdges, E_in);
adj_list = [1 2; 1 3; 2 3; 3 4; 4 1];

NF = rand(numNodes, F_in);
LB = -0.1 * ones(numNodes, F_in);
UB = 0.1 * ones(numNodes, F_in);
GS_in = GraphStar(NF, LB, UB);

%% 1) Constructor test
assert(L.InputSize == F_in, 'InputSize should be F_in');
assert(L.HiddenSize == hidden, 'HiddenSize should match');
assert(L.OutputSize == F_out, 'OutputSize should be F_out');
assert(L.EdgeInputSize == E_in, 'EdgeInputSize should be E_in');
assert(strcmp(L.Name, 'test_gine_conv'), 'Name should match');
assert(isequal(L.MLPWeights1, W1), 'MLP weights 1 should match');
assert(isequal(L.MLPWeights2, W2), 'MLP weights 2 should match');
assert(isequal(L.EdgeWeights, W_edge), 'Edge weights should match');
assert(L.Epsilon == 0, 'Default epsilon should be 0');

%% 2) Evaluate test
Y = L.evaluate(X, E, adj_list);
assert(size(Y, 1) == numNodes, 'Output should have same number of nodes');
assert(size(Y, 2) == F_out, 'Output should have F_out features');

% Verify computation manually (full GINEConv architecture)
src_nodes = adj_list(:, 1);
dst_nodes = adj_list(:, 2);

% Message passing
E_proj = E * W_edge + b_edge';
X_src = X(src_nodes, :);
edge_msg = max(0, X_src + E_proj);  % ReLU

agg = zeros(numNodes, F_in);
for e = 1:numEdges
    agg(dst_nodes(e), :) = agg(dst_nodes(e), :) + edge_msg(e, :);
end

% Self-loop + MLP
combined = X + agg;  % eps=0
H = max(0, combined * W1 + b1');  % MLP layer 1 + ReLU
expected_Y = H * W2 + b2';       % MLP layer 2

assert(max(abs(Y - expected_Y), [], 'all') < 1e-10, 'Evaluate should match manual computation');

%% 3) Evaluate with epsilon test
L_eps = GINEConvLayer('eps_test', W1, b1, W2, b2, W_edge, b_edge, 0.5);
Y_eps = L_eps.evaluate(X, E, adj_list);

% Recompute message passing for this independent test section
src_eps = adj_list(:, 1);
dst_eps = adj_list(:, 2);
E_proj_eps = E * W_edge + b_edge';
X_src_eps = X(src_eps, :);
edge_msg_eps = max(0, X_src_eps + E_proj_eps);
agg_eps = zeros(numNodes, F_in);
for e = 1:numEdges
    agg_eps(dst_eps(e), :) = agg_eps(dst_eps(e), :) + edge_msg_eps(e, :);
end

combined_eps = (1 + 0.5) * X + agg_eps;
H_eps = max(0, combined_eps * W1 + b1');
expected_Y_eps = H_eps * W2 + b2';
assert(max(abs(Y_eps - expected_Y_eps), [], 'all') < 1e-10, 'Epsilon evaluate should match');

%% 4) Reach and soundness test (node-only perturbation)
GS_out = L.reach(GS_in, E, adj_list, 'approx-star');
assert(isa(GS_out, 'GraphStar'), 'Output should be GraphStar');
assert(GS_out.numNodes == numNodes, 'Output should have same number of nodes');
assert(GS_out.numFeatures == F_out, 'Output should have F_out features');

% Soundness: center should match evaluate
center_in = GS_in.V(:, :, 1);
center_out = GS_out.V(:, :, 1);
expected_center = L.evaluate(center_in, E, adj_list);
assert(max(abs(center_out - expected_center), [], 'all') < 1e-10, 'Center should match evaluate');

% Containment: center output should be within bounds
[lb_out, ub_out] = GS_out.getRanges();
Y_center = L.evaluate(GS_in.V(:,:,1), E, adj_list);
tol = 1e-6;
assert(all(Y_center(:) >= lb_out(:) - tol), 'Center output should be >= lower bound');
assert(all(Y_center(:) <= ub_out(:) + tol), 'Center output should be <= upper bound');

%% 5) Edge perturbation reach and soundness test
E_lb = E - 0.05;
E_ub = E + 0.05;
E_star = ImageStar(E_lb, E_ub);

GS_out_edge = L.reach(GS_in, E_star, adj_list, 'approx-star');
assert(isa(GS_out_edge, 'GraphStar'), 'Output should be GraphStar with edge perturbation');
assert(GS_out_edge.numNodes == numNodes, 'Output should have same number of nodes');

% Soundness: center should match evaluate
center_out_edge = GS_out_edge.V(:, :, 1);
expected_center_edge = L.evaluate(GS_in.V(:,:,1), E, adj_list);
assert(max(abs(center_out_edge - expected_center_edge), [], 'all') < 1e-10, ...
    'Center should match evaluate for edge perturbation');

% Containment: center output should be within bounds
[lb_out_edge, ub_out_edge] = GS_out_edge.getRanges();
Y_center_edge = L.evaluate(GS_in.V(:,:,1), E, adj_list);
tol_edge = 1e-6;
assert(all(Y_center_edge(:) >= lb_out_edge(:) - tol_edge), 'Center output should be >= lower bound (edge perturbation)');
assert(all(Y_center_edge(:) <= ub_out_edge(:) + tol_edge), 'Center output should be <= upper bound (edge perturbation)');

%% 6) Precision change test
L_prec = GINEConvLayer('prec_test', W1, b1, W2, b2, W_edge, b_edge);
L_prec.changeParamsPrecision('single');
assert(isa(L_prec.MLPWeights1, 'single'), 'MLP weights 1 should be single');
assert(isa(L_prec.MLPWeights2, 'single'), 'MLP weights 2 should be single');
assert(isa(L_prec.EdgeWeights, 'single'), 'Edge weights should be single');

L_prec.changeParamsPrecision('double');
assert(isa(L_prec.MLPWeights1, 'double'), 'MLP weights 1 should be double');

%% 7) Constructor with 6 args (no name)
L_noname = GINEConvLayer(W1, b1, W2, b2, W_edge, b_edge);
assert(strcmp(L_noname.Name, 'gine_conv_layer'), 'Default name should be gine_conv_layer');
assert(L_noname.InputSize == F_in, 'InputSize should match with 6-arg constructor');

disp('All GINEConvLayer tests passed!');
