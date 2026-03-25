% test_SAGEConvLayer.m - Unit tests for SAGEConvLayer class
%
% Tests: constructor, evaluate, reach + soundness, sample containment,
%        precision conversion

% Shared setup (before any %% sections)
rng(42);
F_in = 4; F_out = 8; numNodes = 5;
W_node = rand(F_in, F_out);
W_edge = rand(F_in, F_out);
b = rand(F_out, 1);
L = SAGEConvLayer('test_sage', W_node, W_edge, b);

X = rand(numNodes, F_in);

% Binary adjacency (no self-loops)
A = double(rand(numNodes) > 0.5);
A(1:numNodes+1:end) = 0;  % Remove self-loops
A = double(A | A');        % Symmetric

% GraphStar input
NF = rand(numNodes, F_in);
LB = -0.1 * ones(numNodes, F_in);
UB = 0.1 * ones(numNodes, F_in);
GS_in = GraphStar(NF, LB, UB);

%% 1) Constructor test
assert(L.InputSize == F_in, 'InputSize should be F_in');
assert(L.OutputSize == F_out, 'OutputSize should be F_out');
assert(strcmp(L.Name, 'test_sage'), 'Name should match');
assert(isequal(L.NodeWeights, W_node), 'NodeWeights should match');
assert(isequal(L.EdgeWeights, W_edge), 'EdgeWeights should match');
assert(isequal(L.Bias, b), 'Bias should match');

% Test unnamed constructor
L2 = SAGEConvLayer(W_node, W_edge, b);
assert(L2.InputSize == F_in, '3-arg constructor should set InputSize');

%% 2) Evaluate test
Y = L.evaluate(X, A);
assert(size(Y, 1) == numNodes, 'Output should have same number of nodes');
assert(size(Y, 2) == F_out, 'Output should have F_out features');

% Verify computation manually: Y = X * W_node + (A * X) * W_edge + b'
expected_Y = X * W_node + (A * X) * W_edge + b';
assert(max(abs(Y - expected_Y), [], 'all') < 1e-10, ...
    'Evaluate should match manual computation');

%% 3) Reach and soundness test
GS_out = L.reach(GS_in, A, 'approx-star');
assert(isa(GS_out, 'GraphStar'), 'Output should be GraphStar');
assert(GS_out.numNodes == numNodes, 'Output should have same number of nodes');
assert(GS_out.numFeatures == F_out, 'Output should have F_out features');
assert(GS_out.numPred == GS_in.numPred, 'Number of predicates should be preserved');

% Soundness: center of output should match evaluate on center of input
center_in = GS_in.V(:, :, 1);
center_out = GS_out.V(:, :, 1);
expected_center = L.evaluate(center_in, A);
assert(max(abs(center_out - expected_center), [], 'all') < 1e-10, ...
    'Center should match evaluate (exact for linear layer)');

% Containment: center output should be within bounds
[lb_out, ub_out] = GS_out.getRanges();
Y_center = L.evaluate(GS_in.V(:,:,1), A);
tol = 1e-6;
assert(all(Y_center(:) >= lb_out(:) - tol), 'Center output should be >= lower bound');
assert(all(Y_center(:) <= ub_out(:) + tol), 'Center output should be <= upper bound');

%% 4) Random sample containment test
rng(42);
GS_out4 = L.reach(GS_in, A, 'approx-star');
[lb_out4, ub_out4] = GS_out4.getRanges();
tol4 = 1e-6;

num_samples = 100;
all_in_bounds = true;
for s = 1:num_samples
    % Generate random input within GraphStar bounds
    alpha = rand(GS_in.numPred, 1) .* (GS_in.pred_ub - GS_in.pred_lb) + GS_in.pred_lb;
    X_sample = GS_in.V(:, :, 1);
    for k = 1:GS_in.numPred
        X_sample = X_sample + alpha(k) * GS_in.V(:, :, k+1);
    end

    % Evaluate and check containment
    Y_sample = L.evaluate(X_sample, A);
    if ~(all(Y_sample(:) >= lb_out4(:) - tol4) && all(Y_sample(:) <= ub_out4(:) + tol4))
        all_in_bounds = false;
        fprintf('Sample %d FAILED containment check\n', s);
        break;
    end
end
assert(all_in_bounds, 'All 100 random samples should be within output bounds');

%% 5) Precision change test
L_prec = SAGEConvLayer('prec_test', W_node, W_edge, b);
L_prec.changeParamsPrecision('single');
assert(isa(L_prec.NodeWeights, 'single'), 'NodeWeights should be single precision');
assert(isa(L_prec.EdgeWeights, 'single'), 'EdgeWeights should be single precision');
assert(isa(L_prec.Bias, 'single'), 'Bias should be single precision');

L_prec.changeParamsPrecision('double');
assert(isa(L_prec.NodeWeights, 'double'), 'NodeWeights should be double precision');
assert(isa(L_prec.EdgeWeights, 'double'), 'EdgeWeights should be double precision');
assert(isa(L_prec.Bias, 'double'), 'Bias should be double precision');

disp('All SAGEConvLayer tests passed!');
