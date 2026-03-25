%% test_HuGINEConvLayer_subgraph.m
% Correctness test: subgraph-based reachability must produce identical
% output bounds for the target node as full-graph reachability.
%
% Tests:
%   1. khop_subgraph + extractSubgraph: predicate pruning works
%   2. Subgraph GNN reach matches full-graph reach for a target node
%   3. reachSubgraph API works end-to-end
%   4. Multi-layer subgraph correctness (2-layer + interleaved ReluLayer)
%
% Author: Anne Tumlin
% Date: 03/17/2026

addpath(genpath('../../../../engine'));

fprintf('=== test_HuGINEConvLayer_subgraph ===\n');

%% Shared setup: small synthetic HuGINEConvLayer network
% 6 nodes, chain graph with self-loops, F_in=4, hidden=6, F_out=3, E_in=2

rng(42);
N = 6; F_in = 4; H = 6; F_out = 3; E_in = 2;

% Random layer weights
W1 = randn(F_in, H) * 0.1;
b1 = randn(H, 1) * 0.1;
W2 = randn(H, F_out) * 0.1;
b2 = randn(F_out, 1) * 0.1;
W_edge = randn(E_in, F_in) * 0.1;
b_edge = randn(F_in, 1) * 0.1;

layer = HuGINEConvLayer('hugine1', W1, b1, W2, b2, W_edge, b_edge);

% Build a chain graph: 1-2-3-4-5-6 (bidirectional) + self-loops
adj_list = [1 2; 2 1; 2 3; 3 2; 3 4; 4 3; 4 5; 5 4; 5 6; 6 5; ...
            1 1; 2 2; 3 3; 4 4; 5 5; 6 6];
num_real_edges = 10;
E_mat = [randn(num_real_edges, E_in) * 0.1; zeros(N, E_in)];
edge_weights = ones(size(adj_list, 1), 1);

gnn = GNN({layer});
gnn.adj_list = adj_list;
gnn.E = E_mat;
gnn.edge_weights = edge_weights;

% Create input GraphStar: perturb features [1,2] on all nodes
X_center = randn(N, F_in) * 0.5;
eps_mat = zeros(N, F_in);
eps_val = 0.05;
eps_mat(:, 1:2) = eps_val;  % perturb features 1 and 2

GS_in = GraphStar(X_center, -eps_mat, eps_mat);

fprintf('  Network: 1-layer HuGINEConvLayer, %d nodes, chain graph\n', N);
fprintf('  Perturbation eps=%.3f on features [1,2]\n', eps_val);

%% Test 1: extractSubgraph predicate pruning
fprintf('\nTest 1: extractSubgraph predicate pruning...\n');

% Extract subgraph for nodes {3,4}
sub_nodes = [3; 4];
sub_gs = GS_in.extractSubgraph(sub_nodes);

% Full GraphStar has N*2 = 12 predicates (perturbing 2 features x 6 nodes)
% Subgraph {3,4} uses only 4 predicates (nodes 3 and 4, features 1 and 2)
assert(sub_gs.numPred <= 4, ...
    sprintf('Test 1 FAIL: expected <=4 predicates after pruning, got %d', sub_gs.numPred));
assert(sub_gs.numNodes == 2, ...
    sprintf('Test 1 FAIL: subgraph should have 2 nodes, got %d', sub_gs.numNodes));
assert(sub_gs.numFeatures == F_in, ...
    sprintf('Test 1 FAIL: subgraph should have %d features, got %d', F_in, sub_gs.numFeatures));

% The center values should match original
for ni = 1:2
    for fi = 1:F_in
        expected = GS_in.V(sub_nodes(ni), fi, 1);
        actual = sub_gs.V(ni, fi, 1);
        assert(abs(actual - expected) < 1e-10, ...
            sprintf('Test 1 FAIL: center mismatch at node %d feature %d', ni, fi));
    end
end

fprintf('  Test 1 PASS: predicate count=%d (<=4), center values match\n', sub_gs.numPred);

%% Test 2: Full-graph reach vs subgraph reach for target node
fprintf('\nTest 2: Full-graph vs subgraph reach for target node...\n');

reachOpts = struct('reachMethod', 'approx-star');
tolerance = 1e-4;

% Full-graph reachability
GS_full_out = gnn.reach(GS_in, reachOpts);

% Subgraph reachability for target node 3 (middle of chain)
target = 3;
[sub_nodes_t, sub_adj_t, sub_E_t, sub_ew_t, t_local] = ...
    khop_subgraph(target, 1, adj_list, E_mat, edge_weights);

sub_GS_in = GS_in.extractSubgraph(sub_nodes_t);

% Build subgraph GNN
sub_gnn = GNN({layer});
sub_gnn.adj_list = sub_adj_t;
sub_gnn.E = sub_E_t;
sub_gnn.edge_weights = sub_ew_t;

GS_sub_out = sub_gnn.reach(sub_GS_in, reachOpts);

% Compare output bounds for the target node on each output feature
all_match = true;
for fi = 1:F_out
    [full_lb, full_ub] = GS_full_out.getRange(target, fi);
    [sub_lb, sub_ub] = GS_sub_out.getRange(t_local, fi);

    lb_diff = abs(full_lb - sub_lb);
    ub_diff = abs(full_ub - sub_ub);

    if lb_diff > tolerance || ub_diff > tolerance
        fprintf('  MISMATCH at feature %d: full=[%.6f,%.6f] sub=[%.6f,%.6f]\n', ...
            fi, full_lb, full_ub, sub_lb, sub_ub);
        all_match = false;
    end
end

assert(all_match, 'Test 2 FAIL: subgraph bounds differ from full-graph bounds');
fprintf('  Test 2 PASS: subgraph bounds match full-graph bounds (tol=%.0e)\n', tolerance);

%% Test 3: reachSubgraph API end-to-end
fprintf('\nTest 3: reachSubgraph API...\n');

target_nodes_test = [2; 3; 4];
[node_results, sg_info] = gnn.reachSubgraph(GS_in, target_nodes_test, reachOpts);

assert(length(node_results) == 3, 'Test 3 FAIL: expected 3 node results');
assert(length(sg_info) == 3, 'Test 3 FAIL: expected 3 subgraph_info entries');

for ti = 1:3
    t = target_nodes_test(ti);
    t_local_api = sg_info(ti).target_local_idx;
    gs_out_api = node_results{ti};

    % Compare against full-graph output for this target node
    for fi = 1:F_out
        [full_lb, full_ub] = GS_full_out.getRange(t, fi);
        [api_lb, api_ub] = gs_out_api.getRange(t_local_api, fi);

        assert(abs(full_lb - api_lb) < tolerance, ...
            sprintf('Test 3 FAIL: lb mismatch for target %d, feature %d', t, fi));
        assert(abs(full_ub - api_ub) < tolerance, ...
            sprintf('Test 3 FAIL: ub mismatch for target %d, feature %d', t, fi));
    end

    fprintf('  Node %d: subgraph size=%d nodes/%d edges, time=%.4fs\n', ...
        t, sg_info(ti).n_sub_nodes, sg_info(ti).n_sub_edges, sg_info(ti).time);
end

fprintf('  Test 3 PASS: reachSubgraph API produces correct bounds for all targets\n');

%% Test 4: Multi-layer subgraph correctness (2-layer with interleaved ReluLayer)
fprintf('\nTest 4: Multi-layer subgraph correctness...\n');

% Build 2-layer network: HuGINEConv -> ReLU -> HuGINEConv
W1b = randn(F_out, H) * 0.1;
b1b = randn(H, 1) * 0.1;
W2b = randn(H, F_out) * 0.1;
b2b = randn(F_out, 1) * 0.1;
W_edge_b = randn(E_in, F_out) * 0.1;
b_edge_b = randn(F_out, 1) * 0.1;

layer2 = HuGINEConvLayer('hugine2', W1b, b1b, W2b, b2b, W_edge_b, b_edge_b);
relu_layer = ReluLayer();

gnn2 = GNN({layer, relu_layer, layer2});
gnn2.adj_list = adj_list;
gnn2.E = E_mat;
gnn2.edge_weights = edge_weights;

% Full-graph reach
GS_full_out2 = gnn2.reach(GS_in, reachOpts);

% Subgraph reach for target node 3 (k=2 for 2 message-passing layers)
target2 = 3;
[node_results2, sg_info2] = gnn2.reachSubgraph(GS_in, target2, reachOpts);

t_local2 = sg_info2.target_local_idx;
gs_out2 = node_results2{1};

all_match2 = true;
for fi = 1:F_out
    [full_lb2, full_ub2] = GS_full_out2.getRange(target2, fi);
    [sub_lb2, sub_ub2] = gs_out2.getRange(t_local2, fi);

    if abs(full_lb2 - sub_lb2) > tolerance || abs(full_ub2 - sub_ub2) > tolerance
        fprintf('  MISMATCH at feature %d: full=[%.6f,%.6f] sub=[%.6f,%.6f]\n', ...
            fi, full_lb2, full_ub2, sub_lb2, sub_ub2);
        all_match2 = false;
    end
end

assert(all_match2, 'Test 4 FAIL: multi-layer subgraph bounds differ from full-graph');
fprintf('  Test 4 PASS: 2-layer subgraph matches full-graph (k=2, %d nodes/%d edges)\n', ...
    sg_info2.n_sub_nodes, sg_info2.n_sub_edges);

%% Summary
fprintf('\n=== All HuGINEConvLayer subgraph verification tests PASSED ===\n');
