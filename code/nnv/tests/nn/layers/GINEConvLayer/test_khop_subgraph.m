%% test_khop_subgraph.m
% Unit tests for khop_subgraph utility function.
%
% Tests k-hop neighborhood extraction on known small graphs where
% correct answers can be verified by hand.
%
% Author: Anne Tumlin
% Date: 03/11/2026

addpath(genpath('../../../../engine'));

fprintf('=== test_khop_subgraph ===\n');
all_passed = true;

%% Shared setup: 6-node chain graph 1-2-3-4-5-6
% Edges (bidirectional): (1,2),(2,3),(3,4),(4,5),(5,6)
adj_chain = [1 2; 2 1; 2 3; 3 2; 3 4; 4 3; 4 5; 5 4; 5 6; 6 5];
m_chain = size(adj_chain, 1);
E_chain = ones(m_chain, 2);       % dummy edge features
ew_chain = ones(m_chain, 1);      % uniform weights

%% Test 1: 1-hop from node 3 in chain
[sub_nodes, sub_adj, sub_E, sub_ew, t_local] = ...
    khop_subgraph(3, 1, adj_chain, E_chain, ew_chain);

expected_nodes = [2; 3; 4];
assert(isequal(sub_nodes(:), expected_nodes), ...
    'Test 1 FAIL: 1-hop from node 3 should give nodes {2,3,4}');
assert(t_local == 2, ...
    'Test 1 FAIL: node 3 should be at local index 2 in {2,3,4}');
assert(~isempty(sub_adj), 'Test 1 FAIL: subgraph should have edges');
% All subgraph edge endpoints should be in [1, length(sub_nodes)]
assert(all(sub_adj(:) >= 1) && all(sub_adj(:) <= length(sub_nodes)), ...
    'Test 1 FAIL: remapped edge indices out of range');
fprintf('  Test 1 PASS: 1-hop from node 3 in chain\n');

%% Test 2: 2-hop from node 3 in chain
[sub_nodes, sub_adj, ~, ~, t_local] = ...
    khop_subgraph(3, 2, adj_chain, E_chain, ew_chain);

expected_nodes = [1; 2; 3; 4; 5];
assert(isequal(sub_nodes(:), expected_nodes), ...
    'Test 2 FAIL: 2-hop from node 3 should give nodes {1,2,3,4,5}');
assert(t_local == 3, ...
    'Test 2 FAIL: node 3 should be at local index 3 in {1,2,3,4,5}');
fprintf('  Test 2 PASS: 2-hop from node 3 in chain\n');

%% Test 3: 3-hop from node 3 covers entire chain
[sub_nodes, ~, ~, ~, t_local] = ...
    khop_subgraph(3, 3, adj_chain, E_chain, ew_chain);

assert(isequal(sub_nodes(:), (1:6)'), ...
    'Test 3 FAIL: 3-hop from node 3 in 6-chain should cover all nodes');
assert(t_local == 3, 'Test 3 FAIL: node 3 local index should be 3');
fprintf('  Test 3 PASS: 3-hop from node 3 covers full chain\n');

%% Test 4: boundary node (node 1, 2-hop in chain)
[sub_nodes, sub_adj, ~, ~, t_local] = ...
    khop_subgraph(1, 2, adj_chain, E_chain, ew_chain);

expected_nodes = [1; 2; 3];
assert(isequal(sub_nodes(:), expected_nodes), ...
    'Test 4 FAIL: 2-hop from node 1 should give {1,2,3}');
assert(t_local == 1, 'Test 4 FAIL: node 1 local index should be 1');
fprintf('  Test 4 PASS: boundary node 1, 2-hop\n');

%% Test 5: edge count in subgraph
[sub_nodes, sub_adj, sub_E, sub_ew, ~] = ...
    khop_subgraph(3, 1, adj_chain, E_chain, ew_chain);
% 1-hop from node 3: nodes {2,3,4}, edges {(2,3),(3,2),(3,4),(4,3)} → 4 edges
assert(size(sub_adj, 1) == 4, ...
    sprintf('Test 5 FAIL: expected 4 edges in subgraph, got %d', size(sub_adj, 1)));
assert(size(sub_E, 1) == size(sub_adj, 1), ...
    'Test 5 FAIL: sub_E and sub_adj should have same number of rows');
assert(length(sub_ew) == size(sub_adj, 1), ...
    'Test 5 FAIL: sub_ew and sub_adj should have same length');
fprintf('  Test 5 PASS: edge count in subgraph correct\n');

%% Test 6: star graph (node 1 at center, connects to all others)
% Edges: 1-2, 1-3, 1-4, 1-5 (bidirectional)
adj_star = [1 2; 2 1; 1 3; 3 1; 1 4; 4 1; 1 5; 5 1];
E_star = ones(size(adj_star, 1), 1);
ew_star = ones(size(adj_star, 1), 1);

% 1-hop from leaf node 2: should get {1,2} (center + leaf)
[sub_nodes, ~, ~, ~, t_local] = khop_subgraph(2, 1, adj_star, E_star, ew_star);
assert(isequal(sub_nodes(:), [1;2]), ...
    'Test 6a FAIL: 1-hop from leaf 2 in star should give {1,2}');
assert(t_local == 2, 'Test 6a FAIL: leaf 2 should be at local index 2');

% 2-hop from leaf node 2: should get all 5 nodes
[sub_nodes, ~, ~, ~, ~] = khop_subgraph(2, 2, adj_star, E_star, ew_star);
assert(isequal(sub_nodes(:), (1:5)'), ...
    'Test 6b FAIL: 2-hop from leaf in star should cover all nodes');
fprintf('  Test 6 PASS: star graph neighborhoods\n');

%% Test 7: index remapping correctness
% Verify that remapped sub_adj correctly references local node positions
[sub_nodes, sub_adj, ~, ~, ~] = khop_subgraph(3, 1, adj_chain, E_chain, ew_chain);
% sub_nodes = [2,3,4], so local index 1=node2, 2=node3, 3=node4
% Check that the edge (3,4) appears as (2,3) in local indices
node_map_test = zeros(6,1);
node_map_test(sub_nodes) = 1:length(sub_nodes);
% Reconstruct global edges from local
global_edges_check = [sub_nodes(sub_adj(:,1)), sub_nodes(sub_adj(:,2))];
% All global edges should exist in original adj_chain
for e = 1:size(global_edges_check, 1)
    s = global_edges_check(e,1);
    d = global_edges_check(e,2);
    found = any(adj_chain(:,1)==s & adj_chain(:,2)==d);
    assert(found, sprintf('Test 7 FAIL: remapped edge (%d,%d) not in original graph', s, d));
end
fprintf('  Test 7 PASS: index remapping consistent with original graph\n');

%% Summary
if all_passed
    fprintf('\nAll khop_subgraph tests PASSED.\n');
end
