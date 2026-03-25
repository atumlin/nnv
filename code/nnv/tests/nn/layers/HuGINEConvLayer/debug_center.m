addpath(genpath('../../../../engine'));

rng(42);
F_in = 4; hidden = 6; F_out = 3; E_in = 2;
W1 = randn(F_in, hidden) * 0.1; b1 = randn(hidden, 1) * 0.1;
W2 = randn(hidden, F_out) * 0.1; b2 = randn(F_out, 1) * 0.1;
W_edge = randn(E_in, F_in) * 0.1; b_edge = randn(F_in, 1) * 0.1;
L = HuGINEConvLayer('test', W1, b1, W2, b2, W_edge, b_edge);

numNodes = 4; base_edges = 5;
adj_list = [1 2; 1 3; 2 3; 3 4; 4 1; 1 1; 2 2; 3 3; 4 4];
numEdges = size(adj_list, 1);
E_base = randn(base_edges, E_in) * 0.1;
E = [E_base; zeros(numNodes, E_in)];

NF = randn(numNodes, F_in) * 0.5;
LB = -0.1 * ones(numNodes, F_in);
UB = 0.1 * ones(numNodes, F_in);
GS_in = GraphStar(NF, LB, UB);

center_in = GS_in.V(:,:,1);
Y_eval = L.evaluate(center_in, E, adj_list);

GS_out = L.reach(GS_in, E, adj_list, 'approx-star');
center_out = GS_out.V(:,:,1);

fprintf('Max diff: %.2e\n', max(abs(center_out(:) - Y_eval(:))));
fprintf('Y_eval:\n'); disp(Y_eval);
fprintf('center_out:\n'); disp(center_out);

% Trace step by step through reach
src_nodes = adj_list(:,1);
dst_nodes = adj_list(:,2);

% Edge projection (constant)
E_proj = E * L.EdgeProjWeights + L.EdgeProjBias';

% Gather center to edges
V_center_edges = center_in(src_nodes, :);

% Add E_proj
msg_center = V_center_edges + E_proj;

% Aggregate
agg_center = zeros(numNodes, F_in);
for e = 1:numEdges
    agg_center(dst_nodes(e), :) = agg_center(dst_nodes(e), :) + msg_center(e, :);
end

% MLP
H_center = max(0, agg_center * W1 + b1');
Y_manual = H_center * W2 + b2';

fprintf('Y_manual:\n'); disp(Y_manual);
fprintf('Max diff eval vs manual: %.2e\n', max(abs(Y_eval(:) - Y_manual(:))));

% Now check what reach produces at center
% V_edge center should be
V_edge_reach = GS_in.V(src_nodes, :, 1);
fprintf('Max diff gather: %.2e\n', max(abs(V_edge_reach(:) - V_center_edges(:))));
