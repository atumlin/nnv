% test_sage_mutag.m - Test SAGEConvLayer with MUTAG model from SCIP-MPNN
run('code/nnv/startup_nnv.m');

gnn_file = '/home/verivital/Anne/dev/SCIP-MPNN-v1.0/christopherhojny-SCIP-MPNN-0b8d73c/data_experiments/gnn_instances/model_MUTAG.gnn';
gcinfo_file = '/home/verivital/Anne/dev/SCIP-MPNN-v1.0/christopherhojny-SCIP-MPNN-0b8d73c/data_experiments/graph_classification_instances/graph_MUTAG_0.gcinfo';

[gnn, graph_data] = gnn_from_scipmpnn(gnn_file, gcinfo_file);

fprintf('\n=== GNN Info ===\n');
disp(gnn);

fprintf('\n=== Graph Data ===\n');
fprintf('Nodes: %d, Features: %d, Classes: %d, Label: %d\n', ...
    graph_data.n_nodes, graph_data.n_features, graph_data.n_classes, graph_data.true_label);

fprintf('\n=== Forward Evaluation ===\n');
Y = gnn.evaluate(graph_data.X);
fprintf('Output Y: [%.6f, %.6f]\n', Y(1), Y(2));
predicted = find(Y == max(Y)) - 1;
fprintf('Predicted class: %d (true: %d)\n', predicted, graph_data.true_label);

fprintf('\n=== Reachability Test (epsilon=0.01) ===\n');
epsilon = 0.01;
X = graph_data.X;
eps_matrix = epsilon * ones(size(X));
GS_in = GraphStar(X, -eps_matrix, eps_matrix);

reachOpts = struct('reachMethod', 'approx-star');
t = tic;
GS_out = gnn.reach(GS_in, reachOpts);
t_reach = toc(t);

[lb, ub] = GS_out.getRanges();
fprintf('Reachability time: %.4fs\n', t_reach);
fprintf('Output bounds: lb=[%.6f, %.6f], ub=[%.6f, %.6f]\n', lb(1), lb(2), ub(1), ub(2));

% Soundness check
Y_center = gnn.evaluate(GS_in.V(:,:,1));
tol = 1e-6;
assert(all(Y_center(:) >= lb(:) - tol), 'LB soundness fail');
assert(all(Y_center(:) <= ub(:) + tol), 'UB soundness fail');
fprintf('Soundness: PASSED\n');

% Classification verification
if lb(predicted+1) > max(ub(setdiff(1:length(ub), predicted+1)))
    fprintf('Classification: VERIFIED ROBUST\n');
else
    fprintf('Classification: UNKNOWN (bounds overlap)\n');
end

fprintf('\nDone.\n');
