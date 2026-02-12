% test_sage_mutag_batch.m - Batch test on multiple MUTAG graphs
run('code/nnv/startup_nnv.m');

gnn_file = '/home/verivital/Anne/dev/SCIP-MPNN-v1.0/christopherhojny-SCIP-MPNN-0b8d73c/data_experiments/gnn_instances/model_MUTAG.gnn';
gcinfo_dir = '/home/verivital/Anne/dev/SCIP-MPNN-v1.0/christopherhojny-SCIP-MPNN-0b8d73c/data_experiments/graph_classification_instances';

% Get list of MUTAG gcinfo files
files = dir(fullfile(gcinfo_dir, 'graph_MUTAG_*.gcinfo'));
fprintf('Found %d MUTAG graphs\n', length(files));

% Test first 10
n_test = min(10, length(files));
epsilon = 0.01;
results = {};

for i = 1:n_test
    gcinfo_path = fullfile(gcinfo_dir, files(i).name);
    [gnn, gd] = gnn_from_scipmpnn(gnn_file, gcinfo_path);

    Y = gnn.evaluate(gd.X);
    predicted = find(Y == max(Y)) - 1;
    correct = (predicted == gd.true_label);

    % Reachability
    eps_matrix = epsilon * ones(size(gd.X));
    GS_in = GraphStar(gd.X, -eps_matrix, eps_matrix);
    reachOpts = struct('reachMethod', 'approx-star');
    t = tic;
    GS_out = gnn.reach(GS_in, reachOpts);
    t_reach = toc(t);
    [lb, ub] = GS_out.getRanges();

    % Soundness
    Y_center = gnn.evaluate(GS_in.V(:,:,1));
    tol = 1e-6;
    sound = all(Y_center(:) >= lb(:) - tol) && all(Y_center(:) <= ub(:) + tol);

    % Verification
    if lb(predicted+1) > max(ub(setdiff(1:length(ub), predicted+1)))
        status = 'VERIFIED';
    else
        status = 'UNKNOWN';
    end

    fprintf('%s: %d nodes, pred=%d (true=%d, %s), reach=%.3fs, %s, sound=%d\n', ...
        files(i).name, gd.n_nodes, predicted, gd.true_label, ...
        ternary(correct, 'correct', 'wrong'), t_reach, status, sound);
end

function s = ternary(cond, a, b)
    if cond; s = a; else; s = b; end
end
