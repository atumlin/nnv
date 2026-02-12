%% Quick test: exact-star reachMethod on small MUTAG graphs at eps=0.1
% WARNING: exact-star is exponential in the number of unstable neurons.
% This may take a very long time or run out of memory.
scip_base = '/home/verivital/Anne/dev/SCIP-MPNN-v1.0/christopherhojny-SCIP-MPNN-0b8d73c';
gnn_path = fullfile(scip_base, 'data_experiments', 'gnn_instances', 'model_MUTAG.gnn');
gcinfo_dir = fullfile(scip_base, 'data_experiments', 'graph_classification_instances');

[gnn, ~] = gnn_from_scipmpnn(gnn_path);
eps_val = 0.01;  % tiny epsilon to reduce unstable neuron count

reachOpts = struct('reachMethod', 'exact-star');

% Try just one small graph first
graph_ids = [4];  % Graph 4 has 11 nodes (smallest of the first 5)
for i = 1:length(graph_ids)
    gid = graph_ids(i);
    gcinfo_path = fullfile(gcinfo_dir, sprintf('graph_MUTAG_%d.gcinfo', gid));
    [~, gd] = gnn_from_scipmpnn(gnn_path, gcinfo_path);

    fprintf('Graph %d: %d nodes, %d features\n', gid, size(gd.X, 1), size(gd.X, 2));

    gnn.A_norm = gd.A;
    Y = gnn.evaluate(gd.X);
    [~, pred] = max(Y); pred = pred - 1;
    fprintf('  Nominal output: [%.4f, %.4f], pred=%d\n', Y(1), Y(2), pred);

    X_lb = max(gd.X - eps_val, 0);
    X_ub = min(gd.X + eps_val, 1);
    gs_in = GraphStar(X_lb, X_ub);

    fprintf('  Starting exact-star reachability...\n');
    t = tic;
    gs_out = gnn.reach(gs_in, reachOpts);
    reach_time = toc(t);

    % exact-star may return multiple star sets
    if iscell(gs_out)
        fprintf('  Got %d star sets from exact-star\n', length(gs_out));
    else
        fprintf('  Got single output set\n');
        [lb, ub] = gs_out.getRanges();

        if pred == 0
            verified = lb(1,1) > ub(1,2);
        else
            verified = lb(1,2) > ub(1,1);
        end

        gap = max(ub(1,:) - lb(1,:));
        if verified, status = 'VERIFIED'; else, status = 'UNKNOWN'; end
        fprintf('  bounds=[%.4f..%.4f, %.4f..%.4f], gap=%.4f, %s (%.3fs)\n', ...
            lb(1,1), ub(1,1), lb(1,2), ub(1,2), gap, status, reach_time);
    end
end
