%% Quick cross-validation: NNV vs SCIP-MPNN on 5 MUTAG graphs
scip_base = '/home/verivital/Anne/dev/SCIP-MPNN-v1.0/christopherhojny-SCIP-MPNN-0b8d73c';
gnn_path = fullfile(scip_base, 'data_experiments', 'gnn_instances', 'model_MUTAG.gnn');
gcinfo_dir = fullfile(scip_base, 'data_experiments', 'graph_classification_instances');

[gnn, ~] = gnn_from_scipmpnn(gnn_path);
eps_val = 0.1;

graph_ids = [0, 1, 2, 3, 4];
for i = 1:length(graph_ids)
    gid = graph_ids(i);
    gcinfo_path = fullfile(gcinfo_dir, sprintf('graph_MUTAG_%d.gcinfo', gid));
    [~, gd] = gnn_from_scipmpnn(gnn_path, gcinfo_path);

    gnn.A_norm = gd.A;
    Y = gnn.evaluate(gd.X);
    [~, pred] = max(Y); pred = pred - 1;

    X_lb = max(gd.X - eps_val, 0);
    X_ub = min(gd.X + eps_val, 1);
    gs_in = GraphStar(X_lb, X_ub);

    t = tic;
    gs_out = gnn.reach(gs_in);
    reach_time = toc(t);

    [lb, ub] = gs_out.getRanges();

    if pred == 0
        verified = lb(1,1) > ub(1,2);
    else
        verified = lb(1,2) > ub(1,1);
    end

    gap = max(ub(1,:) - lb(1,:));
    fprintf('Graph %d: pred=%d, scores=[%.4f, %.4f], bounds=[%.4f..%.4f, %.4f..%.4f], gap=%.4f, %s (%.3fs)\n', ...
        gid, pred, Y(1), Y(2), lb(1,1), ub(1,1), lb(1,2), ub(1,2), gap, ...
        ternary(verified, 'VERIFIED', 'UNKNOWN'), reach_time);
end

function s = ternary(cond, a, b)
    if cond, s = a; else, s = b; end
end
