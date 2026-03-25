%% soundness_check_subgraph.m
% Soundness validation for subgraph-based GINEConv verification.
%
% Validates three properties on real IEEE24 GINE Conv data:
%
%   1. BOUND EQUIVALENCE: subgraph voltage bounds must match full-graph
%      bounds within LP tolerance (validates k-hop locality correctness).
%
%   2. GNN OUTPUT EQUALITY: for random ΔX within the perturbation ball,
%      GNN(X + ΔX)[target] must be identical on full graph vs subgraph
%      (validates that subgraph computation is exact, not approximate).
%
%   3. REACHABILITY SOUNDNESS: each random sample point must lie within
%      the subgraph reachable bounds (validates over-approximation).
%
% Usage:
%   soundness_check_subgraph()
%   soundness_check_subgraph('n_graphs', 5, 'n_samples', 50)
%
% Author: Anne Tumlin
% Date: 03/11/2026

function soundness_check_subgraph(varargin)

p = inputParser;
addParameter(p, 'n_graphs', 3, @isnumeric);
addParameter(p, 'n_samples', 20, @isnumeric);
addParameter(p, 'epsilon', 0.005, @isnumeric);
addParameter(p, 'tol', 1e-3, @isnumeric);  % LP bound comparison tolerance
parse(p, varargin{:});

n_graphs  = p.Results.n_graphs;
n_samples = p.Results.n_samples;
epsilon   = p.Results.epsilon;
tol       = p.Results.tol;

fprintf('=== Soundness Check: Subgraph GINEConv Verification ===\n');
fprintf('  Graphs: %d | Samples/graph: %d | eps=%.3f | tol=%.0e\n\n', ...
    n_graphs, n_samples, epsilon, tol);

scriptDir = fileparts(mfilename('fullpath'));
mat_path  = fullfile(scriptDir, 'IEEE24', 'models', 'gine_conv_pf_ieee24.mat');

[gnn, test_data, norm_stats] = gnn2nnv(mat_path);
fprintf('  Loaded IEEE24 GINE Conv (%d layers)\n\n', gnn.numLayers);

voltage_idx  = 3;
bus_type_idx = 4;

reachOpts = struct('reachMethod', 'approx-star');

% Counters
total_pq_nodes   = 0;
max_bound_diff   = 0;
total_evals      = 0;
n_eq_failures    = 0;
n_sound_failures = 0;

for gi = 1:n_graphs
    X = test_data.X_all{gi};

    if isfield(norm_stats, 'X_max')
        X_phys = X .* norm_stats.X_max;
    else
        X_phys = X;
    end
    pq_nodes = find(round(X_phys(:, bus_type_idx)) == 1);
    n_pq = length(pq_nodes);
    total_pq_nodes = total_pq_nodes + n_pq;

    range_per_col = max(X) - min(X);
    eps_mat = zeros(size(X));
    eps_mat(:, 1) = range_per_col(1) * epsilon;
    eps_mat(:, 2) = range_per_col(2) * epsilon;

    GS_in = GraphStar(X, -eps_mat, eps_mat);

    fprintf('--- Graph %d (%d nodes, %d PQ buses) ---\n', gi, size(X,1), n_pq);

    % --- CHECK 1: Bound equivalence ---
    GS_full_out = gnn.reach(GS_in, reachOpts);
    [node_outputs, sg_info] = gnn.reachSubgraph(GS_in, pq_nodes, reachOpts);

    max_diff_graph = 0;
    for ti = 1:n_pq
        t       = pq_nodes(ti);
        t_local = sg_info(ti).target_local_idx;

        [full_lb, full_ub] = GS_full_out.getRange(t, voltage_idx);
        [sub_lb,  sub_ub ] = node_outputs{ti}.getRange(t_local, voltage_idx);

        diff = max(abs(full_lb - sub_lb), abs(full_ub - sub_ub));
        max_diff_graph = max(max_diff_graph, diff);
        max_bound_diff = max(max_bound_diff, diff);
    end
    fprintf('  Check 1 (bound equivalence): max diff = %.2e  %s\n', ...
        max_diff_graph, pass_str(max_diff_graph <= tol));

    % --- CHECK 2+3: Point containment ---
    % Sample random ΔX within perturbation bounds, evaluate GNN on both
    % full graph and each target node's subgraph, compare outputs.
    rng(gi * 42);

    % Pre-extract subgraph structures for each PQ node
    k = count_mp_layers(gnn);
    sub_gnns = cell(n_pq, 1);
    sub_node_sets = cell(n_pq, 1);
    for ti = 1:n_pq
        t = pq_nodes(ti);
        [sub_nodes, sub_adj, sub_E, sub_ew, ~] = ...
            khop_subgraph(t, k, gnn.adj_list, gnn.E, gnn.edge_weights);
        sub_node_sets{ti} = sub_nodes;
        sg = GNN(gnn.Layers);
        sg.adj_list    = sub_adj;
        sg.E           = sub_E;
        sg.edge_weights = sub_ew;
        sub_gnns{ti} = sg;
    end

    n_eq_fail_graph    = 0;
    n_sound_fail_graph = 0;

    for si = 1:n_samples
        % Random ΔX within [-eps_mat, +eps_mat]
        dX = (2 * rand(size(X)) - 1) .* eps_mat;
        X_sample = X + dX;

        % Full-graph evaluation
        Y_full = gnn.evaluate(X_sample);

        for ti = 1:n_pq
            t       = pq_nodes(ti);
            t_local = sg_info(ti).target_local_idx;

            % Subgraph evaluation on the same perturbed features
            X_sub_sample = X_sample(sub_node_sets{ti}, :);
            Y_sub = sub_gnns{ti}.evaluate(X_sub_sample);

            y_full = Y_full(t, voltage_idx);
            y_sub  = Y_sub(t_local, voltage_idx);

            % Check 2: GNN outputs must be identical
            eq_err = abs(y_full - y_sub);
            if eq_err > 1e-8
                n_eq_failures    = n_eq_failures + 1;
                n_eq_fail_graph  = n_eq_fail_graph + 1;
                if n_eq_fail_graph <= 2
                    fprintf('    EQ FAIL node %d sample %d: diff=%.2e\n', t, si, eq_err);
                end
            end

            % Check 3: subgraph sample must lie within reachable bounds
            [sub_lb_v, sub_ub_v] = node_outputs{ti}.getRange(t_local, voltage_idx);
            if y_sub < sub_lb_v - 1e-6 || y_sub > sub_ub_v + 1e-6
                n_sound_failures   = n_sound_failures + 1;
                n_sound_fail_graph = n_sound_fail_graph + 1;
                if n_sound_fail_graph <= 2
                    fprintf('    SOUND FAIL node %d sample %d: y=%.6f not in [%.6f, %.6f]\n', ...
                        t, si, y_sub, sub_lb_v, sub_ub_v);
                end
            end

            total_evals = total_evals + 1;
        end
    end

    fprintf('  Check 2 (GNN output equality):   %d/%d match  %s\n', ...
        n_pq*n_samples - n_eq_fail_graph, n_pq*n_samples, ...
        pass_str(n_eq_fail_graph == 0));
    fprintf('  Check 3 (reachability soundness): %d/%d within bounds  %s\n\n', ...
        n_pq*n_samples - n_sound_fail_graph, n_pq*n_samples, ...
        pass_str(n_sound_fail_graph == 0));
end

%% Final summary
fprintf('=== SUMMARY ===\n');
fprintf('  Total PQ bus verifications: %d\n', total_pq_nodes);
fprintf('  Total random sample evals:  %d\n', total_evals);
fprintf('\n');
fprintf('  [1] Bound equivalence (full vs subgraph):  max diff = %.2e  %s\n', ...
    max_bound_diff, pass_str(max_bound_diff <= tol));
fprintf('  [2] GNN output equality (random samples):  %d failures  %s\n', ...
    n_eq_failures, pass_str(n_eq_failures == 0));
fprintf('  [3] Reachability soundness (over-approx):  %d violations  %s\n', ...
    n_sound_failures, pass_str(n_sound_failures == 0));
fprintf('\n');

if max_bound_diff <= tol && n_eq_failures == 0 && n_sound_failures == 0
    fprintf('OVERALL RESULT: ALL SOUNDNESS CHECKS PASSED\n');
else
    fprintf('OVERALL RESULT: SOME CHECKS FAILED\n');
end

end


%% Helper: count message-passing layers
function k = count_mp_layers(gnn)
    k = 0;
    for i = 1:gnn.numLayers
        L = gnn.Layers{i};
        if isa(L,'GINELayer') || isa(L,'GINEConvLayer') || isa(L,'GINEConvLayerOptimized') || ...
           isa(L,'GCNLayer') || isa(L,'SAGEConvLayer')
            k = k + 1;
        end
    end
end


%% Helper: pass/fail string
function s = pass_str(cond)
    if cond, s = 'PASS'; else, s = 'FAIL'; end
end
