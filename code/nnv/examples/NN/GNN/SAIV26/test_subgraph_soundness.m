function test_subgraph_soundness(varargin)
% TEST_SUBGRAPH_SOUNDNESS - Validate subgraph verification for all architectures.
%
% For each architecture x grid combination, validates three properties:
%
%   1. BOUND EQUIVALENCE: subgraph output bounds must match full-graph
%      bounds within LP tolerance (validates k-hop locality correctness).
%
%   2. GNN OUTPUT EQUALITY: for random perturbations within the epsilon ball,
%      GNN(X+dX)[target] must be identical on full graph vs subgraph
%      (validates that subgraph computation is exact, not approximate).
%
%   3. REACHABILITY SOUNDNESS: each random sample must lie within the
%      subgraph reachable bounds (validates over-approximation).
%
% Tests GCN, SAGEConv, GINE Linear, and GINE Conv on IEEE24 PF models
% (small enough for full-graph comparison in reasonable time).
%
% Usage:
%   test_subgraph_soundness()
%   test_subgraph_soundness('n_graphs', 5, 'n_samples', 50)
%   test_subgraph_soundness('arch', 'gcn')
%   test_subgraph_soundness('grid', 'ieee24')
%
% Author: Anne Tumlin
% Date: 03/13/2026

p = inputParser;
addParameter(p, 'n_graphs', 3, @isnumeric);
addParameter(p, 'n_samples', 20, @isnumeric);
addParameter(p, 'epsilon', 0.005, @isnumeric);
addParameter(p, 'tol', 1e-3, @isnumeric);
addParameter(p, 'arch', 'all', @ischar);
addParameter(p, 'grid', 'ieee24', @ischar);
parse(p, varargin{:});

n_graphs  = p.Results.n_graphs;
n_samples = p.Results.n_samples;
epsilon   = p.Results.epsilon;
tol       = p.Results.tol;

all_archs = {'gcn', 'sage', 'gine_linear', 'gine_conv'};
all_labels = {'GCN', 'SAGEConv', 'GINE Linear', 'GINE Conv'};

if strcmp(p.Results.arch, 'all')
    archs = all_archs;
    labels = all_labels;
else
    idx = find(strcmp(all_archs, p.Results.arch));
    if isempty(idx)
        error('Unknown arch: %s. Options: %s', p.Results.arch, strjoin(all_archs, ', '));
    end
    archs = all_archs(idx);
    labels = all_labels(idx);
end

grid = p.Results.grid;
grid_upper = upper(grid);

fprintf('\n========================================================\n');
fprintf('  Subgraph Verification Soundness Check\n');
fprintf('========================================================\n');
fprintf('Grid: %s | Graphs: %d | Samples/graph: %d | eps=%.3f | tol=%.0e\n', ...
    grid_upper, n_graphs, n_samples, epsilon, tol);
fprintf('Architectures: %s\n', strjoin(labels, ', '));
fprintf('========================================================\n\n');

scriptDir = fileparts(mfilename('fullpath'));
perturb_features = [1, 2];  % Power injections
voltage_idx = 3;
bus_type_idx = 4;

overall_pass = true;

for ai = 1:length(archs)
    arch = archs{ai};
    label = labels{ai};

    mat_file = sprintf('%s_pf_%s.mat', arch, grid);
    mat_path = fullfile(scriptDir, 'PowerFlow', grid_upper, 'models', mat_file);

    if ~isfile(mat_path)
        fprintf('SKIP: %s not found\n\n', mat_file);
        continue;
    end

    fprintf('############################################\n');
    fprintf('  %s on %s\n', label, grid_upper);
    fprintf('############################################\n');

    [gnn, test_data, norm_stats] = gnn2nnv(mat_path);
    fprintf('  Loaded (%d layers)\n\n', gnn.numLayers);

    is_gine = strcmp(arch, 'gine_conv') || strcmp(arch, 'gine_linear');

    reachOpts = struct('reachMethod', 'approx-star');

    % Counters for this architecture
    arch_total_nodes = 0;
    arch_max_bound_diff = 0;
    arch_eq_failures = 0;
    arch_sound_failures = 0;
    arch_total_evals = 0;

    for gi = 1:min(n_graphs, test_data.num_graphs)
        X = test_data.X_all{gi};

        % Identify PQ bus nodes for target selection
        if isfield(norm_stats, 'X_max')
            X_phys = X .* norm_stats.X_max;
        else
            X_phys = X;
        end
        pq_nodes = find(round(X_phys(:, bus_type_idx)) == 1);
        n_pq = length(pq_nodes);
        arch_total_nodes = arch_total_nodes + n_pq;

        % Build perturbation
        range_per_col = max(X) - min(X);
        eps_mat = zeros(size(X));
        for f = perturb_features
            if f <= size(X, 2)
                eps_mat(:, f) = range_per_col(f) * epsilon;
            end
        end

        GS_in = GraphStar(X, -eps_mat, eps_mat);

        fprintf('--- Graph %d (%d nodes, %d PQ buses) ---\n', gi, size(X,1), n_pq);

        % --- CHECK 1: Bound equivalence ---
        fprintf('  Full-graph reach...');
        GS_full_out = gnn.reach(GS_in, reachOpts);
        fprintf('done. Subgraph reach...');
        [node_outputs, sg_info] = gnn.reachSubgraph(GS_in, pq_nodes, reachOpts);
        fprintf('done.\n');

        max_diff_graph = 0;
        for ti = 1:n_pq
            t = pq_nodes(ti);
            t_local = sg_info(ti).target_local_idx;

            [full_lb, full_ub] = GS_full_out.getRange(t, voltage_idx);
            [sub_lb,  sub_ub ] = node_outputs{ti}.getRange(t_local, voltage_idx);

            diff = max(abs(full_lb - sub_lb), abs(full_ub - sub_ub));
            max_diff_graph = max(max_diff_graph, diff);
            arch_max_bound_diff = max(arch_max_bound_diff, diff);
        end

        % Report subgraph sizes
        sg_sizes = [sg_info.n_sub_nodes];
        fprintf('  Subgraph sizes: min=%d, max=%d, mean=%.1f (full=%d)\n', ...
            min(sg_sizes), max(sg_sizes), mean(sg_sizes), size(X,1));
        fprintf('  Check 1 (bound equivalence): max diff = %.2e  %s\n', ...
            max_diff_graph, pass_str(max_diff_graph <= tol));

        % --- CHECK 2+3: Point evaluation ---
        k = count_mp_layers(gnn);
        sub_gnns = cell(n_pq, 1);
        sub_node_sets = cell(n_pq, 1);

        for ti = 1:n_pq
            t = pq_nodes(ti);
            if is_gine
                [sub_nodes, sub_adj, sub_E, sub_ew, ~] = ...
                    khop_subgraph(t, k, gnn.adj_list, gnn.E, gnn.edge_weights);
                sub_node_sets{ti} = sub_nodes;
                sg = GNN(gnn.Layers);
                sg.adj_list = sub_adj;
                sg.E = sub_E;
                sg.edge_weights = sub_ew;
            else
                [sub_nodes, sub_A, ~] = khop_subgraph_matrix(t, k, gnn.A_norm);
                sub_node_sets{ti} = sub_nodes;
                sg = GNN(gnn.Layers);
                sg.A_norm = sub_A;
            end
            sub_gnns{ti} = sg;
        end

        n_eq_fail_graph = 0;
        n_sound_fail_graph = 0;
        rng(gi * 42);

        for si = 1:n_samples
            dX = (2 * rand(size(X)) - 1) .* eps_mat;
            X_sample = X + dX;

            % Full-graph evaluation
            Y_full = gnn.evaluate(X_sample);

            for ti = 1:n_pq
                t = pq_nodes(ti);
                t_local = sg_info(ti).target_local_idx;

                % Subgraph evaluation
                X_sub_sample = X_sample(sub_node_sets{ti}, :);
                Y_sub = sub_gnns{ti}.evaluate(X_sub_sample);

                y_full = Y_full(t, voltage_idx);
                y_sub  = Y_sub(t_local, voltage_idx);

                % Check 2: outputs must be identical
                eq_err = abs(y_full - y_sub);
                if eq_err > 1e-8
                    arch_eq_failures = arch_eq_failures + 1;
                    n_eq_fail_graph = n_eq_fail_graph + 1;
                    if n_eq_fail_graph <= 3
                        fprintf('    EQ FAIL node %d sample %d: diff=%.2e\n', t, si, eq_err);
                    end
                end

                % Check 3: sample within reachable bounds
                [sub_lb_v, sub_ub_v] = node_outputs{ti}.getRange(t_local, voltage_idx);
                if y_sub < sub_lb_v - 1e-6 || y_sub > sub_ub_v + 1e-6
                    arch_sound_failures = arch_sound_failures + 1;
                    n_sound_fail_graph = n_sound_fail_graph + 1;
                    if n_sound_fail_graph <= 3
                        fprintf('    SOUND FAIL node %d sample %d: y=%.6f not in [%.6f, %.6f]\n', ...
                            t, si, y_sub, sub_lb_v, sub_ub_v);
                    end
                end

                arch_total_evals = arch_total_evals + 1;
            end
        end

        fprintf('  Check 2 (output equality):     %d/%d match  %s\n', ...
            n_pq*n_samples - n_eq_fail_graph, n_pq*n_samples, ...
            pass_str(n_eq_fail_graph == 0));
        fprintf('  Check 3 (reachability sound):   %d/%d within bounds  %s\n\n', ...
            n_pq*n_samples - n_sound_fail_graph, n_pq*n_samples, ...
            pass_str(n_sound_fail_graph == 0));
    end

    % Architecture summary
    arch_pass = arch_max_bound_diff <= tol && arch_eq_failures == 0 && arch_sound_failures == 0;
    if ~arch_pass, overall_pass = false; end

    fprintf('--- %s Summary ---\n', label);
    fprintf('  PQ nodes tested:  %d\n', arch_total_nodes);
    fprintf('  Sample evals:     %d\n', arch_total_evals);
    fprintf('  [1] Bound equiv:  max diff = %.2e  %s\n', ...
        arch_max_bound_diff, pass_str(arch_max_bound_diff <= tol));
    fprintf('  [2] Output equal: %d failures  %s\n', ...
        arch_eq_failures, pass_str(arch_eq_failures == 0));
    fprintf('  [3] Soundness:    %d violations  %s\n', ...
        arch_sound_failures, pass_str(arch_sound_failures == 0));
    fprintf('  RESULT: %s\n\n', pass_str(arch_pass));
end

fprintf('========================================================\n');
if overall_pass
    fprintf('  OVERALL: ALL ARCHITECTURES PASSED\n');
else
    fprintf('  OVERALL: SOME ARCHITECTURES FAILED\n');
end
fprintf('========================================================\n');

end


%% Helper: count message-passing layers
function k = count_mp_layers(gnn)
    k = 0;
    for i = 1:gnn.numLayers
        L = gnn.Layers{i};
        if isa(L, 'GINELayer') || isa(L, 'GINEConvLayer') || isa(L, 'GINEConvLayerOptimized') || ...
           isa(L, 'GCNLayer') || isa(L, 'SAGEConvLayer')
            k = k + 1;
        end
    end
end


%% Helper: pass/fail string
function s = pass_str(cond)
    if cond, s = 'PASS'; else, s = 'FAIL'; end
end
