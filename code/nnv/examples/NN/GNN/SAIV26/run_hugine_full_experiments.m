function results = run_hugine_full_experiments(varargin)
% run_hugine_full_experiments - HuGINE (Hu et al. ICLR 2020) verification
%
% Runs gine_pretrain models across 3 grids x 2 tasks x 4 epsilons,
% plus edge perturbation trials (PF only, multiple edge epsilons).
% IEEE118 uses subgraph verification; IEEE24/39 use full-graph.
%
% Usage:
%   run_hugine_full_experiments()
%   run_hugine_full_experiments('num_graphs', 2)          % smoke test
%   run_hugine_full_experiments('grid', 'ieee24')         % single grid
%   run_hugine_full_experiments('task', 'pf')             % PF only
%   run_hugine_full_experiments('mode', 'node_only')      % skip edge
%   run_hugine_full_experiments('parallel', true)         % parfor

%% Parse arguments
p = inputParser;
addParameter(p, 'num_graphs', 100, @isnumeric);
addParameter(p, 'grid', 'all', @ischar);
addParameter(p, 'task', 'all', @ischar);
addParameter(p, 'mode', 'all', @ischar);       % 'node_only', 'node_edge', or 'all'
addParameter(p, 'parallel', false, @islogical);
addParameter(p, 'num_workers', 0, @isnumeric);
addParameter(p, 'node_epsilons', [1e-5, 1e-4, 1e-3, 1e-2], @isnumeric);
addParameter(p, 'edge_epsilons', [1e-3, 1e-2], @isnumeric);
parse(p, varargin{:});

%% Configuration
arch = 'gine_pretrain';
arch_label = 'GINE Pretrain';

if strcmp(p.Results.grid, 'all')
    grids = {'ieee24', 'ieee39', 'ieee118'};
else
    grids = {p.Results.grid};
end

if strcmp(p.Results.task, 'all')
    tasks = {'pf', 'opf'};
elseif strcmp(p.Results.task, 'pf')
    tasks = {'pf'};
elseif strcmp(p.Results.task, 'opf')
    tasks = {'opf'};
else
    error('task must be ''pf'', ''opf'', or ''all''');
end

switch p.Results.mode
    case 'node_only'
        perturbation_modes = {'node_only'};
    case 'node_edge'
        perturbation_modes = {'node_edge'};
    otherwise
        perturbation_modes = {'node_only', 'node_edge'};
end

epsilon_values = p.Results.node_epsilons;
edge_epsilons = p.Results.edge_epsilons;
num_graphs = p.Results.num_graphs;
perturb_node_features = [1, 2];  % P, Q
perturb_edge_features = [1];     % impedance

grid_v_bounds = struct( ...
    'ieee24', [0.95, 1.05], ...
    'ieee39', [0.94, 1.06], ...
    'ieee118', [0.94, 1.09]);

%% Parallel pool
use_parallel = p.Results.parallel;
num_workers = p.Results.num_workers;

% Kill any stale parallel pool from previous runs
pool = gcp('nocreate');
if ~isempty(pool) && ~use_parallel
    delete(pool);
    fprintf('Deleted stale parallel pool.\n');
end

if use_parallel
    if num_workers == 0
        num_workers = min(8, feature('numcores'));
    end
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('local', num_workers);
    elseif pool.NumWorkers ~= num_workers
        delete(pool);
        pool = parpool('local', num_workers);
    end
    fprintf('Parallel mode: %d workers\n', pool.NumWorkers);
end

scriptDir = fileparts(mfilename('fullpath'));

%% Create timestamped results directory
ts = datestr(datetime, 'yymmdd-HHMMSS');
out_dir = fullfile(scriptDir, 'results', ['hugine_' ts]);
mkdir(out_dir);

%% Start diary log
log_file = fullfile(out_dir, 'hugine_experiments.log');
diary(log_file);

%% Print header
fprintf('\n');
fprintf('================================================================\n');
fprintf('   HuGINE (Hu et al.) Full Verification Experiments\n');
fprintf('================================================================\n');
fprintf('Architecture: %s\n', arch_label);
fprintf('Tasks: %s\n', strjoin(tasks, ', '));
fprintf('Grids: %s\n', strjoin(grids, ', '));
fprintf('Node epsilon values: %s\n', mat2str(epsilon_values));
fprintf('Edge epsilon values: %s\n', mat2str(edge_epsilons));
fprintf('Test graphs: %d\n', num_graphs);
fprintf('Perturbed node features: %s (P, Q)\n', mat2str(perturb_node_features));
fprintf('Perturbed edge features: %s (impedance)\n', mat2str(perturb_edge_features));
fprintf('Voltage spec: per-grid (ieee24:[0.95,1.05], ieee39:[0.94,1.06], ieee118:[0.94,1.09])\n');
fprintf('Perturbation modes: %s\n', strjoin(perturbation_modes, ', '));
fprintf('IEEE118: subgraph verification | IEEE24/39: full-graph verification\n');
fprintf('Method: approx-star (all grids)\n');
fprintf('Results: %s\n', out_dir);
fprintf('================================================================\n\n');

total_start = tic;

%% ntfy helper
ntfy_topic = ['gnnv-' strtrim(char(java.net.InetAddress.getLocalHost.getHostName)) '-hugine'];
ntfy_url = ['https://ntfy.sh/' ntfy_topic];
notify = @(msg) try_ntfy(ntfy_url, msg);

%% Initialize results
results = struct();
results.config = struct('arch', arch, 'arch_label', arch_label, ...
    'grids', {grids}, 'tasks', {tasks}, ...
    'epsilon_values', epsilon_values, 'edge_epsilons', edge_epsilons, ...
    'v_bounds', grid_v_bounds, 'num_graphs', num_graphs, ...
    'perturb_node_features', perturb_node_features, ...
    'perturb_edge_features', perturb_edge_features, ...
    'perturbation_modes', {perturbation_modes});

%% Run all experiments
for ti = 1:length(tasks)
    task = tasks{ti};
    task_upper = upper(task);

    for mi = 1:length(perturbation_modes)
        mode = perturbation_modes{mi};
        is_edge = strcmp(mode, 'node_edge');

        % Edge perturbation is PF only
        if is_edge && ~strcmp(task, 'pf')
            fprintf('Skipping edge perturbation for %s (PF only)\n', task_upper);
            continue;
        end

        fprintf('\n########################################################\n');
        fprintf('  %s - %s\n', task_upper, strrep(mode, '_', ' '));
        fprintf('########################################################\n\n');

        for g = 1:length(grids)
            grid = grids{g};
            grid_upper = upper(grid);
            bounds = grid_v_bounds.(grid);
            v_min = bounds(1);
            v_max = bounds(2);
            use_subgraph = true;  % subgraph for all grids

            fprintf('=== %s %s [%s] (v=[%.2f,%.2f], subgraph=%d) ===\n\n', ...
                task_upper, grid_upper, mode, v_min, v_max, use_subgraph);

            % Load model
            if strcmp(task, 'pf')
                mat_file = sprintf('%s_pf_%s.mat', arch, grid);
                mat_path = fullfile(scriptDir, 'PowerFlow', grid_upper, 'models', mat_file);
            else
                mat_file = sprintf('%s_opf_%s.mat', arch, grid);
                mat_path = fullfile(scriptDir, 'OptimalPowerFlow', grid_upper, 'models', mat_file);
            end

            if ~isfile(mat_path)
                fprintf('  SKIP: %s not found\n', mat_file);
                continue;
            end

            fprintf('Loading %s from %s...\n', arch_label, mat_file);
            [gnn, test_data, norm_stats] = gnn2nnv(mat_path);

            n_graphs_avail = min(num_graphs, test_data.num_graphs);
            fprintf('  Available test graphs: %d, using: %d\n', test_data.num_graphs, n_graphs_avail);

            % Pre-filter safe graphs
            valid_graphs = filter_safe_graphs(test_data, norm_stats, v_min, v_max, n_graphs_avail);
            n_valid = length(valid_graphs);
            n_skipped = n_graphs_avail - n_valid;
            fprintf('  GT voltage filter: %d/%d graphs within [%.2f, %.2f] p.u. (%d skipped)\n', ...
                n_valid, n_graphs_avail, v_min, v_max, n_skipped);

            % Pre-compute valid graph inputs
            X_all_valid = cell(n_valid, 1);
            for vi = 1:n_valid
                X_all_valid{vi} = test_data.X_all{valid_graphs(vi)};
            end

            % Extract GNN components for parfor
            gnn_layers = gnn.Layers;
            gnn_adj_list = gnn.adj_list;
            gnn_E = gnn.E;
            gnn_edge_weights = gnn.edge_weights;
            gnn_name = gnn.Name;

            if is_edge
                % Run for each edge epsilon
                for ee = 1:length(edge_epsilons)
                    epsilon_edge = edge_epsilons(ee);

                    % Build edge perturbation
                    E_const = gnn.E;
                    range_per_edge_col = max(E_const) - min(E_const);
                    eps_matrix_edge = zeros(size(E_const));
                    for f = perturb_edge_features
                        if f <= size(E_const, 2)
                            eps_matrix_edge(:, f) = range_per_edge_col(f) * epsilon_edge;
                        end
                    end
                    E_star = GraphStar(E_const, -eps_matrix_edge, eps_matrix_edge);
                    gnn_E_edge = E_star;

                    fprintf('\n  Edge perturbation: eps_edge=%.0e, features=%s\n', ...
                        epsilon_edge, mat2str(perturb_edge_features));

                    for e = 1:length(epsilon_values)
                        epsilon = epsilon_values(e);
                        fprintf('\n--- %s %s %s, eps=%.0e, edge_eps=%.0e (%d graphs, %d safe) ---\n', ...
                            task_upper, grid_upper, arch_label, epsilon, epsilon_edge, n_graphs_avail, n_valid);

                        [graph_results] = run_epsilon_batch(X_all_valid, valid_graphs, ...
                            epsilon, perturb_node_features, use_subgraph, norm_stats, ...
                            v_min, v_max, gnn_layers, gnn_adj_list, gnn_E_edge, ...
                            gnn_edge_weights, gnn_name, use_parallel, n_graphs_avail, n_skipped);

                        % Store results
                        eps_key = make_eps_key(epsilon);
                        edge_key = make_eps_key(epsilon_edge);
                        results.(task).node_edge.(grid).(edge_key).(eps_key) = graph_results;

                        print_summary(task_upper, grid_upper, arch_label, epsilon, ...
                            epsilon_edge, graph_results, n_skipped);
                    end
                end
            else
                % Node-only
                for e = 1:length(epsilon_values)
                    epsilon = epsilon_values(e);
                    fprintf('\n--- %s %s %s, eps=%.0e (%d graphs, %d safe) ---\n', ...
                        task_upper, grid_upper, arch_label, epsilon, n_graphs_avail, n_valid);

                    [graph_results] = run_epsilon_batch(X_all_valid, valid_graphs, ...
                        epsilon, perturb_node_features, use_subgraph, norm_stats, ...
                        v_min, v_max, gnn_layers, gnn_adj_list, gnn_E, ...
                        gnn_edge_weights, gnn_name, use_parallel, n_graphs_avail, n_skipped);

                    % Store results
                    eps_key = make_eps_key(epsilon);
                    results.(task).node_only.(grid).(eps_key) = graph_results;

                    print_summary(task_upper, grid_upper, arch_label, epsilon, ...
                        [], graph_results, n_skipped);
                end
            end

            fprintf('\n');
            elapsed = toc(total_start);
            notify(sprintf('%s %s %s done (%.1f min elapsed)', task_upper, grid_upper, mode, elapsed/60));
            clear gnn test_data norm_stats;
        end

        % Save intermediate results
        results.elapsed_time = toc(total_start);
        save(fullfile(out_dir, 'results_partial.mat'), 'results');
        fprintf('Intermediate save: %.1f minutes elapsed\n', results.elapsed_time / 60);
    end
end

total_time = toc(total_start);

%% Print grand summary
print_grand_summary(results, tasks, perturbation_modes, grids, ...
    epsilon_values, edge_epsilons);

fprintf('\nTotal time: %.1f seconds (%.1f minutes)\n', total_time, total_time/60);

%% Save final results
results.total_time = total_time;
save(fullfile(out_dir, 'results.mat'), 'results');
write_csv_summaries(results, tasks, perturbation_modes, grids, ...
    epsilon_values, edge_epsilons, out_dir);

fprintf('Results saved to: %s\n', out_dir);
diary off;
end


%% =========================================================================
%  RUN ONE EPSILON BATCH (parfor or serial)
%  =========================================================================

function graph_results = run_epsilon_batch(X_all_valid, valid_graphs, ...
        epsilon, perturb_node_features, use_subgraph, norm_stats, ...
        v_min, v_max, gnn_layers, gnn_adj_list, gnn_E, ...
        gnn_edge_weights, gnn_name, use_parallel, n_total, n_skipped)

    n_valid = length(valid_graphs);
    tmp_reach_time = zeros(n_valid, 1);
    tmp_verified = zeros(n_valid, 1);
    tmp_unknown = zeros(n_valid, 1);
    tmp_violated = zeros(n_valid, 1);
    tmp_na_nodes = zeros(n_valid, 1);
    tmp_mean_width = zeros(n_valid, 1);
    tmp_max_width = zeros(n_valid, 1);

    if use_parallel
        parfor vi = 1:n_valid
            X = X_all_valid{vi};
            gnn_local = GNN(gnn_layers, [], gnn_adj_list, gnn_E, gnn_edge_weights, gnn_name);
            [t_vi, v_vi, u_vi, viol_vi, na_vi, mw_vi, xw_vi] = ...
                verify_single_graph(gnn_local, X, epsilon, perturb_node_features, ...
                    use_subgraph, norm_stats, v_min, v_max);
            tmp_reach_time(vi) = t_vi;
            tmp_verified(vi) = v_vi;
            tmp_unknown(vi) = u_vi;
            tmp_violated(vi) = viol_vi;
            tmp_na_nodes(vi) = na_vi;
            tmp_mean_width(vi) = mw_vi;
            tmp_max_width(vi) = xw_vi;
        end
    else
        for vi = 1:n_valid
            X = X_all_valid{vi};
            gi = valid_graphs(vi);
            gnn_local = GNN(gnn_layers, [], gnn_adj_list, gnn_E, gnn_edge_weights, gnn_name);
            [t_vi, v_vi, u_vi, viol_vi, na_vi, mw_vi, xw_vi] = ...
                verify_single_graph(gnn_local, X, epsilon, perturb_node_features, ...
                    use_subgraph, norm_stats, v_min, v_max);
            tmp_reach_time(vi) = t_vi;
            tmp_verified(vi) = v_vi;
            tmp_unknown(vi) = u_vi;
            tmp_violated(vi) = viol_vi;
            tmp_na_nodes(vi) = na_vi;
            tmp_mean_width(vi) = mw_vi;
            tmp_max_width(vi) = xw_vi;

            if mod(vi, 10) == 0 || vi == n_valid
                fprintf('  [%d/%d] (graph %d) time=%.3fs, verified=%d, unknown=%d, violated=%d\n', ...
                    vi, n_valid, gi, t_vi, v_vi, u_vi, viol_vi);
            end
        end
    end

    graph_results = struct();
    graph_results.reach_time = tmp_reach_time;
    graph_results.verified = tmp_verified;
    graph_results.unknown = tmp_unknown;
    graph_results.violated = tmp_violated;
    graph_results.na_nodes = tmp_na_nodes;
    graph_results.mean_width = tmp_mean_width;
    graph_results.max_width = tmp_max_width;
    graph_results.graph_indices = valid_graphs;
    graph_results.n_total = n_total;
    graph_results.n_skipped = n_skipped;
end


%% =========================================================================
%  SINGLE-GRAPH VERIFICATION (parfor-compatible)
%  =========================================================================

function [reach_time, n_verified, n_unknown, n_violated, n_na, mean_w, max_w] = ...
        verify_single_graph(gnn_local, X, epsilon, perturb_node_features, ...
            use_subgraph, norm_stats, v_min, v_max)

    % Create node perturbation in normalized space
    range_per_col = max(X) - min(X);
    eps_matrix = zeros(size(X));
    for f = perturb_node_features
        if f <= size(X, 2)
            eps_matrix(:, f) = range_per_col(f) * epsilon;
        end
    end
    GS_in = GraphStar(X, -eps_matrix, eps_matrix);

    % Always use approx-star for HuGINE
    reachOpts = struct('reachMethod', 'approx-star');

    t_start = tic;  % time everything: reach + verification

    if use_subgraph
        % Subgraph verification: per-PQ-bus k-hop subgraph
        if isfield(norm_stats, 'X_max')
            X_phys_sg = X .* norm_stats.X_max;
        else
            X_phys_sg = X;
        end
        pq_nodes = find(round(X_phys_sg(:, 4)) == 1);

        [node_outputs, sg_info] = gnn_local.reachSubgraph(GS_in, pq_nodes, reachOpts);

        voltage_idx_sg = 3;
        if isfield(norm_stats, 'Y_max')
            Y_max_v_sg = norm_stats.Y_max(voltage_idx_sg);
        else
            Y_max_v_sg = 1;
        end
        v_min_norm_sg = v_min / Y_max_v_sg;
        v_max_norm_sg = v_max / Y_max_v_sg;

        verif = -1 * ones(size(X, 1), 1);
        all_widths = [];
        for ti_sg = 1:length(pq_nodes)
            t_node = pq_nodes(ti_sg);
            t_local = sg_info(ti_sg).target_local_idx;
            gs_t = node_outputs{ti_sg};
            [v_lb, v_ub] = gs_t.getRange(t_local, voltage_idx_sg);
            if v_lb >= v_min_norm_sg && v_ub <= v_max_norm_sg
                verif(t_node) = 1;
            elseif v_ub < v_min_norm_sg || v_lb > v_max_norm_sg
                verif(t_node) = 0;
            else
                verif(t_node) = 2;
            end
            [lb_t, ub_t] = gs_t.estimateRanges();
            all_widths = [all_widths; ub_t(t_local,:)' - lb_t(t_local,:)']; %#ok<AGROW>
        end
        if ~isempty(all_widths)
            mean_w = mean(all_widths(:));
            max_w = max(all_widths(:));
        else
            mean_w = 0;
            max_w = 0;
        end
    else
        % Full-graph verification
        GS_out = gnn_local.reach(GS_in, reachOpts);

        mean_w = 0;
        max_w = 0;

        verif = verify_voltage_maxnorm(GS_out, norm_stats, X, v_min, v_max);
    end

    reach_time = toc(t_start);  % total: reach + verification

    n_verified = sum(verif == 1);
    n_unknown = sum(verif == 2);
    n_violated = sum(verif == 0);
    n_na = sum(verif == -1);
end


%% =========================================================================
%  GROUND TRUTH PRE-FILTER
%  =========================================================================

function valid = filter_safe_graphs(test_data, norm_stats, v_min, v_max, n_graphs)
    voltage_idx = 3;
    bus_type_idx = 4;
    valid = [];

    for gi = 1:n_graphs
        X = test_data.X_all{gi};
        Y = test_data.Y_all{gi};

        if isfield(norm_stats, 'X_max')
            X_phys = X .* norm_stats.X_max;
        else
            X_phys = X;
        end
        if isfield(norm_stats, 'Y_max')
            Y_max_v = norm_stats.Y_max(voltage_idx);
        else
            Y_max_v = 1;
        end

        voltage_mask = (round(X_phys(:, bus_type_idx)) == 1);
        gt_voltages = Y(voltage_mask, voltage_idx) * Y_max_v;

        if all(gt_voltages >= v_min) && all(gt_voltages <= v_max)
            valid(end+1) = gi; %#ok<AGROW>
        end
    end
end


%% =========================================================================
%  VOLTAGE VERIFICATION (full-graph)
%  =========================================================================

function res = verify_voltage_maxnorm(GS_out, norm_stats, X, v_min, v_max)
    voltage_idx = 3;
    bus_type_idx = 4;
    numNodes = size(GS_out.V, 1);
    numFeatures = size(GS_out.V, 2);
    res = zeros(numNodes, 1);

    if isfield(norm_stats, 'Y_max')
        v_min_norm = v_min / norm_stats.Y_max(voltage_idx);
        v_max_norm = v_max / norm_stats.Y_max(voltage_idx);
    else
        v_min_norm = v_min;
        v_max_norm = v_max;
    end

    if isfield(norm_stats, 'X_max')
        X_physical = X .* norm_stats.X_max;
    else
        X_physical = X;
    end
    voltage_mask = (round(X_physical(:, bus_type_idx)) == 1);

    Y_star = GS_out.toStar();

    for i = 1:numNodes
        if ~voltage_mask(i)
            res(i) = -1;
            continue;
        end

        matIdx = zeros(1, numNodes * numFeatures);
        flat_idx = (voltage_idx - 1) * numNodes + i;
        matIdx(flat_idx) = 1;
        Y_node = Y_star.affineMap(matIdx, []);

        G = [1; -1];
        g_vec = [v_max_norm; -v_min_norm];
        Hs = [HalfSpace(G(1,:), g_vec(1)); HalfSpace(G(2,:), g_vec(2))];
        r = verify_specification(Y_node, Hs);

        if r == 2
            [lb, ub] = Y_node.getRanges;
            if lb(1) >= v_min_norm && ub(1) <= v_max_norm
                r = 1;
            elseif ub(1) < v_min_norm || lb(1) > v_max_norm
                r = 0;
            end
        end

        res(i) = r;
    end
end


%% =========================================================================
%  SUMMARY HELPERS
%  =========================================================================

function print_summary(task_upper, grid_upper, arch_label, epsilon, epsilon_edge, r, n_skipped)
    total_v = sum(r.verified);
    total_u = sum(r.unknown);
    total_viol = sum(r.violated);
    total_nodes = total_v + total_u + total_viol;
    avg_time = mean(r.reach_time);

    if isempty(epsilon_edge)
        fprintf('  Summary: verified=%d/%d (%.1f%%), unknown=%d, violated=%d, avg_time=%.3fs\n', ...
            total_v, total_nodes, 100*total_v/max(1,total_nodes), total_u, total_viol, avg_time);
    else
        fprintf('  Summary: verified=%d/%d (%.1f%%), unknown=%d, violated=%d, avg_time=%.3fs [edge_eps=%.0e]\n', ...
            total_v, total_nodes, 100*total_v/max(1,total_nodes), total_u, total_viol, avg_time, epsilon_edge);
    end
    if n_skipped > 0
        fprintf('  (%d graphs skipped -- GT voltages outside spec)\n', n_skipped);
    end
end


function key = make_eps_key(epsilon)
    key = ['eps_' strrep(sprintf('%.0e', epsilon), '-', 'n')];
    key = strrep(key, '+', '');
    key = strrep(key, '.', '_');
end


%% =========================================================================
%  GRAND SUMMARY
%  =========================================================================

function print_grand_summary(results, tasks, modes, grids, epsilon_values, edge_epsilons)
    fprintf('\n================================================================\n');
    fprintf('  GRAND SUMMARY — HuGINE (Hu et al.)\n');
    fprintf('================================================================\n');

    for ti = 1:length(tasks)
        task = tasks{ti};
        if ~isfield(results, task), continue; end

        for mi = 1:length(modes)
            mode = modes{mi};
            if ~isfield(results.(task), mode), continue; end
            is_edge = strcmp(mode, 'node_edge');

            fprintf('\n--- %s / %s ---\n', upper(task), strrep(mode, '_', ' '));

            for g = 1:length(grids)
                grid = grids{g};
                if ~isfield(results.(task).(mode), grid), continue; end

                if is_edge
                    for ee = 1:length(edge_epsilons)
                        edge_key = make_eps_key(edge_epsilons(ee));
                        if ~isfield(results.(task).(mode).(grid), edge_key), continue; end

                        fprintf('\n  %s (edge_eps=%.0e):\n', upper(grid), edge_epsilons(ee));
                        print_summary_table(results.(task).(mode).(grid).(edge_key), epsilon_values);
                    end
                else
                    fprintf('\n  %s:\n', upper(grid));
                    print_summary_table(results.(task).(mode).(grid), epsilon_values);
                end
            end
        end
    end
    fprintf('\n================================================================\n');
end


function print_summary_table(grid_results, epsilon_values)
    % Timing row
    fprintf('  %-10s |', 'Time');
    for e = 1:length(epsilon_values)
        eps_key = make_eps_key(epsilon_values(e));
        if isfield(grid_results, eps_key)
            r = grid_results.(eps_key);
            fprintf(' %8.3fs |', mean(r.reach_time));
        else
            fprintf('      N/A  |');
        end
    end
    fprintf('\n');

    % Verified row
    fprintf('  %-10s |', 'Verified');
    for e = 1:length(epsilon_values)
        eps_key = make_eps_key(epsilon_values(e));
        if isfield(grid_results, eps_key)
            r = grid_results.(eps_key);
            tv = sum(r.verified);
            total = tv + sum(r.unknown) + sum(r.violated);
            fprintf(' %4d/%-4d  |', tv, total);
        else
            fprintf('      N/A  |');
        end
    end
    fprintf('\n');

    % Pct row
    fprintf('  %-10s |', 'Pct');
    for e = 1:length(epsilon_values)
        eps_key = make_eps_key(epsilon_values(e));
        if isfield(grid_results, eps_key)
            r = grid_results.(eps_key);
            tv = sum(r.verified);
            total = tv + sum(r.unknown) + sum(r.violated);
            fprintf(' %7.1f%%  |', 100*tv/max(1,total));
        else
            fprintf('      N/A  |');
        end
    end
    fprintf('\n');
end


%% =========================================================================
%  CSV OUTPUT
%  =========================================================================

function write_csv_summaries(results, tasks, modes, grids, epsilon_values, edge_epsilons, out_dir)
    csv_file = fullfile(out_dir, 'hugine_results.csv');
    fid = fopen(csv_file, 'w');
    fprintf(fid, 'Task,Mode,Grid,Node_Epsilon,Edge_Epsilon,Total_Graphs,Safe_Graphs,Skipped_Graphs,Avg_Time_s,Total_Verified,Total_Unknown,Total_Violated,Total_Voltage_Nodes,Pct_Verified,Mean_Bound_Width,Max_Bound_Width\n');

    for ti = 1:length(tasks)
        task = tasks{ti};
        if ~isfield(results, task), continue; end

        for mi = 1:length(modes)
            mode = modes{mi};
            if ~isfield(results.(task), mode), continue; end
            is_edge = strcmp(mode, 'node_edge');

            for g = 1:length(grids)
                grid = grids{g};
                if ~isfield(results.(task).(mode), grid), continue; end

                if is_edge
                    for ee = 1:length(edge_epsilons)
                        edge_key = make_eps_key(edge_epsilons(ee));
                        if ~isfield(results.(task).(mode).(grid), edge_key), continue; end

                        for e = 1:length(epsilon_values)
                            eps_key = make_eps_key(epsilon_values(e));
                            if ~isfield(results.(task).(mode).(grid).(edge_key), eps_key), continue; end
                            r = results.(task).(mode).(grid).(edge_key).(eps_key);
                            write_csv_row(fid, task, mode, grid, epsilon_values(e), edge_epsilons(ee), r);
                        end
                    end
                else
                    for e = 1:length(epsilon_values)
                        eps_key = make_eps_key(epsilon_values(e));
                        if ~isfield(results.(task).(mode).(grid), eps_key), continue; end
                        r = results.(task).(mode).(grid).(eps_key);
                        write_csv_row(fid, task, mode, grid, epsilon_values(e), NaN, r);
                    end
                end
            end
        end
    end

    fclose(fid);
    fprintf('Saved: %s\n', csv_file);
end


function write_csv_row(fid, task, mode, grid, node_eps, edge_eps, r)
    tv = sum(r.verified);
    tu = sum(r.unknown);
    tviol = sum(r.violated);
    total = tv + tu + tviol;
    n_safe = length(r.reach_time);

    if isnan(edge_eps)
        edge_eps_str = 'N/A';
    else
        edge_eps_str = sprintf('%.0e', edge_eps);
    end

    fprintf(fid, '%s,%s,%s,%.0e,%s,%d,%d,%d,%.4f,%d,%d,%d,%d,%.2f,%.6f,%.6f\n', ...
        upper(task), mode, grid, node_eps, edge_eps_str, ...
        r.n_total, n_safe, r.n_skipped, mean(r.reach_time), ...
        tv, tu, tviol, total, 100*tv/max(1,total), ...
        mean(r.mean_width), max(r.max_width));
end


function try_ntfy(url, msg)
    try
        webwrite(url, msg);
    catch
        % ignore ntfy errors
    end
end
