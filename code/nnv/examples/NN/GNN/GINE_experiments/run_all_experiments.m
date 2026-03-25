function results = run_all_experiments(varargin)
% run_all_experiments - Full PF + OPF verification with edge perturbation
%
% Runs all 4 architectures x 3 grids x 3 epsilons x 100 graphs,
% plus edge perturbation trials for GINE architectures.
%
% Usage:
%   run_all_experiments()
%   run_all_experiments('num_graphs', 2)          % smoke test
%   run_all_experiments('grid', 'ieee24')         % single grid
%   run_all_experiments('task', 'pf')             % PF only
%   run_all_experiments('skip_edge', true)        % skip edge perturbation
%   run_all_experiments('parallel', true)         % parfor over graphs (default: false)

%% Parse arguments
p = inputParser;
addParameter(p, 'num_graphs', 100, @isnumeric);
addParameter(p, 'grid', 'all', @ischar);
addParameter(p, 'task', 'all', @ischar);        % 'pf', 'opf', or 'all'
addParameter(p, 'skip_edge', false, @islogical);
addParameter(p, 'arch', 'all', @ischar);        % single arch or 'all'
addParameter(p, 'subgraph', false, @islogical); % use k-hop subgraph verification for GINE archs
addParameter(p, 'mode', 'all', @ischar);        % 'all', 'node_only', or 'node_edge'
addParameter(p, 'parallel', false, @islogical); % parfor over graphs (per-graph timing preserved)
addParameter(p, 'num_workers', 0, @isnumeric);  % 0 = auto (maxNumCompThreads)
addParameter(p, 'epsilon_edge', 0.01, @isnumeric); % edge perturbation magnitude
addParameter(p, 'epsilons', [], @isnumeric);       % custom node epsilon values (default: [0.001, 0.005, 0.01])
parse(p, varargin{:});

%% Configuration
all_architectures = {'gcn', 'sage', 'gine_linear', 'gine_conv'};
all_arch_labels = {'GCN', 'SAGEConv', 'GINE Linear', 'GINE Conv'};
if strcmp(p.Results.arch, 'all')
    architectures = all_architectures;
    arch_labels = all_arch_labels;
else
    idx = find(strcmp(all_architectures, p.Results.arch));
    if isempty(idx)
        error('Unknown arch: %s. Must be one of: %s', p.Results.arch, strjoin(all_architectures, ', '));
    end
    architectures = all_architectures(idx);
    arch_labels = all_arch_labels(idx);
end
edge_archs = {'gine_linear', 'gine_conv'};  % Only these support edge perturbation

if strcmp(p.Results.grid, 'all')
    grids = {'ieee24', 'ieee39', 'ieee118'};
else
    grids = {p.Results.grid};
end

if isempty(p.Results.epsilons)
    epsilon_values = [0.001, 0.005, 0.01];
else
    epsilon_values = p.Results.epsilons;
end
epsilon_edge = p.Results.epsilon_edge;  % Edge perturbation magnitude
num_graphs = p.Results.num_graphs;
perturb_node_features = [1, 2];  % Power injections
perturb_edge_features = [1];     % Impedance
% Per-grid voltage bounds from actual network specifications.
% IEEE24: vmax=1.05, IEEE39: vmax=1.06, IEEE118: vmax=1.09
% (confirmed from raw simulation data; using 0.94 as common lower bound for 39/118)
grid_v_bounds = struct( ...
    'ieee24', [0.95, 1.05], ...
    'ieee39', [0.94, 1.06], ...
    'ieee118', [0.94, 1.09]);
% v_min/v_max are set per-grid inside the grid loop from grid_v_bounds

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
    otherwise  % 'all'
        perturbation_modes = {'node_only'};
        if ~p.Results.skip_edge
            perturbation_modes{end+1} = 'node_edge';
        end
end

use_subgraph = p.Results.subgraph;  % k-hop subgraph verification for all architectures
if use_subgraph
    fprintf('Subgraph mode: enabled (k-hop locality, all architectures)\n');
end

use_parallel = p.Results.parallel;
num_workers = p.Results.num_workers;
if use_parallel
    if num_workers == 0
        num_workers = min(8, feature('numcores'));  % limited by local cluster NumWorkers
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
opfDir = fullfile(scriptDir, 'OptimalPowerFlow');

%% Create timestamped results directory
ts = datestr(datetime, 'yymmdd-HHMMSS');
out_dir = fullfile(scriptDir, 'results', ts);
mkdir(out_dir);

%% Start diary log
log_file = fullfile(out_dir, 'all_experiments.log');
diary(log_file);

%% Print header
fprintf('\n');
fprintf('================================================================\n');
fprintf('   Full PF + OPF Verification Experiments\n');
fprintf('================================================================\n');
fprintf('Tasks: %s\n', strjoin(tasks, ', '));
fprintf('Architectures: %s\n', strjoin(arch_labels, ', '));
fprintf('Edge perturbation archs: %s\n', strjoin(edge_archs, ', '));
fprintf('Grids: %s\n', strjoin(grids, ', '));
fprintf('Node epsilon values: %s\n', mat2str(epsilon_values));
fprintf('Edge epsilon: %.3f (fixed)\n', epsilon_edge);
fprintf('Test graphs: %d\n', num_graphs);
fprintf('Perturbed node features: %s (power injections)\n', mat2str(perturb_node_features));
fprintf('Perturbed edge features: %s (impedance)\n', mat2str(perturb_edge_features));
fprintf('Voltage spec: per-grid (ieee24:[0.95,1.05], ieee39:[0.94,1.06], ieee118:[0.94,1.09])\n');
fprintf('Perturbation modes: %s\n', strjoin(perturbation_modes, ', '));
fprintf('Results: %s\n', out_dir);
fprintf('================================================================\n\n');

total_start = tic;

%% Initialize results
results = struct();
results.config = struct('architectures', {architectures}, ...
    'arch_labels', {arch_labels}, 'grids', {grids}, ...
    'epsilon_values', epsilon_values, 'epsilon_edge', epsilon_edge, ...
    'v_bounds', grid_v_bounds, ...
    'num_graphs', num_graphs, 'perturb_node_features', perturb_node_features, ...
    'perturb_edge_features', perturb_edge_features, 'tasks', {tasks}, ...
    'perturbation_modes', {perturbation_modes});

%% Run all experiments
for ti = 1:length(tasks)
    task = tasks{ti};
    task_upper = upper(task);

    for mi = 1:length(perturbation_modes)
        mode = perturbation_modes{mi};
        is_edge = strcmp(mode, 'node_edge');

        % Select architectures for this mode
        if is_edge
            run_archs = intersect(edge_archs, architectures, 'stable');
            run_labels = {};
            for a = 1:length(run_archs)
                idx = find(strcmp(all_architectures, run_archs{a}));
                run_labels{a} = all_arch_labels{idx}; %#ok<AGROW>
            end
        else
            run_archs = architectures;
            run_labels = arch_labels;
        end

        fprintf('\n########################################################\n');
        fprintf('  %s — %s\n', task_upper, strrep(mode, '_', ' '));
        if is_edge
            fprintf('  (edge eps=%.3f fixed, archs: %s)\n', epsilon_edge, strjoin(run_labels, ', '));
        end
        fprintf('########################################################\n\n');

        for g = 1:length(grids)
            grid = grids{g};
            grid_upper = upper(grid);
            % Set per-grid voltage bounds
            bounds = grid_v_bounds.(grid);
            v_min = bounds(1);
            v_max = bounds(2);
            fprintf('=== %s %s [%s] (v=[%.2f,%.2f]) ===\n\n', task_upper, grid_upper, mode, v_min, v_max);

            for a = 1:length(run_archs)
                arch = run_archs{a};
                label = run_labels{a};

                % Determine model path
                if strcmp(task, 'pf')
                    mat_file = sprintf('%s_pf_%s.mat', arch, grid);
                    mat_path = fullfile(scriptDir, 'PowerFlow', grid_upper, 'models', mat_file);
                else
                    mat_file = sprintf('%s_opf_%s.mat', arch, grid);
                    mat_path = fullfile(opfDir, grid_upper, 'models', mat_file);
                end

                if ~isfile(mat_path)
                    fprintf('  SKIP: %s not found\n', mat_file);
                    continue;
                end

                fprintf('Loading %s from %s...\n', label, mat_file);
                [gnn, test_data, norm_stats] = gnn2nnv(mat_path);

                % For edge perturbation, rebuild GNN with E_star
                if is_edge
                    E_const = gnn.E;
                    range_per_edge_col = max(E_const) - min(E_const);
                    eps_matrix_edge = zeros(size(E_const));
                    for f = perturb_edge_features
                        if f <= size(E_const, 2)
                            eps_matrix_edge(:, f) = range_per_edge_col(f) * epsilon_edge;
                        end
                    end
                    E_star = GraphStar(E_const, -eps_matrix_edge, eps_matrix_edge);
                    gnn = GNN(gnn.Layers, [], gnn.adj_list, E_star, gnn.edge_weights);
                    fprintf('  Edge perturbation: eps=%.3f, features=%s\n', ...
                        epsilon_edge, mat2str(perturb_edge_features));
                end

                n_graphs = min(num_graphs, test_data.num_graphs);
                fprintf('  Available test graphs: %d, using: %d\n', test_data.num_graphs, n_graphs);

                % Pre-filter: only verify graphs where GT voltages are within spec
                valid_graphs = filter_safe_graphs(test_data, norm_stats, v_min, v_max, n_graphs);
                n_valid = length(valid_graphs);
                n_skipped = n_graphs - n_valid;
                fprintf('  GT voltage filter: %d/%d graphs within [%.2f, %.2f] p.u. (%d skipped)\n', ...
                    n_valid, n_graphs, v_min, v_max, n_skipped);

                % Extract value-type data from handle for parfor compatibility
                gnn_layers = gnn.Layers;
                gnn_A_norm = gnn.A_norm;
                gnn_adj_list = gnn.adj_list;
                gnn_E = gnn.E;
                gnn_edge_weights = gnn.edge_weights;
                gnn_name = gnn.Name;

                for e = 1:length(epsilon_values)
                    epsilon = epsilon_values(e);
                    fprintf('\n--- %s %s %s, eps=%.3f (%d graphs, %d safe) ---\n', ...
                        task_upper, grid_upper, label, epsilon, n_graphs, n_valid);

                    % Pre-compute per-graph inputs (avoid test_data broadcast)
                    X_all_valid = cell(n_valid, 1);
                    for vi = 1:n_valid
                        X_all_valid{vi} = test_data.X_all{valid_graphs(vi)};
                    end

                    % Temporary arrays for parfor
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

                            % Reconstruct GNN per worker (avoids handle sharing)
                            gnn_local = GNN(gnn_layers, gnn_A_norm, gnn_adj_list, gnn_E, gnn_edge_weights, gnn_name);

                            [t_vi, v_vi, u_vi, viol_vi, na_vi, mw_vi, xw_vi] = ...
                                verify_single_graph(gnn_local, X, epsilon, perturb_node_features, ...
                                    use_subgraph, norm_stats, v_min, v_max, is_edge, grid, arch);
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

                            % Reconstruct GNN per iteration for consistency
                            gnn_local = GNN(gnn_layers, gnn_A_norm, gnn_adj_list, gnn_E, gnn_edge_weights, gnn_name);

                            [t_vi, v_vi, u_vi, viol_vi, na_vi, mw_vi, xw_vi] = ...
                                verify_single_graph(gnn_local, X, epsilon, perturb_node_features, ...
                                    use_subgraph, norm_stats, v_min, v_max, is_edge, grid, arch);
                            tmp_reach_time(vi) = t_vi;
                            tmp_verified(vi) = v_vi;
                            tmp_unknown(vi) = u_vi;
                            tmp_violated(vi) = viol_vi;
                            tmp_na_nodes(vi) = na_vi;
                            tmp_mean_width(vi) = mw_vi;
                            tmp_max_width(vi) = xw_vi;

                            % Progress (serial only — parfor output is non-deterministic)
                            if mod(vi, 10) == 0 || vi == n_valid
                                fprintf('  [%d/%d] (graph %d) time=%.3fs, verified=%d, unknown=%d, violated=%d\n', ...
                                    vi, n_valid, gi, t_vi, v_vi, u_vi, viol_vi);
                            end
                        end
                    end

                    % Assemble results
                    graph_results = struct();
                    graph_results.reach_time = tmp_reach_time;
                    graph_results.verified = tmp_verified;
                    graph_results.unknown = tmp_unknown;
                    graph_results.violated = tmp_violated;
                    graph_results.na_nodes = tmp_na_nodes;
                    graph_results.mean_width = tmp_mean_width;
                    graph_results.max_width = tmp_max_width;
                    graph_results.graph_indices = valid_graphs;
                    graph_results.n_total = n_graphs;
                    graph_results.n_skipped = n_skipped;

                    % Store results
                    eps_key = ['eps_' strrep(sprintf('%.3f', epsilon), '.', '_')];
                    results.(task).(mode).(grid).(arch).(eps_key) = graph_results;

                    % Summary
                    total_v = sum(graph_results.verified);
                    total_u = sum(graph_results.unknown);
                    total_viol = sum(graph_results.violated);
                    total_nodes = total_v + total_u + total_viol;
                    avg_time = mean(graph_results.reach_time);
                    fprintf('  Summary: verified=%d/%d (%.1f%%), unknown=%d, violated=%d, avg_time=%.3fs\n', ...
                        total_v, total_nodes, 100*total_v/max(1,total_nodes), total_u, total_viol, avg_time);
                    if n_skipped > 0
                        fprintf('  (%d graphs skipped — GT voltages outside spec)\n', n_skipped);
                    end
                end
                fprintf('\n');

                % Clear to free memory
                clear gnn test_data norm_stats;
            end
        end

        % Save intermediate results after each task/mode block
        results.elapsed_time = toc(total_start);
        save(fullfile(out_dir, 'results_partial.mat'), 'results');
        fprintf('Intermediate save: %.1f minutes elapsed\n', results.elapsed_time / 60);
    end
end

total_time = toc(total_start);

%% Print grand summary
print_grand_summary(results, tasks, perturbation_modes, grids, architectures, ...
    arch_labels, edge_archs, epsilon_values, epsilon_edge, num_graphs);

fprintf('\nTotal time: %.1f seconds (%.1f minutes)\n', total_time, total_time/60);

%% Save final results
results.total_time = total_time;
save(fullfile(out_dir, 'results.mat'), 'results');
write_csv_summaries(results, tasks, perturbation_modes, grids, architectures, ...
    arch_labels, edge_archs, epsilon_values, epsilon_edge, out_dir);

fprintf('Results saved to: %s\n', out_dir);
diary off;
end


%% =========================================================================
%  SINGLE-GRAPH VERIFICATION (parfor-compatible)
%  =========================================================================

function [reach_time, n_verified, n_unknown, n_violated, n_na, mean_w, max_w] = ...
        verify_single_graph(gnn_local, X, epsilon, perturb_node_features, ...
            use_subgraph, norm_stats, v_min, v_max, is_edge, grid, arch) %#ok<INUSD>
    % Verify a single graph — extracted for parfor compatibility.
    % Each worker gets its own GNN instance (no handle sharing).

    % Create node perturbation
    range_per_col = max(X) - min(X);
    eps_matrix = zeros(size(X));
    for f = perturb_node_features
        if f <= size(X, 2)
            eps_matrix(:, f) = range_per_col(f) * epsilon;
        end
    end

    GS_in = GraphStar(X, -eps_matrix, eps_matrix);

    % Use relax-star for IEEE-118 GINE Conv to avoid intractable LP
    % accumulation on large subgraphs (~49 nodes) with near-zero activations.
    % All other grid/arch combinations use approx-star (full LP).
    if strcmp(grid, 'ieee118') && strcmp(arch, 'gine_conv')
        reachOpts = struct('reachMethod', 'relax-star-range', 'relaxFactor', 0.95);
    else
        reachOpts = struct('reachMethod', 'approx-star');
    end

    if use_subgraph
        % Subgraph verification: per-PQ-bus k-hop subgraph
        if isfield(norm_stats, 'X_max')
            X_phys_sg = X .* norm_stats.X_max;
        else
            X_phys_sg = X;
        end
        pq_nodes = find(round(X_phys_sg(:, 4)) == 1);  % bus_type==1

        t_start = tic;
        [node_outputs, sg_info] = gnn_local.reachSubgraph(GS_in, pq_nodes, reachOpts);
        reach_time = toc(t_start);

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
        % Standard full-graph verification
        t_start = tic;
        GS_out = gnn_local.reach(GS_in, reachOpts);
        reach_time = toc(t_start);

        [lb_out, ub_out] = GS_out.getRanges();
        bound_widths = ub_out - lb_out;
        mean_w = mean(bound_widths(:));
        max_w = max(bound_widths(:));

        verif = verify_voltage_maxnorm(GS_out, norm_stats, X, v_min, v_max);
    end

    n_verified = sum(verif == 1);
    n_unknown = sum(verif == 2);
    n_violated = sum(verif == 0);
    n_na = sum(verif == -1);
end


%% =========================================================================
%  GROUND TRUTH PRE-FILTER
%  =========================================================================

function valid = filter_safe_graphs(test_data, norm_stats, v_min, v_max, n_graphs)
% Filter graphs where all PQ bus GT voltages are within [v_min, v_max].
% Returns indices of graphs that pass the filter.
    voltage_idx = 3;
    bus_type_idx = 4;
    valid = [];

    for gi = 1:n_graphs
        X = test_data.X_all{gi};
        Y = test_data.Y_all{gi};

        % Recover physical values
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

        % PQ bus mask (bus_type == 1)
        voltage_mask = (round(X_phys(:, bus_type_idx)) == 1);
        gt_voltages = Y(voltage_mask, voltage_idx) * Y_max_v;

        if all(gt_voltages >= v_min) && all(gt_voltages <= v_max)
            valid(end+1) = gi; %#ok<AGROW>
        end
    end
end


%% =========================================================================
%  VOLTAGE VERIFICATION
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
%  SUMMARY TABLES
%  =========================================================================

function print_grand_summary(results, tasks, modes, grids, architectures, ...
        arch_labels, edge_archs, epsilon_values, epsilon_edge, ~)

    fprintf('\n================================================================\n');
    fprintf('  GRAND SUMMARY (GT-filtered: only safe operating conditions)\n');
    fprintf('================================================================\n');

    for ti = 1:length(tasks)
        task = tasks{ti};
        if ~isfield(results, task), continue; end

        for mi = 1:length(modes)
            mode = modes{mi};
            if ~isfield(results.(task), mode), continue; end
            is_edge = strcmp(mode, 'node_edge');

            if is_edge
                run_archs = edge_archs;
            else
                run_archs = architectures;
            end

            fprintf('\n--- %s / %s ---\n', upper(task), strrep(mode, '_', ' '));
            if is_edge
                fprintf('  (edge eps=%.3f fixed)\n', epsilon_edge);
            end

            for g = 1:length(grids)
                grid = grids{g};
                if ~isfield(results.(task).(mode), grid), continue; end

                % Show how many graphs passed GT filter (from first arch/eps)
                first_arch = '';
                for a2 = 1:length(run_archs)
                    if isfield(results.(task).(mode).(grid), run_archs{a2})
                        first_arch = run_archs{a2};
                        break;
                    end
                end
                if ~isempty(first_arch)
                    eps_key1 = ['eps_' strrep(sprintf('%.3f', epsilon_values(1)), '.', '_')];
                    r1 = results.(task).(mode).(grid).(first_arch).(eps_key1);
                    fprintf('\n  %s: (%d/%d graphs with GT within spec)\n', ...
                        upper(grid), length(r1.reach_time), r1.n_total);
                else
                    fprintf('\n  %s:\n', upper(grid));
                end

                % Time table
                fprintf('  %-14s |', 'Architecture');
                for e = 1:length(epsilon_values)
                    fprintf(' eps=%.3f |', epsilon_values(e));
                end
                fprintf('\n');
                fprintf('  %s\n', repmat('-', 1, 15 + 11*length(epsilon_values)));

                for a = 1:length(run_archs)
                    arch = run_archs{a};
                    if ~isfield(results.(task).(mode).(grid), arch), continue; end
                    idx = find(strcmp(architectures, arch));
                    fprintf('  %-14s |', arch_labels{idx});
                    for e = 1:length(epsilon_values)
                        eps_key = ['eps_' strrep(sprintf('%.3f', epsilon_values(e)), '.', '_')];
                        if isfield(results.(task).(mode).(grid).(arch), eps_key)
                            r = results.(task).(mode).(grid).(arch).(eps_key);
                            fprintf(' %7.3fs |', mean(r.reach_time));
                        else
                            fprintf('     N/A  |');
                        end
                    end
                    fprintf('\n');
                end

                % Robustness table
                fprintf('\n  %-14s |', 'Verified');
                for e = 1:length(epsilon_values)
                    fprintf('   eps=%.3f   |', epsilon_values(e));
                end
                fprintf('\n');
                fprintf('  %s\n', repmat('-', 1, 15 + 15*length(epsilon_values)));

                for a = 1:length(run_archs)
                    arch = run_archs{a};
                    if ~isfield(results.(task).(mode).(grid), arch), continue; end
                    idx = find(strcmp(architectures, arch));
                    fprintf('  %-14s |', arch_labels{idx});
                    for e = 1:length(epsilon_values)
                        eps_key = ['eps_' strrep(sprintf('%.3f', epsilon_values(e)), '.', '_')];
                        if isfield(results.(task).(mode).(grid).(arch), eps_key)
                            r = results.(task).(mode).(grid).(arch).(eps_key);
                            tv = sum(r.verified);
                            tu = sum(r.unknown);
                            tviol = sum(r.violated);
                            total = tv + tu + tviol;
                            fprintf(' %4d/%-4d %4.0f%% |', tv, total, 100*tv/max(1,total));
                        else
                            fprintf('     N/A       |');
                        end
                    end
                    fprintf('\n');
                end
            end
        end
    end
    fprintf('\n================================================================\n');
end


%% =========================================================================
%  CSV OUTPUT
%  =========================================================================

function write_csv_summaries(results, tasks, modes, grids, architectures, ...
        arch_labels, edge_archs, epsilon_values, epsilon_edge, out_dir)

    csv_file = fullfile(out_dir, 'all_results.csv');
    fid = fopen(csv_file, 'w');
    fprintf(fid, 'Task,Perturbation_Mode,Grid,Architecture,Node_Epsilon,Edge_Epsilon,Total_Graphs,Safe_Graphs,Skipped_Graphs,Avg_Time_s,Total_Verified,Total_Unknown,Total_Violated,Total_Voltage_Nodes,Pct_Verified,Mean_Bound_Width,Max_Bound_Width\n');

    for ti = 1:length(tasks)
        task = tasks{ti};
        if ~isfield(results, task), continue; end

        for mi = 1:length(modes)
            mode = modes{mi};
            if ~isfield(results.(task), mode), continue; end
            is_edge = strcmp(mode, 'node_edge');

            if is_edge
                run_archs = edge_archs;
                edge_eps_str = sprintf('%.3f', epsilon_edge);
            else
                run_archs = architectures;
                edge_eps_str = 'N/A';
            end

            for g = 1:length(grids)
                grid = grids{g};
                if ~isfield(results.(task).(mode), grid), continue; end

                for a = 1:length(run_archs)
                    arch = run_archs{a};
                    if ~isfield(results.(task).(mode).(grid), arch), continue; end
                    idx = find(strcmp(architectures, arch));

                    for e = 1:length(epsilon_values)
                        eps_key = ['eps_' strrep(sprintf('%.3f', epsilon_values(e)), '.', '_')];
                        if ~isfield(results.(task).(mode).(grid).(arch), eps_key), continue; end

                        r = results.(task).(mode).(grid).(arch).(eps_key);
                        tv = sum(r.verified);
                        tu = sum(r.unknown);
                        tviol = sum(r.violated);
                        total = tv + tu + tviol;

                        n_total = r.n_total;
                        n_skipped = r.n_skipped;
                        n_safe = length(r.reach_time);

                        fprintf(fid, '%s,%s,%s,%s,%.3f,%s,%d,%d,%d,%.4f,%d,%d,%d,%d,%.2f,%.6f,%.6f\n', ...
                            upper(task), mode, grid, arch_labels{idx}, ...
                            epsilon_values(e), edge_eps_str, n_total, n_safe, n_skipped, ...
                            mean(r.reach_time), tv, tu, tviol, total, ...
                            100*tv/max(1,total), mean(r.mean_width), max(r.max_width));
                    end
                end
            end
        end
    end

    fclose(fid);
    fprintf('Saved: %s\n', csv_file);
end
