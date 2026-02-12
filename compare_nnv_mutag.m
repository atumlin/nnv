%% NNV vs SCIP-MPNN Comparison: MUTAG Feature Perturbation
% Runs NNV reachability on the same MUTAG instances used by SCIP-MPNN.
% Outputs results to CSV for comparison.
%
% Configure reachability method below:
%   'approx-star'  — default, interval-arithmetic Phase 1 (baseline)
%   'abs-dom'      — LP-based bounds via getRanges (more precise)
%   'relax-star-range' with relaxFactor=0 — LP for all unstable neurons

%% Configuration
reach_method = 'approx-star';  % Change this to test different methods
relax_factor = 0;              % Only used for relax-star-range
method_tag = reach_method;     % Used in output filenames

%% Setup
scip_base = '/home/verivital/Anne/dev/SCIP-MPNN-v1.0/christopherhojny-SCIP-MPNN-0b8d73c';
gnn_path = fullfile(scip_base, 'data_experiments', 'gnn_instances', 'model_MUTAG.gnn');
gcinfo_dir = fullfile(scip_base, 'data_experiments', 'graph_classification_instances');
results_dir = fullfile(scip_base, 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

% Load the MUTAG model once
[gnn, ~] = gnn_from_scipmpnn(gnn_path);
fprintf('Loaded MUTAG model: %d layers\n', gnn.numLayers);

% Set reachability options
reachOpts = struct();
reachOpts.reachMethod = reach_method;
reachOpts.relaxFactor = relax_factor;

fprintf('Reachability method: %s (relaxFactor=%.1f)\n', reach_method, relax_factor);

% Find all MUTAG gcinfo files
gcinfo_files = dir(fullfile(gcinfo_dir, 'graph_MUTAG_*.gcinfo'));
n_instances = length(gcinfo_files);
fprintf('Found %d MUTAG graphs\n', n_instances);

% Pre-load all graph data
fprintf('Pre-loading graph data...\n');
graphs = cell(n_instances, 1);
for gi = 1:n_instances
    gcinfo_path = fullfile(gcinfo_files(gi).folder, gcinfo_files(gi).name);
    tokens = regexp(gcinfo_files(gi).name, 'graph_MUTAG_(\d+)\.gcinfo', 'tokens');
    gd = read_gcinfo_file(gcinfo_path);
    gd.graph_id = str2double(tokens{1}{1});
    graphs{gi} = gd;
end
fprintf('Loaded %d graphs\n', n_instances);

% Epsilon values to test
epsilons = [0.01, 0.05, 0.1, 0.2, 0.5];

%% Run experiments
for ei = 1:length(epsilons)
    eps_val = epsilons(ei);
    fprintf('\n========== Epsilon = %.2f ==========\n', eps_val);

    csv_path = fullfile(results_dir, sprintf('nnv_mutag_%s_eps%.2f.csv', method_tag, eps_val));
    fid = fopen(csv_path, 'w');
    fprintf(fid, 'graph_id,n_nodes,n_features,true_label,predicted_class,result,reach_time,bound_gap\n');

    n_verified = 0;
    n_violated = 0;
    n_unknown = 0;
    n_error = 0;
    total_time = 0;

    for gi = 1:n_instances
        gd = graphs{gi};
        graph_id = gd.graph_id;

        % Set graph adjacency (reuse layer weights)
        gnn.A_norm = gd.A;

        % Evaluate (get predicted class)
        Y = gnn.evaluate(gd.X);
        [~, pred_class] = max(Y);
        pred_class = pred_class - 1;  % 0-indexed

        % Create perturbed bounds
        X_lb = max(gd.X - eps_val, 0);
        X_ub = min(gd.X + eps_val, 1);
        gs_input = GraphStar(X_lb, X_ub);

        % Reachability with configured method
        t_start = tic;
        try
            gs_output = gnn.reach(gs_input, reachOpts);
            reach_time = toc(t_start);

            % Get output bounds (1 x 2 after AddPool + Dense)
            [lb, ub] = gs_output.getRanges();

            class0_lb = lb(1, 1); class0_ub = ub(1, 1);
            class1_lb = lb(1, 2); class1_ub = ub(1, 2);

            % Check robustness
            if pred_class == 0
                if class0_lb > class1_ub
                    result = 'VERIFIED'; n_verified = n_verified + 1;
                elseif class1_lb > class0_ub
                    result = 'VIOLATED'; n_violated = n_violated + 1;
                else
                    result = 'UNKNOWN'; n_unknown = n_unknown + 1;
                end
            else
                if class1_lb > class0_ub
                    result = 'VERIFIED'; n_verified = n_verified + 1;
                elseif class0_lb > class1_ub
                    result = 'VIOLATED'; n_violated = n_violated + 1;
                else
                    result = 'UNKNOWN'; n_unknown = n_unknown + 1;
                end
            end

            bound_gap = max(class0_ub - class0_lb, class1_ub - class1_lb);

        catch ME
            reach_time = toc(t_start);
            result = 'ERROR'; n_error = n_error + 1;
            bound_gap = NaN;
            fprintf('  ERROR on graph %d: %s\n', graph_id, ME.message);
        end

        total_time = total_time + reach_time;

        fprintf(fid, '%d,%d,%d,%d,%d,%s,%.6f,%.6f\n', ...
            graph_id, gd.n_nodes, gd.n_features, gd.true_label, pred_class, ...
            result, reach_time, bound_gap);

        if mod(gi, 20) == 0
            fprintf('  Progress: %d/%d (V:%d, X:%d, U:%d) time=%.1fs\n', ...
                gi, n_instances, n_verified, n_violated, n_unknown, total_time);
        end
    end

    fclose(fid);

    fprintf('\n--- Epsilon %.2f Summary (%s) ---\n', eps_val, reach_method);
    fprintf('VERIFIED: %d/%d (%.1f%%)\n', n_verified, n_instances, 100*n_verified/n_instances);
    fprintf('VIOLATED: %d/%d (%.1f%%)\n', n_violated, n_instances, 100*n_violated/n_instances);
    fprintf('UNKNOWN:  %d/%d (%.1f%%)\n', n_unknown, n_instances, 100*n_unknown/n_instances);
    if n_error > 0
        fprintf('ERROR:    %d/%d\n', n_error, n_instances);
    end
    fprintf('Total time: %.2f s (avg: %.4f s/instance)\n', total_time, total_time/n_instances);
    fprintf('Results saved to: %s\n', csv_path);
end

fprintf('\nDone! All results in: %s\n', results_dir);


%% Local function to read .gcinfo files
function gd = read_gcinfo_file(filepath)
    fid = fopen(filepath, 'r');
    header = str2num(fgetl(fid)); %#ok<ST2NM>
    gd.n_nodes = header(1);
    n_edges = header(2);
    gd.n_features = header(3);
    gd.n_classes = header(4);
    gd.true_label = header(5);

    src = str2num(fgetl(fid)); %#ok<ST2NM>
    dst = str2num(fgetl(fid)); %#ok<ST2NM>

    gd.A = zeros(gd.n_nodes, gd.n_nodes);
    for e = 1:n_edges
        gd.A(src(e)+1, dst(e)+1) = 1;
    end

    gd.X = zeros(gd.n_nodes, gd.n_features);
    for v = 1:gd.n_nodes
        line = strtrim(fgetl(fid));
        if ~isempty(line)
            feat_indices = str2num(line); %#ok<ST2NM>
            for fi = 1:length(feat_indices)
                gd.X(v, feat_indices(fi) + 1) = 1;
            end
        end
    end
    fclose(fid);
end
