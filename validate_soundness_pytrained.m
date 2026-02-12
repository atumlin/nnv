% validate_soundness_pytrained.m - Thorough soundness validation
%
% For each model, generates random samples within the input perturbation
% set, evaluates the GNN, and checks that every output falls within the
% computed reachability bounds. A single violation would indicate the
% reachability analysis is unsound.
%
% Tests: center point, vertices, and random interior samples.

run('code/nnv/startup_nnv.m');

models = {'gcn_pf_ieee24', 'gine_linear_pf_ieee24', 'gine_conv_pf_ieee24'};
model_dir = '/home/verivital/Anne/dev/gnn_training/outputs/ieee24_pf';

epsilon = 0.01;
num_random_samples = 1000;
tol = 1e-6;

fprintf('=== Soundness Validation for Python-Trained Models ===\n');
fprintf('Epsilon: %.3f | Random samples: %d | Tolerance: %.1e\n', ...
    epsilon, num_random_samples, tol);

for m = 1:length(models)
    fprintf('\n================================================================\n');
    fprintf('Model: %s\n', models{m});
    fprintf('================================================================\n');

    mat_path = fullfile(model_dir, [models{m} '.mat']);
    [gnn, test_data, ~] = gnn2nnv(mat_path);

    %% Build input set
    X = test_data.X;
    [N, F] = size(X);
    eps_matrix = epsilon * ones(N, F);
    GS_in = GraphStar(X, -eps_matrix, eps_matrix);

    %% Reachability
    reachOpts = struct('reachMethod', 'approx-star');
    t = tic;
    GS_out = gnn.reach(GS_in, reachOpts);
    t_reach = toc(t);

    [lb, ub] = GS_out.getRanges();
    fprintf('Reachability: %.4fs\n', t_reach);
    fprintf('Output bound widths — mean: %.6f, max: %.6f\n', ...
        mean(ub(:)-lb(:)), max(ub(:)-lb(:)));

    %% Test 1: Center point
    Y_center = gnn.evaluate(X);
    lb_viol = min(Y_center(:) - lb(:));
    ub_viol = min(ub(:) - Y_center(:));
    center_ok = lb_viol >= -tol && ub_viol >= -tol;
    fprintf('\n[Test 1] Center point: %s (margin lb: %.2e, ub: %.2e)\n', ...
        tf_str(center_ok), lb_viol, ub_viol);

    %% Test 2: Corner vertices (min/max of each predicate)
    num_pred = GS_in.numPred;
    vertex_violations = 0;
    num_vertices = 0;
    worst_vertex_margin = inf;

    % Test all-lower and all-upper corners
    corners = {GS_in.pred_lb, GS_in.pred_ub};
    corner_names = {'all-lower', 'all-upper'};

    for c = 1:length(corners)
        alpha = corners{c};
        X_corner = GS_in.V(:,:,1);
        for k = 1:num_pred
            X_corner = X_corner + alpha(k) * GS_in.V(:,:,k+1);
        end
        Y_corner = gnn.evaluate(X_corner);
        num_vertices = num_vertices + 1;

        lb_margin = min(Y_corner(:) - lb(:));
        ub_margin = min(ub(:) - Y_corner(:));
        worst_vertex_margin = min(worst_vertex_margin, min(lb_margin, ub_margin));

        if lb_margin < -tol || ub_margin < -tol
            vertex_violations = vertex_violations + 1;
            fprintf('  VIOLATION at %s corner: lb margin=%.2e, ub margin=%.2e\n', ...
                corner_names{c}, lb_margin, ub_margin);
        end
    end

    % Test random vertex combinations (each predicate at its min or max)
    num_random_vertices = min(200, 2^num_pred);
    for v = 1:num_random_vertices
        alpha = GS_in.pred_lb + (GS_in.pred_ub - GS_in.pred_lb) .* randi([0,1], num_pred, 1);
        X_vtx = GS_in.V(:,:,1);
        for k = 1:num_pred
            X_vtx = X_vtx + alpha(k) * GS_in.V(:,:,k+1);
        end
        Y_vtx = gnn.evaluate(X_vtx);
        num_vertices = num_vertices + 1;

        lb_margin = min(Y_vtx(:) - lb(:));
        ub_margin = min(ub(:) - Y_vtx(:));
        worst_vertex_margin = min(worst_vertex_margin, min(lb_margin, ub_margin));

        if lb_margin < -tol || ub_margin < -tol
            vertex_violations = vertex_violations + 1;
        end
    end

    fprintf('[Test 2] Vertex samples: %d tested, %d violations (worst margin: %.2e) %s\n', ...
        num_vertices, vertex_violations, worst_vertex_margin, tf_str(vertex_violations == 0));

    %% Test 3: Random interior samples
    random_violations = 0;
    worst_random_margin = inf;
    max_lb_violation = 0;
    max_ub_violation = 0;

    for s = 1:num_random_samples
        alpha = GS_in.pred_lb + (GS_in.pred_ub - GS_in.pred_lb) .* rand(num_pred, 1);
        X_sample = GS_in.V(:,:,1);
        for k = 1:num_pred
            X_sample = X_sample + alpha(k) * GS_in.V(:,:,k+1);
        end
        Y_sample = gnn.evaluate(X_sample);

        lb_margin = min(Y_sample(:) - lb(:));
        ub_margin = min(ub(:) - Y_sample(:));
        worst_random_margin = min(worst_random_margin, min(lb_margin, ub_margin));

        if lb_margin < -tol
            random_violations = random_violations + 1;
            max_lb_violation = max(max_lb_violation, -lb_margin);
        end
        if ub_margin < -tol
            random_violations = random_violations + 1;
            max_ub_violation = max(max_ub_violation, -ub_margin);
        end
    end

    fprintf('[Test 3] Random samples: %d tested, %d violations (worst margin: %.2e) %s\n', ...
        num_random_samples, random_violations, worst_random_margin, tf_str(random_violations == 0));

    if max_lb_violation > 0
        fprintf('  Max LB violation: %.2e\n', max_lb_violation);
    end
    if max_ub_violation > 0
        fprintf('  Max UB violation: %.2e\n', max_ub_violation);
    end

    %% Summary
    total_violations = ~center_ok + vertex_violations + random_violations;
    if total_violations == 0
        fprintf('\n==> SOUNDNESS VALIDATED (%d total samples, all within bounds)\n', ...
            1 + num_vertices + num_random_samples);
    else
        fprintf('\n==> SOUNDNESS FAILURE: %d violations detected!\n', total_violations);
    end
end

fprintf('\n==================== Validation Complete ====================\n');

function s = tf_str(val)
    if val
        s = 'PASSED';
    else
        s = 'FAILED';
    end
end
