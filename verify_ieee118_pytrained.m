% verify_ieee118_pytrained.m - Verify Python-trained models on real IEEE118 data
%
% Runs reachability analysis and checks voltage safety specification:
%   0.95 <= V_mag <= 1.05 p.u.
%
% Output features: [P_flow, Q_flow, V_mag, V_angle]
% Normalization: max normalization (Y_norm = Y / Y_max)

run('code/nnv/startup_nnv.m');

models = {'gcn_pf_ieee118', 'gine_linear_pf_ieee118', 'gine_conv_pf_ieee118'};
model_dir = '/home/verivital/Anne/dev/gnn_training/outputs/ieee118_pf';

% Voltage safety specification (per-unit)
v_min = 0.95;
v_max = 1.05;

% Feature indices (1-indexed)
voltage_idx = 3;   % V_mag is output feature 3
bus_type_idx = 4;  % Bus type is input feature 4

% Perturbation settings
epsilon = 0.01;  % 1% perturbation on normalized inputs

for m = 1:length(models)
    fprintf('\n================================================================\n');
    fprintf('Model: %s\n', models{m});
    fprintf('================================================================\n');
    mat_path = fullfile(model_dir, [models{m} '.mat']);

    % Load model
    raw_model = load(mat_path);
    [gnn, test_data, norm_stats] = gnn2nnv(mat_path);
    Y_max = norm_stats.Y_max(:)';

    % Identify PQ buses (bus_type == 1) — only these have voltage as output
    X_max = norm_stats.X_max(:)';
    X_phys = test_data.X .* X_max;
    bus_types = round(X_phys(:, bus_type_idx));
    voltage_mask = (bus_types == 1);
    pq_nodes = find(voltage_mask);

    % Evaluate center point
    Y_norm = gnn.evaluate(test_data.X);
    Y_phys = Y_norm .* Y_max;
    fprintf('Nodes: %d (%d PQ buses) | Output features: %d\n', ...
        size(Y_norm,1), sum(voltage_mask), size(Y_norm,2));
    fprintf('Center V_mag range (PQ buses): [%.4f, %.4f] p.u.\n', ...
        min(Y_phys(voltage_mask,voltage_idx)), max(Y_phys(voltage_mask,voltage_idx)));

    %% Reachability analysis
    X = test_data.X;
    eps_matrix = epsilon * ones(size(X));
    GS_in = GraphStar(X, -eps_matrix, eps_matrix);

    reachOpts = struct('reachMethod', 'approx-star');
    t = tic;
    GS_out = gnn.reach(GS_in, reachOpts);
    t_reach = toc(t);

    % Get normalized output bounds
    [lb_norm, ub_norm] = GS_out.getRanges();

    % Soundness check
    Y_center = gnn.evaluate(GS_in.V(:,:,1));
    tol = 1e-6;
    assert(all(Y_center(:) >= lb_norm(:) - tol), 'LB soundness fail');
    assert(all(Y_center(:) <= ub_norm(:) + tol), 'UB soundness fail');
    fprintf('Reachability: %.4fs | Soundness: PASSED\n', t_reach);

    %% Voltage specification verification (PQ buses only)
    % Denormalize voltage bounds to physical units
    V_mag_Y_max = Y_max(voltage_idx);
    lb_v = lb_norm(:, voltage_idx) * V_mag_Y_max;
    ub_v = ub_norm(:, voltage_idx) * V_mag_Y_max;

    numPQ = length(pq_nodes);
    verified = 0;
    violated = 0;
    unknown = 0;

    fprintf('\n--- Voltage Specification: %.2f <= V_mag <= %.2f p.u. (PQ buses only) ---\n', v_min, v_max);
    fprintf('%-6s  %-12s  %-12s  %-10s  %-8s\n', 'Node', 'V_lb (p.u.)', 'V_ub (p.u.)', 'Width', 'Status');
    fprintf('%s\n', repmat('-', 1, 56));

    for k = 1:numPQ
        i = pq_nodes(k);
        width = ub_v(i) - lb_v(i);

        if lb_v(i) >= v_min && ub_v(i) <= v_max
            status = 'SAFE';
            verified = verified + 1;
        elseif ub_v(i) < v_min || lb_v(i) > v_max
            status = 'VIOLATED';
            violated = violated + 1;
        else
            status = 'UNKNOWN';
            unknown = unknown + 1;
        end

        fprintf('%-6d  %-12.6f  %-12.6f  %-10.6f  %-8s\n', i, lb_v(i), ub_v(i), width, status);
    end

    fprintf('%s\n', repmat('-', 1, 56));
    fprintf('Verified SAFE: %d/%d PQ buses | VIOLATED: %d | UNKNOWN: %d\n', ...
        verified, numPQ, violated, unknown);

    if violated == 0 && unknown == 0
        fprintf('==> ALL PQ BUSES VERIFIED SAFE for epsilon=%.3f\n', epsilon);
    elseif violated > 0
        fprintf('==> SPECIFICATION VIOLATED at %d PQ buses\n', violated);
    else
        fprintf('==> %d PQ buses inconclusive (bounds cross spec boundary)\n', unknown);
    end
end

fprintf('\n==================== Verification Complete ====================\n');
