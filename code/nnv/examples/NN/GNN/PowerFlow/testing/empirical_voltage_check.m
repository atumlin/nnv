% empirical_voltage_check.m - Validate voltage verification results empirically
%
% This script checks why GCN had all "unknown" and GINE had all "violated"

%% Setup
clear; clc;
addpath(genpath('/home/verivital/Anne/dev/nnv/code/nnv'));

scriptDir = fileparts(mfilename('fullpath'));

% Voltage specification
v_min = 0.95;
v_max = 1.05;
voltage_idx = 3;
bus_type_idx = 4;

fprintf('==============================================\n');
fprintf('Empirical Voltage Validation\n');
fprintf('Specification: [%.2f, %.2f] p.u.\n', v_min, v_max);
fprintf('==============================================\n\n');

%% Load and check GCN model
fprintf('--- GCN Model ---\n');
gcn_path = fullfile(scriptDir, 'models', 'gcn_test.mat');
if exist(gcn_path, 'file')
    gcn_model = load(gcn_path);

    % Get test data
    X_test = double(gcn_model.X_test_g{1});
    Y_true = double(gcn_model.Y_test_g{1});
    A_norm = double(gcn_model.ANorm_g);

    % Extract weights
    params = gcn_model.best_params;
    W1 = double(extractdata(gather(params.mult1.Weights)));
    W2 = double(extractdata(gather(params.mult2.Weights)));
    W3 = double(extractdata(gather(params.mult3.Weights)));
    b1 = double(extractdata(gather(params.mult1.Bias)));
    b2 = double(extractdata(gather(params.mult2.Bias)));
    b3 = double(extractdata(gather(params.mult3.Bias)));

    % Manual forward pass (GCN with ReLU between layers)
    H1 = A_norm * X_test * W1 + b1';
    H1 = max(0, H1);  % ReLU
    H2 = A_norm * H1 * W2 + b2';
    H2 = max(0, H2);  % ReLU
    Y_pred_norm = A_norm * H2 * W3 + b3';  % No ReLU on output

    % Convert to physical units
    Y_pred_phys = Y_pred_norm .* gcn_model.global_std_labels + gcn_model.global_mean_labels;
    Y_true_phys = Y_true .* gcn_model.global_std_labels + gcn_model.global_mean_labels;
    X_phys = X_test .* gcn_model.global_std + gcn_model.global_mean;

    % Identify voltage nodes (bus_type == 1)
    voltage_mask = (X_phys(:, bus_type_idx) == 1);

    % Extract voltage predictions and ground truth
    V_pred = Y_pred_phys(:, voltage_idx);
    V_true = Y_true_phys(:, voltage_idx);

    fprintf('\nVoltage Predictions vs Ground Truth (GCN):\n');
    fprintf('Node | Predicted (p.u.) | True (p.u.) | Error    | In Spec?\n');
    fprintf('-----|------------------|-------------|----------|----------\n');

    for i = 1:size(X_test, 1)
        if voltage_mask(i)
            err = abs(V_pred(i) - V_true(i));
            in_spec_pred = (V_pred(i) >= v_min && V_pred(i) <= v_max);
            in_spec_true = (V_true(i) >= v_min && V_true(i) <= v_max);

            if in_spec_pred
                spec_str = 'YES';
            else
                spec_str = 'NO';
            end

            fprintf('%4d | %16.4f | %11.4f | %8.4f | %s (true: %s)\n', ...
                i, V_pred(i), V_true(i), err, spec_str, ternary(in_spec_true, 'YES', 'NO'));
        end
    end

    % Statistics
    V_pred_volt = V_pred(voltage_mask);
    V_true_volt = V_true(voltage_mask);

    fprintf('\nGCN Statistics:\n');
    fprintf('  Prediction range: [%.4f, %.4f] p.u.\n', min(V_pred_volt), max(V_pred_volt));
    fprintf('  True range:       [%.4f, %.4f] p.u.\n', min(V_true_volt), max(V_true_volt));
    fprintf('  Voltage RMSE:     %.4f p.u.\n', sqrt(mean((V_pred_volt - V_true_volt).^2)));
    fprintf('  Predictions in spec: %d / %d\n', sum(V_pred_volt >= v_min & V_pred_volt <= v_max), length(V_pred_volt));
    fprintf('  True values in spec: %d / %d\n', sum(V_true_volt >= v_min & V_true_volt <= v_max), length(V_true_volt));
else
    fprintf('GCN model not found.\n');
end

fprintf('\n');

%% Load and check GINE model
fprintf('--- GINE Model ---\n');
gine_path = fullfile(scriptDir, 'models', 'gine_test.mat');
if exist(gine_path, 'file')
    gine_model = load(gine_path);

    % Get test data
    X_test = double(gine_model.X_test_g{1});
    Y_true = double(gine_model.Y_test_g{1});
    src = double(gine_model.src);
    dst = double(gine_model.dst);
    E = double(gine_model.E_edge);
    a = double(gine_model.a);

    % Extract weights
    params = gine_model.best_params;
    W1 = double(extractdata(gather(params.mult1.Weights)));
    W2 = double(extractdata(gather(params.mult2.Weights)));
    W3 = double(extractdata(gather(params.mult3.Weights)));
    b1 = double(extractdata(gather(params.mult1.Bias)));
    b2 = double(extractdata(gather(params.mult2.Bias)));
    b3 = double(extractdata(gather(params.mult3.Bias)));
    W_e1 = double(extractdata(gather(params.edge1.Weights)));
    W_e2 = double(extractdata(gather(params.edge2.Weights)));
    W_e3 = double(extractdata(gather(params.edge3.Weights)));
    b_e1 = double(extractdata(gather(params.edge1.Bias)));
    b_e2 = double(extractdata(gather(params.edge2.Bias)));
    b_e3 = double(extractdata(gather(params.edge3.Bias)));

    % Manual GINE forward pass
    % Note: In GINE, node features are transformed BEFORE adding to edge projections
    numNodes = size(X_test, 1);

    % Layer 1: H (24x4) -> (24x32)
    H = X_test;
    P = E * W_e1 + b_e1';  % (92x32) edge projections
    out_dim = size(W1, 2);
    M = zeros(numNodes, out_dim);
    for e = 1:length(src)
        Hsrc = H(src(e), :) * W1 + b1';  % Transform source node features first (1x32)
        M(dst(e), :) = M(dst(e), :) + a(e) * (Hsrc + P(e, :));
    end
    H1 = max(0, M);  % ReLU

    % Layer 2: H (24x32) -> (24x32)
    H = H1;
    P = E * W_e2 + b_e2';
    out_dim = size(W2, 2);
    M = zeros(numNodes, out_dim);
    for e = 1:length(src)
        Hsrc = H(src(e), :) * W2 + b2';
        M(dst(e), :) = M(dst(e), :) + a(e) * (Hsrc + P(e, :));
    end
    H2 = max(0, M);  % ReLU

    % Layer 3 (no ReLU): H (24x32) -> (24x4)
    H = H2;
    P = E * W_e3 + b_e3';
    out_dim = size(W3, 2);
    M = zeros(numNodes, out_dim);
    for e = 1:length(src)
        Hsrc = H(src(e), :) * W3 + b3';
        M(dst(e), :) = M(dst(e), :) + a(e) * (Hsrc + P(e, :));
    end
    Y_pred_norm = M;  % Linear output

    % Convert to physical units
    Y_pred_phys = Y_pred_norm .* gine_model.global_std_labels + gine_model.global_mean_labels;
    Y_true_phys = Y_true .* gine_model.global_std_labels + gine_model.global_mean_labels;
    X_phys = X_test .* gine_model.global_std + gine_model.global_mean;

    % Identify voltage nodes (bus_type == 1)
    voltage_mask = (X_phys(:, bus_type_idx) == 1);

    % Extract voltage predictions and ground truth
    V_pred = Y_pred_phys(:, voltage_idx);
    V_true = Y_true_phys(:, voltage_idx);

    fprintf('\nVoltage Predictions vs Ground Truth (GINE):\n');
    fprintf('Node | Predicted (p.u.) | True (p.u.) | Error    | In Spec?\n');
    fprintf('-----|------------------|-------------|----------|----------\n');

    for i = 1:size(X_test, 1)
        if voltage_mask(i)
            err = abs(V_pred(i) - V_true(i));
            in_spec_pred = (V_pred(i) >= v_min && V_pred(i) <= v_max);
            in_spec_true = (V_true(i) >= v_min && V_true(i) <= v_max);

            if in_spec_pred
                spec_str = 'YES';
            else
                spec_str = 'NO';
            end

            fprintf('%4d | %16.4f | %11.4f | %8.4f | %s (true: %s)\n', ...
                i, V_pred(i), V_true(i), err, spec_str, ternary(in_spec_true, 'YES', 'NO'));
        end
    end

    % Statistics
    V_pred_volt = V_pred(voltage_mask);
    V_true_volt = V_true(voltage_mask);

    fprintf('\nGINE Statistics:\n');
    fprintf('  Prediction range: [%.4f, %.4f] p.u.\n', min(V_pred_volt), max(V_pred_volt));
    fprintf('  True range:       [%.4f, %.4f] p.u.\n', min(V_true_volt), max(V_true_volt));
    fprintf('  Voltage RMSE:     %.4f p.u.\n', sqrt(mean((V_pred_volt - V_true_volt).^2)));
    fprintf('  Predictions in spec: %d / %d\n', sum(V_pred_volt >= v_min & V_pred_volt <= v_max), length(V_pred_volt));
    fprintf('  True values in spec: %d / %d\n', sum(V_true_volt >= v_min & V_true_volt <= v_max), length(V_true_volt));
else
    fprintf('GINE model not found.\n');
end

fprintf('\n==============================================\n');
fprintf('Analysis Complete\n');
fprintf('==============================================\n');

%% Helper function
function result = ternary(cond, a, b)
    if cond
        result = a;
    else
        result = b;
    end
end
