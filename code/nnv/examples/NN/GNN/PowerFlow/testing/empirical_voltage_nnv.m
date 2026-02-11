% empirical_voltage_nnv.m - Use NNV GNN.evaluate() for accurate validation
%
% Uses the same GNN construction as the verification test

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
fprintf('Empirical Voltage Validation (Using NNV)\n');
fprintf('Specification: [%.2f, %.2f] p.u.\n', v_min, v_max);
fprintf('==============================================\n\n');

%% GCN Model
fprintf('--- GCN Model ---\n');
gcn_path = fullfile(scriptDir, 'models', 'gcn_test.mat');
if exist(gcn_path, 'file')
    gcn_model = load(gcn_path);

    % Extract weights (same as verification test)
    params = gcn_model.best_params;
    W1 = double(extractdata(gather(params.mult1.Weights)));
    W2 = double(extractdata(gather(params.mult2.Weights)));
    W3 = double(extractdata(gather(params.mult3.Weights)));
    b1 = double(extractdata(gather(params.mult1.Bias)));
    b2 = double(extractdata(gather(params.mult2.Bias)));
    b3 = double(extractdata(gather(params.mult3.Bias)));

    % Create GNN (same as verification test)
    L1 = GCNLayer('gcn1', W1, b1);
    R1 = ReluLayer();
    L2 = GCNLayer('gcn2', W2, b2);
    R2 = ReluLayer();
    L3 = GCNLayer('gcn3', W3, b3);

    A_norm = double(gcn_model.ANorm_g);
    X_test = double(gcn_model.X_test_g{1});
    Y_true = double(gcn_model.Y_test_g{1});

    gnn_gcn = GNN({L1, R1, L2, R2, L3}, A_norm);

    % Evaluate using GNN.evaluate()
    Y_pred_norm = gnn_gcn.evaluate(X_test);

    % Convert to physical units
    Y_pred_phys = Y_pred_norm .* gcn_model.global_std_labels + gcn_model.global_mean_labels;
    Y_true_phys = Y_true .* gcn_model.global_std_labels + gcn_model.global_mean_labels;
    X_phys = X_test .* gcn_model.global_std + gcn_model.global_mean;

    % Identify voltage nodes
    voltage_mask = (X_phys(:, bus_type_idx) == 1);

    V_pred = Y_pred_phys(:, voltage_idx);
    V_true = Y_true_phys(:, voltage_idx);

    fprintf('\nVoltage Predictions vs Ground Truth (GCN via NNV):\n');
    fprintf('Node | Predicted (p.u.) | True (p.u.) | Error    | In Spec?\n');
    fprintf('-----|------------------|-------------|----------|----------\n');

    for i = 1:size(X_test, 1)
        if voltage_mask(i)
            err = abs(V_pred(i) - V_true(i));
            in_spec_pred = (V_pred(i) >= v_min && V_pred(i) <= v_max);
            fprintf('%4d | %16.4f | %11.4f | %8.4f | %s (true: %s)\n', ...
                i, V_pred(i), V_true(i), err, ternary(in_spec_pred, 'YES', 'NO'), ...
                ternary(V_true(i) >= v_min && V_true(i) <= v_max, 'YES', 'NO'));
        end
    end

    V_pred_volt = V_pred(voltage_mask);
    V_true_volt = V_true(voltage_mask);

    fprintf('\nGCN Statistics:\n');
    fprintf('  Prediction range: [%.4f, %.4f] p.u.\n', min(V_pred_volt), max(V_pred_volt));
    fprintf('  True range:       [%.4f, %.4f] p.u.\n', min(V_true_volt), max(V_true_volt));
    fprintf('  Voltage RMSE:     %.4f p.u.\n', sqrt(mean((V_pred_volt - V_true_volt).^2)));
    fprintf('  Predictions in spec: %d / %d\n', sum(V_pred_volt >= v_min & V_pred_volt <= v_max), length(V_pred_volt));
end

fprintf('\n');

%% GINE Model
fprintf('--- GINE Model ---\n');
gine_path = fullfile(scriptDir, 'models', 'gine_test.mat');
if exist(gine_path, 'file')
    gine_model = load(gine_path);

    % Extract weights (same as verification test)
    params = gine_model.best_params;
    W_node1 = double(extractdata(gather(params.mult1.Weights)));
    W_node2 = double(extractdata(gather(params.mult2.Weights)));
    W_node3 = double(extractdata(gather(params.mult3.Weights)));
    b_node1 = double(extractdata(gather(params.mult1.Bias)));
    b_node2 = double(extractdata(gather(params.mult2.Bias)));
    b_node3 = double(extractdata(gather(params.mult3.Bias)));
    W_edge1 = double(extractdata(gather(params.edge1.Weights)));
    W_edge2 = double(extractdata(gather(params.edge2.Weights)));
    W_edge3 = double(extractdata(gather(params.edge3.Weights)));
    b_edge1 = double(extractdata(gather(params.edge1.Bias)));
    b_edge2 = double(extractdata(gather(params.edge2.Bias)));
    b_edge3 = double(extractdata(gather(params.edge3.Bias)));

    % Create layers (same as verification test)
    % ApplyRelu=true for hidden layers (default), false for output layer
    L1 = GINELayer('gine1', W_node1, b_node1, W_edge1, b_edge1);
    L2 = GINELayer('gine2', W_node2, b_node2, W_edge2, b_edge2);
    L3 = GINELayer('gine3', W_node3, b_node3, W_edge3, b_edge3);
    L3.ApplyRelu = false;  % Output layer: no ReLU

    % Graph structure
    src = double(gine_model.src);
    dst = double(gine_model.dst);
    adj_list = [src, dst];
    E = double(gine_model.E_edge);
    edge_weights = double(gine_model.a);
    X_test = double(gine_model.X_test_g{1});
    Y_true = double(gine_model.Y_test_g{1});

    % Create GNN (same as verification test)
    gnn_gine = GNN({L1, L2, L3}, [], adj_list, E, edge_weights);

    % Evaluate using GNN.evaluate()
    Y_pred_norm = gnn_gine.evaluate(X_test);

    % Convert to physical units
    Y_pred_phys = Y_pred_norm .* gine_model.global_std_labels + gine_model.global_mean_labels;
    Y_true_phys = Y_true .* gine_model.global_std_labels + gine_model.global_mean_labels;
    X_phys = X_test .* gine_model.global_std + gine_model.global_mean;

    % Identify voltage nodes
    voltage_mask = (X_phys(:, bus_type_idx) == 1);

    V_pred = Y_pred_phys(:, voltage_idx);
    V_true = Y_true_phys(:, voltage_idx);

    fprintf('\nVoltage Predictions vs Ground Truth (GINE via NNV):\n');
    fprintf('Node | Predicted (p.u.) | True (p.u.) | Error    | In Spec?\n');
    fprintf('-----|------------------|-------------|----------|----------\n');

    for i = 1:size(X_test, 1)
        if voltage_mask(i)
            err = abs(V_pred(i) - V_true(i));
            in_spec_pred = (V_pred(i) >= v_min && V_pred(i) <= v_max);
            fprintf('%4d | %16.4f | %11.4f | %8.4f | %s (true: %s)\n', ...
                i, V_pred(i), V_true(i), err, ternary(in_spec_pred, 'YES', 'NO'), ...
                ternary(V_true(i) >= v_min && V_true(i) <= v_max, 'YES', 'NO'));
        end
    end

    V_pred_volt = V_pred(voltage_mask);
    V_true_volt = V_true(voltage_mask);

    fprintf('\nGINE Statistics:\n');
    fprintf('  Prediction range: [%.4f, %.4f] p.u.\n', min(V_pred_volt), max(V_pred_volt));
    fprintf('  True range:       [%.4f, %.4f] p.u.\n', min(V_true_volt), max(V_true_volt));
    fprintf('  Voltage RMSE:     %.4f p.u.\n', sqrt(mean((V_pred_volt - V_true_volt).^2)));
    fprintf('  Predictions in spec: %d / %d\n', sum(V_pred_volt >= v_min & V_pred_volt <= v_max), length(V_pred_volt));
end

fprintf('\n==============================================\n');
fprintf('Analysis Complete\n');
fprintf('==============================================\n');

function result = ternary(cond, a, b)
    if cond
        result = a;
    else
        result = b;
    end
end
