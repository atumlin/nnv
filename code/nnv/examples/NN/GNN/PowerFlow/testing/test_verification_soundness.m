% test_verification_soundness.m - Verification Soundness Tests for Quick-Trained Models
%
% Tests both GCN and GINE models trained with the improved training pipeline
% (with bias terms) to verify:
%   1. Soundness: All sampled points fall within computed bounds
%   2. Voltage specification verification: [0.95, 1.05] p.u.
%
% Author: Claude Code (assisted)
% Date: 2026-01-20

%% Setup
clear; clc;
addpath(genpath('/home/verivital/Anne/dev/nnv/code/nnv'));

fprintf('========================================\n');
fprintf('GNN Verification Soundness Tests\n');
fprintf('========================================\n\n');

% Configuration
num_random_samples = 50;
num_boundary_samples = 10;
tolerance = 1e-6;
epsilon = 0.01;
perturb_features = [1, 2];  % Power injections

% Voltage specification
v_min = 0.95;
v_max = 1.05;
voltage_idx = 3;
bus_type_idx = 4;

% Get script directory
scriptDir = fileparts(mfilename('fullpath'));

all_tests_passed = true;

%% =========================================================================
%  TEST 1: GCN MODEL
%  =========================================================================
fprintf('--- GCN Model ---\n');

% Load model
fprintf('Loading model... ');
gcn_path = fullfile(scriptDir, 'models', 'gcn_test.mat');
if ~exist(gcn_path, 'file')
    fprintf('NOT FOUND (skipping)\n\n');
else
    gcn_model = load(gcn_path);
    fprintf('OK\n');

    % Extract weights with bias (handle dlarray)
    fprintf('Creating GNN... ');
    params = gcn_model.best_params;
    W1 = double(extractdata(gather(params.mult1.Weights)));
    W2 = double(extractdata(gather(params.mult2.Weights)));
    W3 = double(extractdata(gather(params.mult3.Weights)));
    b1 = double(extractdata(gather(params.mult1.Bias)));
    b2 = double(extractdata(gather(params.mult2.Bias)));
    b3 = double(extractdata(gather(params.mult3.Bias)));

    % Create layers (GCN with ReLU between hidden layers, no ReLU on output)
    L1 = GCNLayer('gcn1', W1, b1);
    R1 = ReluLayer();
    L2 = GCNLayer('gcn2', W2, b2);
    R2 = ReluLayer();
    L3 = GCNLayer('gcn3', W3, b3);  % No ReLU after output

    % Graph structure
    A_norm = double(gcn_model.ANorm_g);
    X = double(gcn_model.X_test_g{1});
    numNodes = size(X, 1);
    numFeatures = size(X, 2);

    % Create GNN
    gnn_gcn = GNN({L1, R1, L2, R2, L3}, A_norm);
    fprintf('OK\n');

    % Create input perturbation
    range_per_col = max(X) - min(X);
    LB = zeros(numNodes, numFeatures);
    UB = zeros(numNodes, numFeatures);
    for f = perturb_features
        LB(:, f) = -range_per_col(f) * epsilon;
        UB(:, f) = range_per_col(f) * epsilon;
    end
    GS_in = GraphStar(X, LB, UB);

    % Compute reachability
    fprintf('Computing reachability (eps=%.3f)... ', epsilon);
    t_start = tic;
    reachOpts = struct('reachMethod', 'approx-star');
    GS_out = gnn_gcn.reach(GS_in, reachOpts);
    [lb_out, ub_out] = GS_out.getRanges();
    fprintf('OK (%.2fs)\n', toc(t_start));

    % Test A: Center point
    fprintf('Test A: Center point... ');
    Y_center = gnn_gcn.evaluate(X);
    center_ok = all(Y_center(:) >= lb_out(:) - tolerance) && ...
                all(Y_center(:) <= ub_out(:) + tolerance);
    if center_ok
        fprintf('PASSED\n');
    else
        fprintf('FAILED\n');
        all_tests_passed = false;
    end

    % Test B: Random samples
    fprintf('Test B: Random samples (%d)... ', num_random_samples);
    random_ok = true;
    for i = 1:num_random_samples
        % Sample random perturbation
        alpha = rand(GS_in.numPred, 1) .* (GS_in.pred_ub - GS_in.pred_lb) + GS_in.pred_lb;
        X_sample = GS_in.evaluate(alpha);
        Y_sample = gnn_gcn.evaluate(X_sample);

        if ~(all(Y_sample(:) >= lb_out(:) - tolerance) && ...
             all(Y_sample(:) <= ub_out(:) + tolerance))
            random_ok = false;
            break;
        end
    end
    if random_ok
        fprintf('PASSED\n');
    else
        fprintf('FAILED\n');
        all_tests_passed = false;
    end

    % Test C: Boundary samples
    fprintf('Test C: Boundary samples (%d)... ', num_boundary_samples);
    boundary_ok = true;
    for corner = 1:min(num_boundary_samples, 2^min(GS_in.numPred, 20))
        alpha_bits = bitget(corner, 1:min(GS_in.numPred, 20))';
        if length(alpha_bits) < GS_in.numPred
            alpha_bits = [alpha_bits; zeros(GS_in.numPred - length(alpha_bits), 1)];
        end
        alpha = GS_in.pred_lb + alpha_bits .* (GS_in.pred_ub - GS_in.pred_lb);
        X_sample = GS_in.evaluate(alpha);
        Y_sample = gnn_gcn.evaluate(X_sample);

        if ~(all(Y_sample(:) >= lb_out(:) - tolerance) && ...
             all(Y_sample(:) <= ub_out(:) + tolerance))
            boundary_ok = false;
            break;
        end
    end
    if boundary_ok
        fprintf('PASSED\n');
    else
        fprintf('FAILED\n');
        all_tests_passed = false;
    end

    % Voltage verification
    fprintf('\n--- Voltage Magnitude Verification (GCN) ---\n');
    run_voltage_verification(gcn_model, GS_out, lb_out, ub_out, X, v_min, v_max, voltage_idx, bus_type_idx);

    fprintf('\n');
end

%% =========================================================================
%  TEST 2: GINE MODEL
%  =========================================================================
fprintf('--- GINE Model ---\n');

% Load model
fprintf('Loading model... ');
gine_path = fullfile(scriptDir, 'models', 'gine_test.mat');
if ~exist(gine_path, 'file')
    fprintf('NOT FOUND (skipping)\n\n');
else
    gine_model = load(gine_path);
    fprintf('OK\n');

    % Extract weights with bias (handle dlarray)
    fprintf('Creating GNN... ');
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

    % Create layers (GINE with learned bias)
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
    X = double(gine_model.X_test_g{1});
    numNodes = size(X, 1);
    numFeatures = size(X, 2);

    % Create GNN
    gnn_gine = GNN({L1, L2, L3}, [], adj_list, E, edge_weights);
    fprintf('OK\n');

    % Create input perturbation
    range_per_col = max(X) - min(X);
    LB = zeros(numNodes, numFeatures);
    UB = zeros(numNodes, numFeatures);
    for f = perturb_features
        LB(:, f) = -range_per_col(f) * epsilon;
        UB(:, f) = range_per_col(f) * epsilon;
    end
    GS_in = GraphStar(X, LB, UB);

    % Compute reachability
    fprintf('Computing reachability (eps=%.3f)... ', epsilon);
    t_start = tic;
    reachOpts = struct('reachMethod', 'approx-star');
    GS_out = gnn_gine.reach(GS_in, reachOpts);
    [lb_out, ub_out] = GS_out.getRanges();
    fprintf('OK (%.2fs)\n', toc(t_start));

    % Test A: Center point
    fprintf('Test A: Center point... ');
    Y_center = gnn_gine.evaluate(X);
    center_ok = all(Y_center(:) >= lb_out(:) - tolerance) && ...
                all(Y_center(:) <= ub_out(:) + tolerance);
    if center_ok
        fprintf('PASSED\n');
    else
        fprintf('FAILED\n');
        all_tests_passed = false;
    end

    % Test B: Random samples
    fprintf('Test B: Random samples (%d)... ', num_random_samples);
    random_ok = true;
    for i = 1:num_random_samples
        alpha = rand(GS_in.numPred, 1) .* (GS_in.pred_ub - GS_in.pred_lb) + GS_in.pred_lb;
        X_sample = GS_in.evaluate(alpha);
        Y_sample = gnn_gine.evaluate(X_sample);

        if ~(all(Y_sample(:) >= lb_out(:) - tolerance) && ...
             all(Y_sample(:) <= ub_out(:) + tolerance))
            random_ok = false;
            break;
        end
    end
    if random_ok
        fprintf('PASSED\n');
    else
        fprintf('FAILED\n');
        all_tests_passed = false;
    end

    % Test C: Boundary samples
    fprintf('Test C: Boundary samples (%d)... ', num_boundary_samples);
    boundary_ok = true;
    for corner = 1:min(num_boundary_samples, 2^min(GS_in.numPred, 20))
        alpha_bits = bitget(corner, 1:min(GS_in.numPred, 20))';
        if length(alpha_bits) < GS_in.numPred
            alpha_bits = [alpha_bits; zeros(GS_in.numPred - length(alpha_bits), 1)];
        end
        alpha = GS_in.pred_lb + alpha_bits .* (GS_in.pred_ub - GS_in.pred_lb);
        X_sample = GS_in.evaluate(alpha);
        Y_sample = gnn_gine.evaluate(X_sample);

        if ~(all(Y_sample(:) >= lb_out(:) - tolerance) && ...
             all(Y_sample(:) <= ub_out(:) + tolerance))
            boundary_ok = false;
            break;
        end
    end
    if boundary_ok
        fprintf('PASSED\n');
    else
        fprintf('FAILED\n');
        all_tests_passed = false;
    end

    % Voltage verification
    fprintf('\n--- Voltage Magnitude Verification (GINE) ---\n');
    run_voltage_verification(gine_model, GS_out, lb_out, ub_out, X, v_min, v_max, voltage_idx, bus_type_idx);

    fprintf('\n');
end

%% =========================================================================
%  SUMMARY
%  =========================================================================
fprintf('========================================\n');
if all_tests_passed
    fprintf('All soundness tests PASSED!\n');
else
    fprintf('Some tests FAILED - check output above\n');
end
fprintf('========================================\n');

%% =========================================================================
%  HELPER FUNCTIONS
%  =========================================================================

function run_voltage_verification(model, GS_out, lb_out, ub_out, X, v_min, v_max, voltage_idx, bus_type_idx)
    % Normalize bounds to model space
    v_min_norm = (v_min - model.global_mean_labels(voltage_idx)) / model.global_std_labels(voltage_idx);
    v_max_norm = (v_max - model.global_mean_labels(voltage_idx)) / model.global_std_labels(voltage_idx);

    % Identify voltage-output nodes (bus_type == 1)
    X_physical = X .* model.global_std + model.global_mean;
    voltage_mask = (X_physical(:, bus_type_idx) == 1);

    % Extract voltage bounds
    voltage_lb = lb_out(:, voltage_idx);
    voltage_ub = ub_out(:, voltage_idx);

    % Count verification results
    numNodes = size(X, 1);
    verified = 0;
    violated = 0;
    unknown = 0;

    for i = 1:numNodes
        if ~voltage_mask(i)
            continue;
        end

        if voltage_lb(i) >= v_min_norm && voltage_ub(i) <= v_max_norm
            verified = verified + 1;
        elseif voltage_ub(i) < v_min_norm || voltage_lb(i) > v_max_norm
            violated = violated + 1;
        else
            unknown = unknown + 1;
        end
    end

    fprintf('Voltage Verification: %d verified, %d violated, %d unknown\n', verified, violated, unknown);

    % Convert to physical units and display table
    voltage_lb_phys = voltage_lb * model.global_std_labels(voltage_idx) + model.global_mean_labels(voltage_idx);
    voltage_ub_phys = voltage_ub * model.global_std_labels(voltage_idx) + model.global_mean_labels(voltage_idx);

    fprintf('\nVoltage Bounds (physical units):\n');
    fprintf('Node | Lower (p.u.) | Upper (p.u.) | Width     | Status\n');
    fprintf('-----|--------------|--------------|-----------|--------\n');

    for i = 1:numNodes
        if voltage_mask(i)
            width = voltage_ub_phys(i) - voltage_lb_phys(i);
            if voltage_lb_phys(i) >= v_min && voltage_ub_phys(i) <= v_max
                status = 'SAFE';
            elseif voltage_ub_phys(i) < v_min || voltage_lb_phys(i) > v_max
                status = 'VIOLATED';
            else
                status = 'UNKNOWN';
            end
            fprintf('%4d | %12.4f | %12.4f | %9.6f | %s\n', i, voltage_lb_phys(i), voltage_ub_phys(i), width, status);
        end
    end
end
