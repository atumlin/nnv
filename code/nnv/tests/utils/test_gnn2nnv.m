% test_gnn2nnv.m - Unit tests for gnn2nnv import utility
%
% Tests: GCN import, GINE-linear import, auto-detection, validation,
%        error handling, test data extraction, normalization stats
%
% Creates synthetic .mat files to test each import path.

% Shared setup: create a temp directory for test .mat files
tmp_dir = fullfile(tempdir, 'test_gnn2nnv');
if ~exist(tmp_dir, 'dir')
    mkdir(tmp_dir);
end

% Shared constants
N = 6; F_in = 3; F_h = 4; F_out = 2;

% Shared adjacency matrix
rng(42);
A = rand(N) > 0.5; A = double(A | A'); A(1:N+1:end) = 0;
D = diag(sum(A, 2));
ANorm_g = (D + eye(N))^(-0.5) * (A + eye(N)) * (D + eye(N))^(-0.5);

%% 1) GCN import test
rng(42);
bp = struct();
bp.mult1.Weights = rand(F_in, F_h);
bp.mult2.Weights = rand(F_h, F_out);
Xt = {rand(N, F_in)};
Yt = {rand(N, F_out)};
mt = 'gcn';

p1 = fullfile(tmp_dir, 'test_gcn.mat');
save_gnn_mat(p1, struct('best_params', bp, 'ANorm_g', ANorm_g, ...
    'X_test_g', {Xt}, 'Y_test_g', {Yt}, 'model_type', mt));

[gnn1, td1] = gnn2nnv(p1);
assert(isa(gnn1, 'GNN'), 'Should return GNN object');
assert(gnn1.numLayers == 4, 'GCN with 2 conv + 2 ReLU = 4 layers');
assert(~isempty(gnn1.A_norm), 'Should have adjacency matrix');

Y1 = gnn1.evaluate(td1.X);
assert(size(Y1, 1) == N, 'Output should have N nodes');
assert(size(Y1, 2) == F_out, 'Output should have F_out features');

%% 2) GCN auto-detection (no explicit model_type)
rng(42);
bp2 = struct();
bp2.mult1.Weights = rand(F_in, F_h);
bp2.mult2.Weights = rand(F_h, F_out);
Xt2 = {rand(N, F_in)};
Yt2 = {rand(N, F_out)};

p2 = fullfile(tmp_dir, 'test_gcn_auto.mat');
save_gnn_mat(p2, struct('best_params', bp2, 'ANorm_g', ANorm_g, ...
    'X_test_g', {Xt2}, 'Y_test_g', {Yt2}));

[gnn2, ~] = gnn2nnv(p2);
assert(gnn2.numLayers == 4, 'Auto-detected GCN should have 4 layers');

%% 3) GCN with explicit biases
rng(42);
bp3 = struct();
bp3.mult1.Weights = rand(F_in, F_h);
bp3.mult1.Bias = rand(F_h, 1);
bp3.mult2.Weights = rand(F_h, F_out);
bp3.mult2.Bias = rand(F_out, 1);
Xt3 = {rand(N, F_in)};
Yt3 = {rand(N, F_out)};

p3 = fullfile(tmp_dir, 'test_gcn_bias.mat');
save_gnn_mat(p3, struct('best_params', bp3, 'ANorm_g', ANorm_g, ...
    'X_test_g', {Xt3}, 'Y_test_g', {Yt3}, 'model_type', 'gcn'));

[gnn3, ~] = gnn2nnv(p3);
Y3 = gnn3.evaluate(Xt3{1});
assert(all(size(Y3) == [N, F_out]), 'Biased GCN should produce correct output shape');

%% 4) GINE-linear import test
rng(42);
N4 = 5; Fin4 = 4; Fout4 = 6; Ein4 = 3; nE4 = 8;

bp4 = struct();
bp4.mult1.Weights = rand(Fin4, Fout4);
bp4.edge1.Weights = rand(Ein4, Fout4);

src4 = randi(N4, nE4, 1); dst4 = randi(N4, nE4, 1);
E4 = rand(nE4, Ein4); a4 = rand(nE4, 1);
Xt4 = {rand(N4, Fin4)}; Yt4 = {rand(N4, Fout4)};

p4 = fullfile(tmp_dir, 'test_gine_linear.mat');
save_gnn_mat(p4, struct('best_params', bp4, 'src', src4, 'dst', dst4, ...
    'E_edge', E4, 'a', a4, 'X_test_g', {Xt4}, 'Y_test_g', {Yt4}, ...
    'model_type', 'gine_linear'));

[gnn4, td4] = gnn2nnv(p4);
assert(isa(gnn4, 'GNN'), 'Should return GNN object');
assert(gnn4.numLayers == 1, 'GINE-linear with 1 layer');
assert(~isempty(gnn4.adj_list), 'Should have adj_list');
assert(~isempty(gnn4.E), 'Should have edge features');

Y4 = gnn4.evaluate(td4.X);
assert(size(Y4, 1) == N4, 'Output should have N nodes');
assert(size(Y4, 2) == Fout4, 'Output should have F_out features');

%% 5) GINE-linear auto-detection (no model_type field)
rng(42);
N5 = 5; Fin5 = 4; Fout5 = 6; Ein5 = 3; nE5 = 8;

bp5 = struct();
bp5.mult1.Weights = rand(Fin5, Fout5);
bp5.edge1.Weights = rand(Ein5, Fout5);

src5 = randi(N5, nE5, 1); dst5 = randi(N5, nE5, 1);
E5 = rand(nE5, Ein5); a5 = rand(nE5, 1);
Xt5 = {rand(N5, Fin5)}; Yt5 = {rand(N5, Fout5)};

p5 = fullfile(tmp_dir, 'test_gine_auto.mat');
save_gnn_mat(p5, struct('best_params', bp5, 'src', src5, 'dst', dst5, ...
    'E_edge', E5, 'a', a5, 'X_test_g', {Xt5}, 'Y_test_g', {Yt5}));

[gnn5, ~] = gnn2nnv(p5);
assert(gnn5.numLayers == 1, 'Auto-detected GINE should have 1 layer');

%% 6) Test data extraction (GINE-linear)
rng(42);
N6 = 5; Fin6 = 4; Fout6 = 6; Ein6 = 3; nE6 = 8;

bp6 = struct();
bp6.mult1.Weights = rand(Fin6, Fout6);
bp6.edge1.Weights = rand(Ein6, Fout6);

src6 = randi(N6, nE6, 1); dst6 = randi(N6, nE6, 1);
E6 = rand(nE6, Ein6); a6 = rand(nE6, 1);
Xt6 = {rand(N6, Fin6)}; Yt6 = {rand(N6, Fout6)};

p6 = fullfile(tmp_dir, 'test_td.mat');
save_gnn_mat(p6, struct('best_params', bp6, 'src', src6, 'dst', dst6, ...
    'E_edge', E6, 'a', a6, 'X_test_g', {Xt6}, 'Y_test_g', {Yt6}, ...
    'model_type', 'gine_linear'));

[~, td6] = gnn2nnv(p6);
assert(isfield(td6, 'X'), 'test_data should have X field');
assert(isfield(td6, 'Y'), 'test_data should have Y field');
assert(isfield(td6, 'E'), 'test_data should have E field');
assert(isfield(td6, 'adj_list'), 'test_data should have adj_list field');

%% 7) Normalization stats extraction
rng(42);
bp7 = struct();
bp7.mult1.Weights = rand(F_in, F_h);
bp7.mult2.Weights = rand(F_h, F_out);
Xt7 = {rand(N, F_in)}; Yt7 = {rand(N, F_out)};
xm = rand(1, F_in); xs = rand(1, F_in);
ym = rand(1, F_out); ys = rand(1, F_out);

p7 = fullfile(tmp_dir, 'test_norm.mat');
save_gnn_mat(p7, struct('best_params', bp7, 'ANorm_g', ANorm_g, ...
    'X_test_g', {Xt7}, 'Y_test_g', {Yt7}, 'model_type', 'gcn', ...
    'X_mean', xm, 'X_std', xs, 'Y_mean', ym, 'Y_std', ys));

[~, ~, ns7] = gnn2nnv(p7);
assert(isfield(ns7, 'X_mean'), 'Should extract X_mean');
assert(isfield(ns7, 'X_std'), 'Should extract X_std');
assert(isfield(ns7, 'Y_mean'), 'Should extract Y_mean');
assert(isfield(ns7, 'Y_std'), 'Should extract Y_std');

%% 8) Error: missing file
try
    gnn2nnv('/nonexistent/path.mat');
    assert(false, 'Should have thrown an error');
catch ME
    assert(contains(ME.identifier, 'fileNotFound'), 'Should throw fileNotFound');
end

%% 9) Python prediction validation
rng(42);
bp9 = struct();
bp9.mult1.Weights = rand(F_in, F_h);
bp9.mult2.Weights = rand(F_h, F_out);
Xi9 = rand(N, F_in);
Xt9 = {Xi9}; Yt9 = {rand(N, F_out)};

% Build a temp GNN to get the correct predictions
temp_layers = {GCNLayer('t1', bp9.mult1.Weights, zeros(F_h,1)), ReluLayer(), ...
               GCNLayer('t2', bp9.mult2.Weights, zeros(F_out,1)), ReluLayer()};
temp_gnn = GNN(temp_layers, ANorm_g);
py_pred = temp_gnn.evaluate(Xi9);

p9 = fullfile(tmp_dir, 'test_validation.mat');
save_gnn_mat(p9, struct('best_params', bp9, 'ANorm_g', ANorm_g, ...
    'X_test_g', {Xt9}, 'Y_test_g', {Yt9}, 'model_type', 'gcn', ...
    'python_predictions', py_pred));

[gnn9, ~] = gnn2nnv(p9);
Y9 = gnn9.evaluate(Xi9);
assert(max(abs(Y9(:) - py_pred(:))) < 1e-10, ...
    'Imported GNN should match python predictions exactly');

%% 10) SAGEConv import test
rng(42);
bp10 = struct();
bp10.sage1.NodeWeights = rand(F_in, F_h);
bp10.sage1.EdgeWeights = rand(F_in, F_h);
bp10.sage1.Bias = rand(F_h, 1);
bp10.sage2.NodeWeights = rand(F_h, F_out);
bp10.sage2.EdgeWeights = rand(F_h, F_out);
bp10.sage2.Bias = rand(F_out, 1);

A_adj = double(A | A');
A_adj(1:N+1:end) = 0;  % No self-loops
Xt10 = {rand(N, F_in)};
Yt10 = {rand(N, F_out)};

p10 = fullfile(tmp_dir, 'test_sage.mat');
save_gnn_mat(p10, struct('best_params', bp10, 'A_adj', A_adj, ...
    'X_test_g', {Xt10}, 'Y_test_g', {Yt10}, 'model_type', 'sage'));

[gnn10, td10] = gnn2nnv(p10);
assert(isa(gnn10, 'GNN'), 'Should return GNN object');
assert(gnn10.numLayers == 4, 'SAGEConv with 2 conv + 2 ReLU = 4 layers');
assert(~isempty(gnn10.A_norm), 'Should have adjacency matrix in A_norm');

Y10 = gnn10.evaluate(td10.X);
assert(size(Y10, 1) == N, 'Output should have N nodes');
assert(size(Y10, 2) == F_out, 'Output should have F_out features');

% Verify test_data extraction
assert(isfield(td10, 'A_adj'), 'test_data should have A_adj field');
assert(isequal(size(td10.A_adj), [N, N]), 'A_adj should be NxN');

%% 11) SAGEConv cross-validation against manual GNN
rng(42);
bp11 = struct();
W_node1 = rand(F_in, F_h); W_edge1 = rand(F_in, F_h); b1 = rand(F_h, 1);
W_node2 = rand(F_h, F_out); W_edge2 = rand(F_h, F_out); b2 = rand(F_out, 1);
bp11.sage1.NodeWeights = W_node1; bp11.sage1.EdgeWeights = W_edge1; bp11.sage1.Bias = b1;
bp11.sage2.NodeWeights = W_node2; bp11.sage2.EdgeWeights = W_edge2; bp11.sage2.Bias = b2;

A_adj11 = double(A | A'); A_adj11(1:N+1:end) = 0;
Xi11 = rand(N, F_in);
Xt11 = {Xi11}; Yt11 = {rand(N, F_out)};

% Build manually
manual_layers = {SAGEConvLayer('s1', W_node1, W_edge1, b1), ReluLayer(), ...
                 SAGEConvLayer('s2', W_node2, W_edge2, b2), ReluLayer()};
manual_gnn = GNN(manual_layers, A_adj11);
manual_pred = manual_gnn.evaluate(Xi11);

p11 = fullfile(tmp_dir, 'test_sage_xval.mat');
save_gnn_mat(p11, struct('best_params', bp11, 'A_adj', A_adj11, ...
    'X_test_g', {Xt11}, 'Y_test_g', {Yt11}, 'model_type', 'sage', ...
    'python_predictions', manual_pred));

[gnn11, ~] = gnn2nnv(p11);
Y11 = gnn11.evaluate(Xi11);
assert(max(abs(Y11(:) - manual_pred(:))) < 1e-10, ...
    'Imported SAGEConv should match manual GNN predictions exactly');

%% Cleanup
if exist(tmp_dir, 'dir')
    rmdir(tmp_dir, 's');
end

disp('All gnn2nnv tests passed!');


%% Helper function to save .mat without variable name issues
function save_gnn_mat(filepath, s)
    % Save struct fields as individual variables in .mat file
    fn = fieldnames(s);
    for k = 1:length(fn)
        eval([fn{k} ' = s.(fn{k});']);
    end
    save(filepath, fn{:});
end
