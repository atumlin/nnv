function [gnn, graph_data] = gnn_from_scipmpnn(gnn_path, gcinfo_path)
% GNN_FROM_SCIPMPNN  Import a GNN from SCIP-MPNN text format files
%
%   [gnn, graph_data] = gnn_from_scipmpnn(gnn_path, gcinfo_path)
%
%   Inputs:
%     gnn_path    - Path to .gnn model file (SCIP-MPNN format)
%     gcinfo_path - (Optional) Path to .gcinfo graph instance file
%
%   Outputs:
%     gnn        - GNN object with SAGEConvLayer, ReluLayer, AddPoolLayer,
%                  FullyConnectedLayer
%     graph_data - struct with fields (only if gcinfo_path provided):
%                    .X          - Node features (N x F)
%                    .A          - Binary adjacency matrix (N x N)
%                    .n_nodes    - Number of nodes
%                    .n_classes  - Number of classes
%                    .true_label - True class label (0-indexed)
%
%   The .gnn file uses SCIP-MPNN's text format with layer types:
%     input, sage, addpool, dense
%   Weights are stored in (F_out x F_in) order and transposed to NNV's
%   (F_in x F_out) convention.
%
%   The .gcinfo file contains graph structure and one-hot node features
%   for graph classification benchmarks (MUTAG, ENZYMES).
%
% Author: Anne Tumlin
% Date: 02/12/2026

    %% Parse .gnn model file
    fid = fopen(gnn_path, 'r');
    if fid == -1
        error('Cannot open GNN file: %s', gnn_path);
    end

    n_layers_total = str2double(fgetl(fid));
    fprintf('gnn_from_scipmpnn: Reading %d layers from %s\n', n_layers_total, gnn_path);

    layers = {};
    n_input_features = 0;

    for i = 1:n_layers_total
        layer_type = strtrim(fgetl(fid));

        switch layer_type
            case 'input'
                n_input_features = str2double(fgetl(fid));
                fprintf('  Input: %d features\n', n_input_features);

            case 'sage'
                [sage_layer, activation] = read_sage_layer(fid);
                layers{end+1} = sage_layer; %#ok<AGROW>
                fprintf('  SAGEConvLayer: %d -> %d', sage_layer.InputSize, sage_layer.OutputSize);
                if strcmp(activation, 'relu')
                    layers{end+1} = ReluLayer(); %#ok<AGROW>
                    fprintf(' + ReLU');
                end
                fprintf('\n');

            case 'addpool'
                layers{end+1} = AddPoolLayer(); %#ok<AGROW>
                fprintf('  AddPoolLayer\n');

            case 'dense'
                [dense_layer, activation] = read_dense_layer(fid);
                layers{end+1} = dense_layer; %#ok<AGROW>
                fprintf('  FullyConnectedLayer: %d -> %d', ...
                    size(dense_layer.Weights, 1), size(dense_layer.Weights, 2));
                if strcmp(activation, 'relu')
                    layers{end+1} = ReluLayer(); %#ok<AGROW>
                    fprintf(' + ReLU');
                end
                fprintf('\n');

            otherwise
                fclose(fid);
                error('Unknown layer type: %s', layer_type);
        end
    end

    fclose(fid);

    fprintf('gnn_from_scipmpnn: Built GNN with %d layers\n', length(layers));

    %% Parse .gcinfo file (if provided)
    A = [];
    graph_data = struct();

    if nargin >= 2 && ~isempty(gcinfo_path)
        [X, A, n_nodes, n_features, n_classes, true_label] = read_gcinfo(gcinfo_path);
        graph_data.X = X;
        graph_data.A = A;
        graph_data.n_nodes = n_nodes;
        graph_data.n_features = n_features;
        graph_data.n_classes = n_classes;
        graph_data.true_label = true_label;
        fprintf('gnn_from_scipmpnn: Loaded graph with %d nodes, %d features, %d classes (label=%d)\n', ...
            n_nodes, n_features, n_classes, true_label);
    end

    %% Construct GNN object
    gnn = GNN(layers, A);

end


function [layer, activation] = read_sage_layer(fid)
% Read a SAGE layer from the .gnn file
    header = strsplit(strtrim(fgetl(fid)));
    n_in = str2double(header{1});
    n_out = str2double(header{2});
    activation = header{3};

    % Read nodeweights (F_out x F_in) -> transpose to (F_in x F_out)
    keyword = strtrim(fgetl(fid));  % 'nodeweights'
    assert(strcmp(keyword, 'nodeweights'), 'Expected "nodeweights", got "%s"', keyword);
    nodeweights = read_matrix(fid, n_out, n_in);  % [F_out x F_in]
    NodeWeights = nodeweights';                    % [F_in x F_out]

    % Read edgeweights (F_out x F_in) -> transpose to (F_in x F_out)
    keyword = strtrim(fgetl(fid));  % 'edgeweights'
    assert(strcmp(keyword, 'edgeweights'), 'Expected "edgeweights", got "%s"', keyword);
    edgeweights = read_matrix(fid, n_out, n_in);  % [F_out x F_in]
    EdgeWeights = edgeweights';                    % [F_in x F_out]

    % Read bias (F_out x 1)
    keyword = strtrim(fgetl(fid));  % 'bias'
    assert(strcmp(keyword, 'bias'), 'Expected "bias", got "%s"', keyword);
    bias = zeros(n_out, 1);
    for j = 1:n_out
        bias(j) = str2double(fgetl(fid));
    end

    layer = SAGEConvLayer(NodeWeights, EdgeWeights, bias);
end


function [layer, activation] = read_dense_layer(fid)
% Read a Dense layer from the .gnn file
    header = strsplit(strtrim(fgetl(fid)));
    n_in = str2double(header{1});
    n_out = str2double(header{2});
    activation = header{3};

    % Read denseweights (F_out x F_in) — FullyConnectedLayer uses same convention
    keyword = strtrim(fgetl(fid));  % 'denseweights'
    assert(strcmp(keyword, 'denseweights'), 'Expected "denseweights", got "%s"', keyword);
    W = read_matrix(fid, n_out, n_in);  % [F_out x F_in]

    % Read bias (F_out x 1)
    keyword = strtrim(fgetl(fid));  % 'bias'
    assert(strcmp(keyword, 'bias'), 'Expected "bias", got "%s"', keyword);
    bias = zeros(n_out, 1);
    for j = 1:n_out
        bias(j) = str2double(fgetl(fid));
    end

    % Use FullyConnectedLayer (standard NNV layer for dense/affine)
    layer = FullyConnectedLayer(W, bias);
end


function M = read_matrix(fid, n_rows, n_cols)
% Read a matrix row by row from the file
    M = zeros(n_rows, n_cols);
    for r = 1:n_rows
        vals = str2num(fgetl(fid)); %#ok<ST2NM>
        M(r, :) = vals;
    end
end


function [X, A, n_nodes, n_features, n_classes, true_label] = read_gcinfo(filepath)
% Read a .gcinfo graph classification instance file
%
% Format:
%   Line 1: n_nodes n_edges n_features n_classes true_label
%   Line 2: edge source indices (0-indexed, space-separated)
%   Line 3: edge destination indices (0-indexed, space-separated)
%   Lines 4+: feature indices per node (one line per node)

    fid = fopen(filepath, 'r');
    if fid == -1
        error('Cannot open gcinfo file: %s', filepath);
    end

    % Line 1: header
    header = str2num(fgetl(fid)); %#ok<ST2NM>
    n_nodes = header(1);
    n_edges = header(2);
    n_features = header(3);
    n_classes = header(4);
    true_label = header(5);

    % Line 2-3: edges (0-indexed)
    src = str2num(fgetl(fid)); %#ok<ST2NM>
    dst = str2num(fgetl(fid)); %#ok<ST2NM>

    % Build binary adjacency matrix (0-indexed -> 1-indexed)
    A = zeros(n_nodes, n_nodes);
    for e = 1:n_edges
        A(src(e)+1, dst(e)+1) = 1;
    end

    % Lines 4+: node features (one-hot from feature indices)
    X = zeros(n_nodes, n_features);
    for v = 1:n_nodes
        line = strtrim(fgetl(fid));
        if isempty(line)
            continue;
        end
        feat_indices = str2num(line); %#ok<ST2NM>
        for fi = 1:length(feat_indices)
            X(v, feat_indices(fi) + 1) = 1;  % 0-indexed -> 1-indexed
        end
    end

    fclose(fid);
end
