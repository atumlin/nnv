% Check GINE dimensions
gine_model = load('models/gine_test.mat');
params = gine_model.best_params;
fprintf('W1: %s\n', mat2str(size(double(extractdata(gather(params.mult1.Weights))))));
fprintf('W2: %s\n', mat2str(size(double(extractdata(gather(params.mult2.Weights))))));
fprintf('W3: %s\n', mat2str(size(double(extractdata(gather(params.mult3.Weights))))));
fprintf('W_e1: %s\n', mat2str(size(double(extractdata(gather(params.edge1.Weights))))));
fprintf('W_e2: %s\n', mat2str(size(double(extractdata(gather(params.edge2.Weights))))));
fprintf('W_e3: %s\n', mat2str(size(double(extractdata(gather(params.edge3.Weights))))));
fprintf('X_test: %s\n', mat2str(size(double(gine_model.X_test_g{1}))));
fprintf('E: %s\n', mat2str(size(double(gine_model.E_edge))));
