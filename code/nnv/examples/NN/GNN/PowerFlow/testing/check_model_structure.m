% Check model structure
m = load('/home/verivital/Anne/gnnv/pf/model_training/training_results/2026-01-20_11-30-10/gcn/ieee24_gcn_pf_seed128_summary.mat');
s = m.multi_run_summary;
r = s.all_runs{1};
f = fieldnames(r);
fprintf('Fields in all_runs{1}:\n');
for i = 1:length(f)
    fprintf('  %s: ', f{i});
    v = r.(f{i});
    if isstruct(v)
        fprintf('[struct with %d fields]\n', length(fieldnames(v)));
    elseif iscell(v)
        fprintf('[cell %dx%d]\n', size(v,1), size(v,2));
    elseif isnumeric(v)
        fprintf('[%s %dx%d]\n', class(v), size(v,1), size(v,2));
    else
        fprintf('[%s]\n', class(v));
    end
end

% Check if hyperparams has the model weights
fprintf('\nFields in hyperparams:\n');
h = r.hyperparams;
disp(fieldnames(h));

% Check architecture
fprintf('\nFields in architecture:\n');
a = r.architecture;
disp(fieldnames(a));
