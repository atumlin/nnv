%% run_edge_experiments.m
% Edge perturbation experiment suite for paper.
%
% Runs GINE Linear and GINE Conv with joint node+edge perturbation:
%   - Node epsilon: [0.001, 0.005, 0.01]
%   - Edge epsilon: 0.01 (default, configurable)
%   - Tasks: pf (default, configurable)
%   - Grids: ieee24, ieee39, ieee118
%   - Subgraph verification enabled (required for GINE Conv scalability)
%   - mode='node_edge' avoids re-running node-only results
%
% Run this AFTER run_all_experiments has completed node-only results.
%
% Usage:
%   run_edge_experiments()                         % full run (all tasks)
%   run_edge_experiments('task', 'pf')             % PF only
%   run_edge_experiments('num_graphs', 2)          % smoke test
%
% Author: Anne Tumlin
% Date: 03/11/2026

function run_edge_experiments(varargin)

p = inputParser;
addParameter(p, 'task',         'pf',  @ischar);
addParameter(p, 'num_graphs',   100,   @isnumeric);
addParameter(p, 'grid',         'all', @ischar);
addParameter(p, 'epsilon_edge', 0.01,  @isnumeric);
parse(p, varargin{:});

task       = p.Results.task;
num_graphs = p.Results.num_graphs;
grid       = p.Results.grid;
eps_edge   = p.Results.epsilon_edge;

fprintf('=== Edge Perturbation Experiments ===\n');
fprintf('  Architectures: gine_linear, gine_conv\n');
fprintf('  Node eps: [0.001, 0.005, 0.01]  |  Edge eps: %.3f\n', eps_edge);
fprintf('  Mode: node_edge (joint perturbation)\n');
fprintf('  Subgraph: true\n\n');

base_opts = {'num_graphs', num_graphs, 'mode', 'node_edge', 'subgraph', true, ...
             'skip_edge', false, 'grid', grid, 'parallel', true, ...
             'epsilon_edge', eps_edge};

tasks = {};
if strcmp(task, 'all') || strcmp(task, 'pf')
    tasks{end+1} = 'pf';
end
if strcmp(task, 'all') || strcmp(task, 'opf')
    tasks{end+1} = 'opf';
end

for ti = 1:length(tasks)
    t = tasks{ti};
    fprintf('--- Task: %s ---\n', upper(t));

    fprintf('  Running gine_linear...\n');
    run_all_experiments(base_opts{:}, 'task', t, 'arch', 'gine_linear');

    fprintf('  Running gine_conv...\n');
    run_all_experiments(base_opts{:}, 'task', t, 'arch', 'gine_conv');
end

fprintf('\n=== Edge Perturbation Experiments Complete ===\n');

end
