classdef AddPoolLayer < handle
    % The AddPoolLayer class for graph-level sum pooling
    %   Performs: g = sum(X, 1)
    %   Where:
    %     X: Node features (N x F)
    %     g: Graph-level embedding (1 x F), sum of all node features
    %
    %   This reduces a graph of N nodes to a single graph-level vector
    %   by summing all node feature vectors.
    %
    %   For reachability: since sum is linear, each GraphStar generator
    %   is summed independently. The output is still a GraphStar with
    %   1 node.
    %
    % Author: Anne Tumlin
    % Date: 02/12/2026

    properties
        Name = 'add_pool';
        InputSize = 0;    % F: feature dimension (set dynamically)
        OutputSize = 0;   % Same as InputSize (features preserved, nodes collapsed)

        % Standard layer interface
        NumInputs = 1;
        InputNames = {'in'};
        NumOutputs = 1;
        OutputNames = {'out'};
    end


    methods % main methods

        function obj = AddPoolLayer(varargin)
            % AddPoolLayer constructor
            % Usage:
            %   AddPoolLayer()
            %   AddPoolLayer(name)

            if nargin >= 1
                obj.Name = varargin{1};
            end
        end


        function Y = evaluate(obj, X)
            % Forward pass: sum all node features
            % @X: Input node features (N x F)
            % @Y: Graph embedding (1 x F)

            Y = sum(X, 1);  % [1 x F]
        end


        function S = reach(varargin)
            % Reachability analysis for AddPoolLayer

            switch nargin
                case 7
                    obj = varargin{1};
                    in_sets = varargin{2};
                case 6
                    obj = varargin{1};
                    in_sets = varargin{2};
                case 5
                    obj = varargin{1};
                    in_sets = varargin{2};
                case 4
                    obj = varargin{1};
                    in_sets = varargin{2};
                case 3
                    obj = varargin{1};
                    in_sets = varargin{2};
                case 2
                    obj = varargin{1};
                    in_sets = varargin{2};
                otherwise
                    error('Invalid number of input arguments');
            end

            n = length(in_sets);
            if n == 1
                S = obj.reach_star_single_input(in_sets);
            else
                S(n) = GraphStar;
                for i = 1:n
                    S(i) = obj.reach_star_single_input(in_sets(i));
                end
            end
        end

    end


    methods % reachability methods

        function gs_out = reach_star_single_input(obj, in_gs)
            % Reachability through AddPool for single GraphStar
            % @in_gs: Input GraphStar set (N nodes x F features)
            % @gs_out: Output GraphStar set (1 node x F features)
            %
            % Sum pooling is linear, so reachability is exact.

            if ~isa(in_gs, 'GraphStar')
                error('Input must be a GraphStar');
            end

            numFeatures = in_gs.numFeatures;
            numPred = in_gs.numPred;

            % Sum each generator across nodes: (N x F x K) -> (1 x F x K)
            V_out = zeros(1, numFeatures, numPred + 1, 'like', in_gs.V);
            for k = 1:(numPred + 1)
                V_out(1, :, k) = sum(in_gs.V(:, :, k), 1);
            end

            % Constraints unchanged (linear transformation)
            gs_out = GraphStar(V_out, in_gs.C, in_gs.d, in_gs.pred_lb, in_gs.pred_ub);
        end

    end

end
