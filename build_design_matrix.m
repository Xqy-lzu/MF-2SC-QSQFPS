%% ================== Build design matrix ==================
function R = build_design_matrix(Q_train)
% Construct the linear mapping from vectorized weights to combined quantiles.

    [n_train, K, nModels] = size(Q_train);
    nRows = n_train * K;
    nWeights = K * nModels;

    R = spalloc(nRows, nWeights, nRows * nModels);

    row = 0;

    for i = 1:n_train
        for k = 1:K
            row = row + 1;

            for m = 1:nModels
                col = (m-1)*K + k;
                R(row, col) = Q_train(i, k, m);
            end
        end
    end
end
