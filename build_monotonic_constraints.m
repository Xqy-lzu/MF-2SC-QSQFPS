%% ================== Monotonicity constraints ==================
function [Amono, bmono] = build_monotonic_constraints(Q_train, use_full)

    [n_train, K, nModels] = size(Q_train);

    if use_full

        nRows = n_train * (K - 1);

        Amono = spalloc(nRows, K*nModels, nRows * 2 * nModels);
        bmono = zeros(nRows, 1);

        row = 0;

        for i = 1:n_train
            for k = 1:K-1
                row = row + 1;

                for m = 1:nModels
                    Amono(row, (m-1)*K + k)     =  Q_train(i, k, m);
                    Amono(row, (m-1)*K + k + 1) = -Q_train(i, k+1, m);
                end
            end
        end

    else

        Qmax = squeeze(max(Q_train(:, 1:K-1, :), [], 1));   % (K-1) x nModels
        Qmin = squeeze(min(Q_train(:, 2:K,   :), [], 1));   % (K-1) x nModels

        Amono = spalloc(K-1, K*nModels, (K-1) * 2 * nModels);
        bmono = zeros(K-1, 1);

        for k = 1:K-1
            for m = 1:nModels
                Amono(k, (m-1)*K + k)     =  Qmax(k, m);
                Amono(k, (m-1)*K + k + 1) = -Qmin(k, m);
            end
        end
    end
end
