%% ================== MRAE ==================
function [MRAE, MRAE_hat] = calc_MRAE(y, Q_hat, alpha_set)

    y = y(:);
    alpha_set = alpha_set(:)';

    [N, K] = size(Q_hat);

    assert(K == numel(alpha_set), ...
        'The number of columns in Q_hat must match the length of alpha_set.');

    assert(N == numel(y), ...
        'The number of rows in Q_hat must match the length of y.');

    eta = double(y < Q_hat);

    alpha_hat = mean(eta, 1);

    MRAE_hat = alpha_set - alpha_hat;
    MRAE = mean(abs(MRAE_hat));
end