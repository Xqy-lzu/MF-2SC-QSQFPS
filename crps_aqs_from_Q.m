%% ================== Probabilistic metrics: CRPS and AQS ==================
function [CRPS_mean, AQS_mean] = crps_aqs_from_Q(Qhat, y, alpha)

    y = y(:);
    alpha = alpha(:)';

    n = size(Qhat, 1);
    K = numel(alpha);

    assert(size(Qhat, 2) == K, ...
        'The number of columns in Qhat must match the length of alpha.');

    assert(numel(y) == n, ...
        'crps_aqs_from_Q: the length of y must match the number of rows in Qhat.');

    Y = repmat(y, 1, K);
    A = repmat(alpha, n, 1);

    I = double(Y <= Qhat);
    L = (A - I) .* (Y - Qhat);

    AQS_mean = mean(L(:));

    CRPS_each = trapz(alpha, 2 * L, 2);
    CRPS_mean = mean(CRPS_each);
end