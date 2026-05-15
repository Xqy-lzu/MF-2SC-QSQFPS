function CRPS_each = crps_each_from_Q(Qhat, y, alpha)
% Sample-wise CRPS.

    y = y(:);
    alpha = alpha(:)';

    n = size(Qhat,1);
    K = numel(alpha);

    assert(size(Qhat,2) == K, 'The number of columns in Qhat must match the length of alpha.');
    assert(numel(y) == n, 'The length of y must match the number of rows in Qhat.');

    Y = repmat(y, 1, K);
    A = repmat(alpha, n, 1);

    I = double(Y <= Qhat);
    L = (A - I) .* (Y - Qhat);

    CRPS_each = trapz(alpha, 2*L, 2);
end