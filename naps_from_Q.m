%% ================== NAPS ==================
function naps_value = naps_from_Q(Qhat, alpha, omega_set)

    alpha = alpha(:)';
    omega_set = omega_set(:)';

    [n, K] = size(Qhat);

    assert(K == numel(alpha), ...
        'The number of columns in Qhat must match the length of alpha.');

    idx = @(p) max(1, min(K, round(p * K)));

    vals = zeros(n, 1);

    for i = 1:n

        acc = 0;

        for omega = omega_set

            up = idx(1 - omega/2);
            lo = idx(omega/2);

            acc = acc + (Qhat(i, up) - Qhat(i, lo)) / (1 - omega);
        end

        vals(i) = acc / numel(omega_set);
    end

    naps_value = mean(vals);
end