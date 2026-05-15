%% ================== Interval forecasting metrics ==================
function T = interval_metrics_from_Q(Qhat, y, pinc_set)
% Interval forecasting metrics: PICP, PINAW, and AIS.

    y = y(:);

    n = size(Qhat, 1);
    K = size(Qhat, 2);

    assert(numel(y) == n, ...
        'interval_metrics_from_Q: the length of y must match the number of rows in Qhat.');

    y_range = max(y) - min(y);

    if y_range <= 0
        y_range = 1;
    end

    T = table();

    for c = pinc_set(:)'

        alpha_s = 1 - c;

        lower_prob = alpha_s / 2;
        upper_prob = 1 - lower_prob;

        lower_idx = round(lower_prob * K);
        lower_idx = max(1, min(K, lower_idx));

        upper_idx = round(upper_prob * K);
        upper_idx = max(1, min(K, upper_idx));

        L = Qhat(:, lower_idx);
        U = Qhat(:, upper_idx);

        cover = (y >= L) & (y <= U);

        PICP  = mean(cover);
        PINAW = mean((U - L) / y_range);

        under = (y < L);
        over  = (y > U);
        width = U - L;

        AIS_each = -2 * alpha_s .* width ...
                   - 4 * ((L - y) .* under + (y - U) .* over);

        AIS = mean(AIS_each);

        tag = num2str(round(c * 100));

        T.(['PICP_'  tag]) = PICP;
        T.(['PINAW_' tag]) = PINAW;
        T.(['AIS_'   tag]) = AIS;
    end
end