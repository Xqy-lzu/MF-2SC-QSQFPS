%% ================== Point forecasting metrics ==================
function P = point_metrics_from_Q(Qhat, y)
% Point forecasting metrics: MAE and MAPE.
% The median quantile is used as the point forecast.

    y = y(:);

    K = size(Qhat, 2);

    k50 = round(0.50 * K);
    k50 = max(1, min(K, k50));

    yhat = Qhat(:, k50);

    err = yhat - y;

    mae = mean(abs(err));

    den = mean(abs(y));
    if den <= 0
        den = 1;
    end

    mape = 100 * mean(abs(err)) / den;

    P = struct( ...
        'MAE', mae, ...
        'MAPE', mape, ...
        'Yhat', yhat);
end
