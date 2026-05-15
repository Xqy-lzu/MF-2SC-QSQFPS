%% ================== Linear combination ==================
function Qens = combine_W(Q, W)
% Combine base quantile forecasts using quantile-level weights.

    [~, K, nModels] = size(Q);

    assert(isequal(size(W), [K, nModels]), ...
        'The weight matrix W must have size K x nModels.');

    W3 = reshape(W, [1, K, nModels]);
    Qens = sum(Q .* W3, 3);
end