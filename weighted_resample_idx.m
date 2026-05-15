function idx = weighted_resample_idx(N, w)
% Weighted resampling.

    w = double(w(:)');
    w = max(w,0);

    s = sum(w);

    if s == 0
        w(:) = 1/numel(w);
    else
        w = w / s;
    end

    cdf = cumsum(w);
    u = rand(1,N);

    idx = arrayfun(@(x) find(cdf >= x, 1, 'first'), u);
end

