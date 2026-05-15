function Qmono = pav_rearrange_nondec(Q)

    [n, K] = size(Q);

    Qmono = zeros(n, K);

    for i = 1:n
        Qmono(i, :) = pav_1d_nondec(Q(i, :));
    end
end