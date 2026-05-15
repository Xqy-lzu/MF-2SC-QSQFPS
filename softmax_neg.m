function w = softmax_neg(v, temp)

    if nargin < 2 || isempty(temp)
        temp = 1;
    end

    z = -double(v) / temp;
    z = z - max(z);

    w = exp(z);
    w = w / sum(w);

    w = single(w);
end