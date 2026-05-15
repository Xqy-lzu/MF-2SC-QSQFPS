
function S = softmax_dim(X, dim)
    Xmax = max(X, [], dim);
    E = exp(X - Xmax);
    S = E ./ sum(E, dim);
end


