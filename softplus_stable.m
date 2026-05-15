
function y = softplus_stable(x)
    u = -abs(x);
    y = max(x,0) + log(1 + exp(u));
end
