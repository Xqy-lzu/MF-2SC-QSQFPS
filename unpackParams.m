function [beta_0, beta_2k, sigma, gamma] = unpackParams(Yb, k)

    beta_0  = Yb(:,1);
    beta_2k = Yb(:,2:3);
    sigma   = Yb(:,4:3+k);
    gamma   = Yb(:,k+4:2*k+3);

    beta_2k = max(beta_2k, 0);
    sigma   = softmax_dim(sigma, 2);
    gamma   = softplus_stable(gamma);
end

