function Q = Qpredict_batch(beta_0, beta_2k, sigma, gamma, alpha)

    B = size(beta_0,1);
    M = numel(alpha);
    alpha = reshape(alpha, 1, M);

    beta_1 = gamma(:,1) - 2 * beta_2k(:,1) .* sigma(:,1);
    beta_N = [beta_0, beta_1];

    beta = (gamma - [zeros(B,1,'like',gamma), gamma(:,1:end-1)]) ./ (2*sigma);
    beta(:,1) = beta_2k(:,1);
    beta = beta - [zeros(B,1,'like',beta), beta(:,1:end-1)];
    beta(:,end) = beta_2k(:,2) - sum(beta(:,1:end-1),2);

    ksi = cumsum(sigma,2);
    ksi_pad = [zeros(B,1,'like',ksi), ksi(:,1:end-1)];

    Q = zeros(B, M, 'like', beta_0);

    for m = 1:M
        a = alpha(m) * ones(B,1,'like',beta_0);
        d = max(0, a - ksi_pad);
        Q(:,m) = beta_N(:,1) + beta_N(:,2) .* a + sum((d.^2) .* beta, 2);
    end
end

