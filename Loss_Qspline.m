
function [loss, gradients] = Loss_Qspline(net, Xcbt, Tcb, k)

    Ycb = forward(net, Xcbt);
    Ycb = stripdims(Ycb);
    Yb  = permute(Ycb, [2 1]);

    [beta_0, beta_2k, sigma, gamma] = unpackParams(Yb, k);

    Tb = stripdims(Tcb);
    T  = permute(Tb, [2 1]);

    beta_1 = gamma(:,1) - 2 * beta_2k(:,1) .* sigma(:,1);
    beta_N = [beta_0, beta_1];

    beta = (gamma - [zeros(size(gamma,1),1,'like',gamma), gamma(:,1:end-1)]) ./ (2*sigma);
    beta(:,1) = beta_2k(:,1);
    beta = beta - [zeros(size(beta,1),1,'like',beta), beta(:,1:end-1)];
    beta(:,end) = beta_2k(:,2) - sum(beta(:,1:end-1),2);

    k_ = size(sigma,2);

    if isa(extractdata(sigma),'single')
        U = triu(ones(k_, 'single'));
    else
        U = triu(ones(k_, 'double'));
    end

    ksi = sigma * U;
    ksi_pad = [zeros(size(ksi,1),1,'like',ksi), ksi(:,1:end-1)];

    Bsz = size(T,1);
    Qn = zeros(Bsz, k_, 'like', T);

    for n = 1:k_
        a = ksi(:,n);
        q_lin = beta_N(:,1) + beta_N(:,2) .* a;
        d = max(0, a - ksi_pad);
        q_quad = sum((d.^2) .* beta, 2);
        Qn(:,n) = q_lin + q_quad;
    end

    Qn_pad = [zeros(Bsz,1,'like',Qn), Qn(:,1:end-1)];

    diff = T - Qn_pad;
    alpha_l = diff > 0;

    alpha_A = sum(alpha_l .* beta, 2);
    alpha_B = beta_N(:,2) - 2 * sum(alpha_l .* beta .* ksi_pad, 2);
    alpha_C = beta_N(:,1) - T + sum(alpha_l .* beta .* (ksi_pad.^2), 2);

    alpha = zeros(Bsz,1,'like',T);
    nzA = (alpha_A ~= 0);
    disc = max(alpha_B.^2 - 4 * alpha_A .* alpha_C, 0);

    alpha(nzA)  = (-alpha_B(nzA) + sqrt(disc(nzA))) ./ (2 * alpha_A(nzA));
    alpha(~nzA) = -alpha_C(~nzA) ./ max(alpha_B(~nzA), eps('like', alpha_B));
    alpha = min(max(alpha,0),1);

    crps_1 = T .* (2*alpha - 1);
    crps_2 = beta_N(:,1) .* (1 - 2*alpha) + beta_N(:,2) .* (1/3 - alpha.^2);
    crps_3 = sum((2 * beta ./ 12) .* (1 - ksi_pad).^4, 2);
    crps_4 = sum((alpha_l .* 2 .* beta ./ 3) .* (alpha - ksi_pad).^3, 2);

    crps = crps_1 + crps_2 + crps_3 - crps_4;
    loss = mean(crps);

    gradients = dlgradient(loss, net.Learnables);
end