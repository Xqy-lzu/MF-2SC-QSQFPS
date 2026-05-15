%% ================== Solve one QP and return W ==================
function W_opt = solve_qp_get_W(Q_train, y_train, theta, mu, use_full_monotonic, ...
                                allow_neg, boxB, qp_display)
% Two-objective QP:
%   CRPS + AQS
%
% Decision variables:
%   z = [vec(W); t_plus; t_minus]

    [n_train, K, nModels] = size(Q_train);

    alpha = (1:K) / 100;

    nWeights = K * nModels;
    nLossTerms = n_train * K;
    nVars = nWeights + 2 * nLossTerms;

    %% ---------- Quadratic term: L2 smoothness along quantile levels ----------
    H = sparse(nVars, nVars);

    if mu > 0

        D = spdiags([-ones(K-1,1), ones(K-1,1)], [0, 1], K-1, K);
        H_W = sparse(nWeights, nWeights);

        for m = 1:nModels
            rows = (m-1)*K + (1:K);
            H_W(rows, rows) = H_W(rows, rows) + 2 * mu * (D' * D);
        end

        H(1:nWeights, 1:nWeights) = H_W;
    end

    H = (H + H') / 2;
    H = H + 1e-10 * speye(size(H, 1));

    %% ---------- Linear term: CRPS + AQS ----------
    f = zeros(nVars, 1);

    trap_weights = [0.5, ones(1, K-2), 0.5] * 0.01;

    c_crps_plus  = (2 * trap_weights) .* alpha;
    c_crps_minus = (2 * trap_weights) .* (1 - alpha);

    c_aqs_plus   = (1 / K) * alpha;
    c_aqs_minus  = (1 / K) * (1 - alpha);

    c_plus  = theta(1) * c_crps_plus  + theta(2) * c_aqs_plus;
    c_minus = theta(1) * c_crps_minus + theta(2) * c_aqs_minus;

    f(nWeights+1 : nWeights+nLossTerms) = repmat(c_plus(:), n_train, 1);
    f(nWeights+nLossTerms+1 : nVars)    = repmat(c_minus(:), n_train, 1);

    %% ---------- Linear constraints for pinball loss ----------
    R = build_design_matrix(Q_train);

    y_train = y_train(:);
    y_rep = kron(y_train, ones(K, 1));

    A1 = [-R, -speye(nLossTerms), sparse(nLossTerms, nLossTerms)];
    b1 = -y_rep;

    A2 = [ R,  sparse(nLossTerms, nLossTerms), -speye(nLossTerms)];
    b2 = y_rep;

    %% ---------- Equality constraint: row-wise weight sum equals 1 ----------
    Aeq_W = spalloc(K, nWeights, K * nModels);

    for k = 1:K
        Aeq_W(k, k:K:nWeights) = 1;
    end

    Aeq = [Aeq_W, sparse(K, 2*nLossTerms)];
    beq = ones(K, 1);

    %% ---------- Monotonicity constraints ----------
    [Amono, bmono] = build_monotonic_constraints(Q_train, use_full_monotonic);

    Aineq = [A1; A2; [Amono, sparse(size(Amono,1), 2*nLossTerms)]];
    bineq = [b1; b2; bmono];

    %% ---------- Variable bounds ----------
    lb = -inf(nVars, 1);
    ub =  inf(nVars, 1);

    if allow_neg
        lb(1:nWeights) = -boxB;
        ub(1:nWeights) =  boxB;
    else
        lb(1:nWeights) = 0;
    end

    % t_plus and t_minus must be nonnegative.
    lb(nWeights+1:nVars) = 0;

    %% ---------- QP solver ----------
    qp_opts = optimoptions('quadprog', ...
        'Display', qp_display, ...
        'MaxIterations', 300, ...
        'ConstraintTolerance', 1e-8, ...
        'OptimalityTolerance', 1e-8, ...
        'StepTolerance', 1e-10);

    [z_opt, ~, exitflag] = quadprog( ...
        H, f, Aineq, bineq, Aeq, beq, lb, ub, [], qp_opts);

    assert(exitflag > 0, ...
        'quadprog did not converge. Try reducing mu, using sufficient monotonic constraints, or checking for NaN values.');

    W_opt = reshape(z_opt(1:nWeights), [K, nModels]);
end
