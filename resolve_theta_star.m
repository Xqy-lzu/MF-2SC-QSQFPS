%% ================== Resolve theta_star ==================
function theta_star = resolve_theta_star( ...
    theta_star, Q_train, y_train, alpha, gamma_theta, nModels, verbose)
% Determine the relative objective weights for CRPS and AQS.

    K = numel(alpha);

    if (ischar(theta_star) && strcmpi(theta_star, 'auto')) || ...
       (isstring(theta_star) && strcmpi(theta_star, "auto"))

        W_uniform = ones(K, nModels) / nModels;
        Q_uniform = combine_W(Q_train, W_uniform);

        [CRPS0, AQS0] = crps_aqs_from_Q(Q_uniform, y_train, alpha);

        scale_values = max([CRPS0, AQS0], 1e-12);
        inv_scale = scale_values .^ (-gamma_theta);

        theta_star = inv_scale / sum(inv_scale);

        if verbose
            fprintf('[theta-auto] gamma = %.2f -> theta = [%.3f %.3f]\n', ...
                gamma_theta, theta_star(1), theta_star(2));
        end

    else

        theta_star = theta_star(:)';

        % Backward compatibility: if a previous three-objective vector is provided,
        % only the first two weights are used.
        if numel(theta_star) == 3
            theta_star = theta_star(1:2);
        end

        if numel(theta_star) ~= 2
            error('theta_star must be [theta_CRPS, theta_AQS] or ''auto''.');
        end

        theta_star = max(theta_star, 0);
        theta_star = theta_star / max(sum(theta_star), 1e-12);
    end
end