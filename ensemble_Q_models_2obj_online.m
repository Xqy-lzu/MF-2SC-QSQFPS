function [W_stream, Q_pred_test, metrics_train, metrics_test, baseline_test] = ...
    ensemble_Q_models_2obj_online(M1, M2, M3, y, train_ratio, win, ...
                                     theta_star, mu, use_full_monotonic, ...
                                     omega_set, varargin)
% Online heterogeneous ensemble with dynamic quantile-level weights.
%
% Objective:
%   CRPS + AQS, solved by rolling-window quadratic programming.
%
% Outputs:
%   W_stream      : cell array of dynamic weights for each test sample.
%                   Each cell is a K x 3 weight matrix.
%   Q_pred_test   : final ensemble quantile forecasts, n_test x K.
%   me_tr : reference metrics on the training subset.
%   me_te  : final ensemble metrics on the test subset.
%   base_te : metrics of the three base models on the aligned test subset.

    %% ---------- Default arguments ----------
    if nargin < 5 || isempty(train_ratio)
        train_ratio = 0.80;
    end

    if nargin < 6 || isempty(win)
        win = 96;
    end

    if nargin < 7 || isempty(theta_star)
        theta_star = 'auto';
    end

    if nargin < 8 || isempty(mu)
        mu = 1e-2;
    end

    if nargin < 9 || isempty(use_full_monotonic)
        use_full_monotonic = false;
    end

    if nargin < 10 || isempty(omega_set)
        omega_set = 0.05:0.05:0.95;
    end

    if train_ratio > 1
        train_ratio = train_ratio / 100;
    end

    train_ratio = min(max(train_ratio, 0), 1);

    %% ---------- Optional parameters ----------
    parser = inputParser;
    addParameter(parser, 'ThetaAutoGamma', 1.0);
    addParameter(parser, 'ProjectTestPAV', true);
    addParameter(parser, 'Verbose', false);
    addParameter(parser, 'AllowNegativeWeights', true);
    addParameter(parser, 'WeightBoxB', 1);
    addParameter(parser, 'QPDisplay', 'off');
    parse(parser, varargin{:});

    gamma_theta      = parser.Results.ThetaAutoGamma;
    project_test_pav = parser.Results.ProjectTestPAV;
    verbose          = parser.Results.Verbose;
    allow_neg        = parser.Results.AllowNegativeWeights;
    boxB             = parser.Results.WeightBoxB;
    qp_display       = parser.Results.QPDisplay;

    if allow_neg
        if ~(isscalar(boxB) && isfinite(boxB) && boxB > 0)
            error('WeightBoxB must be a positive finite scalar, e.g., 1 or 2.');
        end
    end

    %% ---------- Dimension checks ----------
    [n, K] = size(M1);

    assert(K == 99, ...
        'Each base model must contain 99 quantile columns, corresponding to alpha = 0.01:0.99.');

    assert(isequal(size(M2), [n, K]) && isequal(size(M3), [n, K]), ...
        'M2 and M3 must have the same size as M1.');

    y = y(:);
    assert(numel(y) == n, ...
        'The length of y must match the number of forecast samples.');

    quantile_levels = (1:K) / 100;
    model_names = {'M1'; 'M2'; 'M3'};

    Qall = cat(3, M1, M2, M3);   % n x K x 3
    nModels = size(Qall, 3);

    %% ---------- Train/test split ----------
    n_test = max(1, round((1 - train_ratio) * n));
    n_train = n - n_test;
    n_train = max(1, min(n - 1, n_train));

    idx_train = 1:n_train;
    idx_test  = (n_train + 1):n;

    Q_train = Qall(idx_train, :, :);
    y_train = y(idx_train);

    Q_test = Qall(idx_test, :, :);
    y_test = y(idx_test);

    %% ---------- Set theta_star: fixed or automatic ----------
    theta_star = resolve_theta_star( ...
        theta_star, Q_train, y_train, quantile_levels, ...
        gamma_theta, nModels, verbose);

    %% ---------- Training subset: one-shot QP for reference metrics ----------
    W_ref = solve_qp_get_W( ...
        Q_train, y_train, theta_star, mu, use_full_monotonic, ...
        allow_neg, boxB, qp_display);

    Q_pred_train = combine_W(Q_train, W_ref);

    [CRPS_train, ~] = crps_aqs_from_Q(Q_pred_train, y_train, quantile_levels);
    NAPS_train = naps_from_Q(Q_pred_train, quantile_levels, omega_set);
    [MRAE_train, ~] = calc_MRAE(y_train, Q_pred_train, quantile_levels);

    metrics_train = struct( ...
        'CRPS', CRPS_train, ...
        'NAPS', NAPS_train, ...
        'MRAE', MRAE_train, ...
        'theta', theta_star);

    %% ---------- Test stage: rolling-window QP ----------
    Q_pred_test_raw = zeros(n_test, K);
    W_stream = cell(n_test, 1);

    for t = 1:n_test

        sample_idx = idx_test(t);

        hist_end = sample_idx - 1;
        hist_start = max(1, hist_end - win + 1);

        % The rolling window uses only historical observations.
        Q_window = Qall(hist_start:hist_end, :, :);
        y_window = y(hist_start:hist_end);

        if isempty(y_window)
            Q_window = Q_train;
            y_window = y_train;
        end

        W_t = solve_qp_get_W( ...
            Q_window, y_window, theta_star, mu, use_full_monotonic, ...
            allow_neg, boxB, qp_display);

        W_stream{t} = W_t;   % K x 3

        Q_current = squeeze(Qall(sample_idx, :, :));   % K x 3
        Q_pred_test_raw(t, :) = sum(Q_current .* W_t, 2)';
    end

    %% ---------- Optional PAV rearrangement for test forecasts ----------
    if project_test_pav
        Q_pred_test = pav_rearrange_nondec(Q_pred_test_raw);
    else
        Q_pred_test = Q_pred_test_raw;
    end

    assert(size(Q_pred_test, 1) == numel(y_test), ...
        'The final ensemble forecasts and y_test have inconsistent lengths.');

    %% ---------- Final ensemble metrics ----------
    [CRPS_raw, ~] = crps_aqs_from_Q(Q_pred_test_raw, y_test, quantile_levels);
    NAPS_raw = naps_from_Q(Q_pred_test_raw, quantile_levels, omega_set);
    [MRAE_raw, ~] = calc_MRAE(y_test, Q_pred_test_raw, quantile_levels);

    [CRPS_test, ~] = crps_aqs_from_Q(Q_pred_test, y_test, quantile_levels);
    NAPS_test = naps_from_Q(Q_pred_test, quantile_levels, omega_set);
    [MRAE_test, ~] = calc_MRAE(y_test, Q_pred_test, quantile_levels);

    point_comb = point_metrics_from_Q(Q_pred_test, y_test);

    pinc_set = [0.70, 0.80, 0.90];
    interval_comb = interval_metrics_from_Q(Q_pred_test, y_test, pinc_set);

    metrics_test = struct( ...
        'CRPS', CRPS_test, ...
        'NAPS', NAPS_test, ...
        'MRAE', MRAE_test, ...
        'theta', theta_star, ...
        'point', point_comb, ...
        'interval', interval_comb, ...
        'raw', struct( ...
            'CRPS', CRPS_raw, ...
            'NAPS', NAPS_raw, ...
            'MRAE', MRAE_raw));

    %% ---------- Baseline model evaluation ----------
    assert(size(Q_test, 1) == numel(y_test), ...
        'The baseline test forecasts and y_test have inconsistent lengths.');

    assert(size(Q_pred_test, 1) == numel(y_test), ...
        'The ensemble test forecasts and y_test have inconsistent lengths.');

    rows = cell(nModels, 1);

    for m = 1:nModels

        Q_base = Q_test(:, :, m);   % n_test x K

        assert(size(Q_base, 1) == numel(y_test), ...
            'The test forecast length of base model %d does not match y_test.', m);

        assert(size(Q_base, 1) == size(Q_pred_test, 1), ...
            'The test forecast length of base model %d does not match the ensemble forecast length.', m);

        [crps_base, ~] = crps_aqs_from_Q(Q_base, y_test, quantile_levels);
        naps_base = naps_from_Q(Q_base, quantile_levels, omega_set);
        [mrae_base, ~] = calc_MRAE(y_test, Q_base, quantile_levels);

        point_base = point_metrics_from_Q(Q_base, y_test);
        interval_base = interval_metrics_from_Q(Q_base, y_test, pinc_set);

        rows{m} = pack_row( ...
            model_names{m}, 'raw', ...
            crps_base, naps_base, mrae_base, ...
            point_base, interval_base);
    end

    baseline_test = struct2table(vertcat(rows{:}));
end