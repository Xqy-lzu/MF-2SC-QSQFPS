clc
clear
load site_15min_all

%% ================== Main settings ==================
train_ratio = 0.2;

% Used only for NAPS evaluation
omega_set = 0.05:0.05:0.95;

theta_star = [];
use_full_monotonic = true;
win = 4;
mu_alpha = 1e-2;

opts = { ...
    'ProjectTestPAV', true, ...
    'Verbose', true, ...
    'WeightBoxB', 1, ...
    'AllowNegativeWeights', false, ...
    'QPDisplay', 'off'};

[W_stream, Q_pred_te, met_tr, met_te, base_te] = ...
    ensemble_Q_models_2obj_online( ...
        M1, M2, M3, actual, train_ratio, win, ...
        theta_star, mu_alpha, use_full_monotonic, omega_set, ...
        opts{:});








































