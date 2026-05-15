clc
clear
%% ====== Whether to perform hyperparameter search ======
doHyperparameterSearch = false;
% true  : use the validation set for hyperparameter search
% false : use the default hyperparameters
%% ====== Epoch settings ======
searchEpochs = 10;
finalEpochs  = 100;
%% ====== Data and lag parameters ======
load site_15min
nHigh = 25920;
horizon = 3;
xlag = 6;
ylag = 5;
splitRatio = [0.7, 0.1, 0.2];
%% ====== Construct MIDAS features and split the data into training/validation/test sets ======
[Data, SplitInfo] = prepare_midas_fixed_split( ...
    X_Power, X_Speed, Y, X_date, Y_date, ...
    nHigh, xlag, ylag, horizon, splitRatio);

TrainX = Data.TrainX;
ValX   = Data.ValX;
TestX  = Data.TestX;

TrainY = Data.TrainY;
ValY   = Data.ValY;
TestY  = Data.TestY;

fprintf('\n========== Data split information ==========\n');
fprintf('xlag = %d, ylag = %d, horizon = %d\n', xlag, ylag, horizon);
fprintf('Number of training samples   = %d\n', size(TrainX,2));
fprintf('Number of validation samples = %d\n', size(ValX,2));
fprintf('Number of test samples       = %d\n', size(TestX,2));
fprintf('Feature dimension            = %d\n', size(TrainX,1));
fprintf('estStart                     = %s\n', SplitInfo.estStart);
fprintf('estEnd                       = %s\n', SplitInfo.estEnd);
fprintf('============================================\n');

%% ====== Normalization ======
[XTrain, inputps]  = mapminmax(TrainX);
XVal               = mapminmax('apply', ValX, inputps);
XTest              = mapminmax('apply', TestX, inputps);

[YTrain, outputps] = mapminmax(TrainY, 0, 1);
YVal               = mapminmax('apply', ValY, outputps);

%% ====== Quantile levels ======
alphas = 0.01:0.01:0.99;
omega_set = 0.05:0.05:0.95;

%% ====== Base configuration ======
cfgBase = struct();

cfgBase.inputSize = 1;

% ===== Training configuration =====
cfgBase.miniBatchSize = 256;
cfgBase.learnRate = 1e-4;
cfgBase.numEpochs = finalEpochs;

% ===== Early stopping configuration =====
cfgBase.useEarlyStopping = true;
cfgBase.earlyStoppingPatience = 30;
cfgBase.earlyStoppingMinDelta = 1e-6;

% ===== Network configuration =====
cfgBase.drop = 0.1;

% ===== Homogeneous ensemble configuration =====
cfgBase.tauTemp  = 0.02;
cfgBase.epsGamma = 1e-10;

%% =========================================================================
%  Determine whether to perform hyperparameter search
% =========================================================================
if doHyperparameterSearch

    %% ====== Hyperparameter search space ======
    hiddenSet = [4, 8, 16];
    depthSet  = [1, 2, 3];
    kSet      = [10, 20, 30];
    NmSet     = [2, 3, 4];

    totalTrials = numel(hiddenSet) * numel(depthSet) * numel(kSet) * numel(NmSet);
    trial = 0;

    bestScore = inf;
    bestInfo  = struct();

    tuningResults = table();

    cfgSearchBase = cfgBase;
    cfgSearchBase.numEpochs = searchEpochs;
    cfgSearchBase.verbose = false;

    fprintf('\n========== Start grid search: %d hyperparameter combinations ==========\n', totalTrials);
    fprintf('Epochs per trial during the search stage = %d\n', searchEpochs);
    fprintf('Search criterion: Validation CRPS\n');

    for hiddenUnits = hiddenSet
        for numLayers = depthSet
            for k = kSet
                for Nm = NmSet

                    trial = trial + 1;

                    cfg = cfgSearchBase;
                    cfg.k = k;
                    cfg.hiddenUnitsVec = hiddenUnits * ones(1, numLayers);

                    fprintf('\nTrial %3d/%3d | hidden=%d, depth=%d, k=%d, Nm=%d, epoch=%d\n', ...
                        trial, totalTrials, hiddenUnits, numLayers, k, Nm, cfg.numEpochs);

                    valCRPS = NaN;
                    meanActualEpochs = NaN;
                    status = "OK";

                    try
                        ensembleTmp = trainHomoEnsemble_TrainOnly( ...
                            XTrain, YTrain, XVal, YVal, alphas, Nm, cfg);

                        QvalEns_norm = predictHomoEnsemble( ...
                            ensembleTmp, XVal, alphas);

                        valCRPS = crps_from_Q(QvalEns_norm, YVal(:), alphas);

                        if isfield(ensembleTmp, 'actualEpochs')
                            meanActualEpochs = mean(ensembleTmp.actualEpochs);
                        end

                        fprintf('Validation result: CRPS = %.6f, meanActualEpochs = %.2f\n', ...
                            valCRPS, meanActualEpochs);

                        if valCRPS < bestScore
                            bestScore = valCRPS;

                            bestInfo.hiddenUnits = hiddenUnits;
                            bestInfo.numLayers = numLayers;
                            bestInfo.hiddenUnitsVec = cfg.hiddenUnitsVec;
                            bestInfo.k = k;
                            bestInfo.Nm = Nm;
                            bestInfo.searchEpochs = searchEpochs;
                            bestInfo.finalEpochs = finalEpochs;
                            bestInfo.valCRPS_search = valCRPS;
                            bestInfo.meanActualEpochs_search = meanActualEpochs;
                            bestInfo.searchMode = "grid search";

                            fprintf('>>> The current best hyperparameters have been updated.\n');
                        end

                    catch ME
                        status = "FAIL: " + string(ME.message);
                        fprintf('This hyperparameter combination failed: %s\n', ME.message);
                    end

                    newRow = table( ...
                        trial, hiddenUnits, numLayers, k, Nm, ...
                        cfg.numEpochs, valCRPS, meanActualEpochs, status, ...
                        'VariableNames', { ...
                        'Trial', 'HiddenUnits', 'NumLayers', 'K', 'Nm', ...
                        'SearchEpochs', 'ValCRPS', ...
                        'MeanActualEpochs', 'Status'});

                    tuningResults = [tuningResults; newRow];

                end
            end
        end
    end

    fprintf('\n========== Grid search completed ==========\n');

    if ~isfield(bestInfo, 'k')
        error('All hyperparameter combinations failed. Please check the model structure or data.');
    end

    fprintf('\nBest hyperparameters obtained from the search stage:\n');
    fprintf('Hidden units hiddenUnits      = %d\n', bestInfo.hiddenUnits);
    fprintf('Network depth numLayers       = %d\n', bestInfo.numLayers);
    fprintf('Hidden structure hiddenUnitsVec = %s\n', mat2str(bestInfo.hiddenUnitsVec));
    fprintf('Number of QSQF knots k        = %d\n', bestInfo.k);
    fprintf('Number of homogeneous learners Nm = %d\n', bestInfo.Nm);
    fprintf('Search-stage epochs           = %d\n', bestInfo.searchEpochs);
    fprintf('Validation CRPS in search stage = %.6f\n', bestInfo.valCRPS_search);

    cfg = cfgBase;
    cfg.k = bestInfo.k;
    cfg.hiddenUnitsVec = bestInfo.hiddenUnitsVec;
    cfg.numEpochs = finalEpochs;
    cfg.verbose = true;

    Nm = bestInfo.Nm;

else

    %% ====== No search: use default hyperparameters ======
    fprintf('\n========== No hyperparameter search: using default hyperparameters ==========\n');

    cfg = cfgBase;

    cfg.k = 20;
    cfg.hiddenUnitsVec = [16, 8];

    cfg.numEpochs = finalEpochs;
    cfg.verbose = true;

    Nm = 3;

    bestInfo = struct();
    bestInfo.hiddenUnitsVec = cfg.hiddenUnitsVec;
    bestInfo.numLayers = numel(cfg.hiddenUnitsVec);
    bestInfo.k = cfg.k;
    bestInfo.Nm = Nm;
    bestInfo.searchMode = "default parameters";
    bestInfo.finalEpochs = finalEpochs;

    tuningResults = table();

end

%% ====== Train the final model ======
fprintf('\n========== Final model training ==========\n');
fprintf('Mode searchMode             = %s\n', string(bestInfo.searchMode));
fprintf('Hidden structure hiddenUnitsVec = %s\n', mat2str(cfg.hiddenUnitsVec));
fprintf('Network depth numLayers      = %d\n', numel(cfg.hiddenUnitsVec));
fprintf('Number of QSQF knots k       = %d\n', cfg.k);
fprintf('Number of homogeneous learners Nm = %d\n', Nm);
fprintf('Final training epochs        = %d\n', cfg.numEpochs);
fprintf('Early stopping enabled       = %d\n', cfg.useEarlyStopping);
fprintf('Early stopping patience      = %d\n', cfg.earlyStoppingPatience);

ensembleModel = trainHomoEnsemble_TrainOnly( ...
    XTrain, YTrain, XVal, YVal, alphas, Nm, cfg);

bestInfo.valCRPS_eachLearner_final = ensembleModel.valCRPS;
bestInfo.valCRPS_final = mean(ensembleModel.valCRPS);
bestInfo.finalModelEpochs = cfg.numEpochs;

%% ====== Test-set prediction ======
QTest_norm = predictHomoEnsemble(ensembleModel, XTest, alphas);

%% ====== Reverse normalization ======
Q  = mapminmax('reverse', QTest_norm', outputps)';
TT = TestY(:);

%% ====== Probabilistic forecasting metrics: CRPS, MRAE, NAPS ======
CRPS = crps_from_Q(Q, TT, alphas);
[MRAE, MRAE_hat] = calc_MRAE(TT, Q, alphas);
NAPS = naps_from_Q(Q, alphas, omega_set);

ProbMetrics = struct();
ProbMetrics.CRPS = CRPS;
ProbMetrics.MRAE = MRAE;
ProbMetrics.NAPS = NAPS;

M = [CRPS, MRAE, NAPS];

%% ====== Output final test-set performance ======
fprintf('\n===== Final test-set probabilistic forecasting metrics =====\n');
fprintf('CRPS = %.6f\n', ProbMetrics.CRPS);
fprintf('MRAE = %.6f\n', ProbMetrics.MRAE);
fprintf('NAPS = %.6f\n', ProbMetrics.NAPS);





