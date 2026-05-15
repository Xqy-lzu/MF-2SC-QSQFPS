function ensembleModel = trainHomoEnsemble_TrainOnly( ...
    XTrain, YTrain, XVal, YVal, alphas, Nm, cfg)

    if ~isfield(cfg, 'verbose')
        cfg.verbose = true;
    end

    Ntr = size(XTrain,2);

    if Ntr < 10
        error('The training set is too short to train the model.');
    end

    gamma = ones(1, Ntr) / Ntr;

    models = cell(Nm,1);

    valCRPS = zeros(1,Nm);
    trainCRPS_mean = zeros(1,Nm);

    actualEpochs  = zeros(Nm,1);
    bestEpochs    = zeros(Nm,1);
    bestValLosses = zeros(Nm,1);

    for m = 1:Nm
        if cfg.verbose
            fprintf('\n================ Base learner %d / %d ================\n', m, Nm);
        end

        idx = weighted_resample_idx(Ntr, gamma);
        Xm = XTrain(:, idx);
        Ym = YTrain(:, idx);

        model_m = trainSingleQSQFModel( ...
            Xm, Ym, XVal, YVal, alphas, cfg);

        models{m} = model_m;

        actualEpochs(m)  = model_m.actualEpochs;
        bestEpochs(m)    = model_m.bestEpoch;
        bestValLosses(m) = model_m.bestValLoss;

        Qtrain_m = predictSingleQSQFModel(model_m, XTrain, alphas);
        CRPS_train_each = crps_each_from_Q(Qtrain_m, YTrain(:), alphas);
        trainCRPS_mean(m) = mean(CRPS_train_each);

        gamma = CRPS_train_each(:)' + cfg.epsGamma;
        gamma = gamma / sum(gamma);

        Qval_m = predictSingleQSQFModel(model_m, XVal, alphas);
        valCRPS_each = crps_each_from_Q(Qval_m, YVal(:), alphas);
        valCRPS(m) = mean(valCRPS_each);

        if cfg.verbose
            fprintf('Base learner %d | train mean CRPS = %.6f | validation mean CRPS = %.6f | actualEpochs = %d | bestEpoch = %d\n', ...
                m, trainCRPS_mean(m), valCRPS(m), actualEpochs(m), bestEpochs(m));
        end
    end

    tau_m = softmax_neg(valCRPS, cfg.tauTemp);

    ensembleModel = struct();
    ensembleModel.models = models;
    ensembleModel.tau_m = tau_m;
    ensembleModel.valCRPS = valCRPS;
    ensembleModel.trainCRPS_mean = trainCRPS_mean;
    ensembleModel.cfg = cfg;

    ensembleModel.actualEpochs = actualEpochs;
    ensembleModel.bestEpochs = bestEpochs;
    ensembleModel.bestValLosses = bestValLosses;
end
