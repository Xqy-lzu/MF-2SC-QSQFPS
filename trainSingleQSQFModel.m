function model = trainSingleQSQFModel(XTrain, YTrain, XVal, YVal, alphas, cfg)

    k = cfg.k;
    inputSize = cfg.inputSize;
    outDim = 2*k + 3;

    layers = sequenceBackbone(inputSize, outDim, ...
        'hiddenUnitsVec', cfg.hiddenUnitsVec, ...
        'drop', cfg.drop);

    net = dlnetwork(layers);

    miniBatchSize = cfg.miniBatchSize;
    numEpochs = cfg.numEpochs;
    numObservations = size(XTrain,2);
    numIterationsPerEpoch = ceil(numObservations / miniBatchSize);

    averageGrad = [];
    averageSqGrad = [];
    iteration = 0;

    if ~isfield(cfg, 'verbose')
        cfg.verbose = true;
    end

    if ~isfield(cfg, 'useEarlyStopping')
        cfg.useEarlyStopping = false;
    end

    if ~isfield(cfg, 'earlyStoppingPatience')
        cfg.earlyStoppingPatience = 30;
    end

    if ~isfield(cfg, 'earlyStoppingMinDelta')
        cfg.earlyStoppingMinDelta = 1e-6;
    end

    useEarlyStopping = cfg.useEarlyStopping;
    patience = cfg.earlyStoppingPatience;
    minDelta = cfg.earlyStoppingMinDelta;

    bestNet = net;
    bestValLoss = inf;
    bestEpoch = 0;
    waitCount = 0;

    trainLossHistory = NaN(numEpochs,1);
    valLossHistory   = NaN(numEpochs,1);

    stopReason = "Reached maximum epochs";

    for epoch = 1:numEpochs

        order = randperm(numObservations);
        epochLoss = zeros(1, numIterationsPerEpoch);

        for i = 1:numIterationsPerEpoch
            iteration = iteration + 1;

            idxStart = (i-1)*miniBatchSize + 1;
            idxEnd   = min(i*miniBatchSize, numObservations);
            idx      = order(idxStart:idxEnd);

            Xtmp = single(XTrain(:,idx));
            Xb3D = permute(Xtmp, [3 2 1]);    % [1 x B x seqLen]
            Xb   = dlarray(Xb3D, 'CBT');

            Tb = dlarray(single(YTrain(:,idx)), 'CB');

            [batchLoss, gradients] = dlfeval(@Loss_Qspline, net, Xb, Tb, k);

            lossValue = double(gather(extractdata(batchLoss)));

            if isnan(lossValue) || isinf(lossValue)
                fprintf('NaN/Inf loss detected at Epoch %d, Iteration %d\n', epoch, iteration);
                error('Training stopped: loss became NaN or Inf.');
            end

            [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, ...
                averageGrad, averageSqGrad, iteration, cfg.learnRate);

            epochLoss(i) = lossValue;
        end

        meanTrainLoss = mean(epochLoss(~isnan(epochLoss)));
        trainLossHistory(epoch) = meanTrainLoss;

        if useEarlyStopping

            tmpModel = struct();
            tmpModel.net = net;
            tmpModel.k   = k;

            Qval_epoch = predictSingleQSQFModel(tmpModel, XVal, alphas);
            valCRPS_each = crps_each_from_Q(Qval_epoch, YVal(:), alphas);
            valLoss = mean(valCRPS_each);

            valLossHistory(epoch) = valLoss;

            if isnan(valLoss) || isinf(valLoss)
                fprintf('NaN/Inf validation loss detected at Epoch %d\n', epoch);
                error('Training stopped: validation loss became NaN or Inf.');
            end

            if valLoss < bestValLoss - minDelta
                bestValLoss = valLoss;
                bestNet = net;
                bestEpoch = epoch;
                waitCount = 0;
            else
                waitCount = waitCount + 1;
            end

            if cfg.verbose
                fprintf('  Epoch %3d/%3d, train loss = %.6f, val CRPS = %.6f, best val CRPS = %.6f, wait = %d/%d\n', ...
                    epoch, numEpochs, meanTrainLoss, valLoss, bestValLoss, waitCount, patience);
            end

            if waitCount >= patience
                stopReason = "Early stopping";
                if cfg.verbose
                    fprintf('  Early stopping triggered at Epoch %d. Best Epoch = %d, Best Val CRPS = %.6f\n', ...
                        epoch, bestEpoch, bestValLoss);
                end
                break;
            end

        else
            if cfg.verbose
                fprintf('  Epoch %3d/%3d, mean loss = %.6f\n', ...
                    epoch, numEpochs, meanTrainLoss);
            end
        end
    end

    if useEarlyStopping
        net = bestNet;
    end

    actualEpochs = find(~isnan(trainLossHistory), 1, 'last');

    model = struct();
    model.net = net;
    model.k = k;

    model.trainLossHistory = trainLossHistory;
    model.valLossHistory = valLossHistory;
    model.bestValLoss = bestValLoss;
    model.bestEpoch = bestEpoch;
    model.stopReason = stopReason;
    model.actualEpochs = actualEpochs;
end
