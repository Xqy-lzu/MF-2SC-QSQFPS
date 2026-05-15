function [Data, SplitInfo] = prepare_midas_fixed_split( ...
    X_Power, X_Speed, Y, X_date, Y_date, ...
    nHigh, xlag, ylag, horizon, splitRatio)

    if abs(sum(splitRatio) - 1) > 1e-10
        error('The three elements of splitRatio must sum to 1.');
    end

    XX  = X_Power(1:nHigh,:);
    XXX = X_Speed(1:nHigh,:);

    L = length(XX);

    if mod(L, horizon) ~= 0
        error('The high-frequency length L cannot be exactly divided by horizon.');
    end

    nLowTotal = L / horizon;

    XX_date = X_date(1:nHigh,:);
    YY      = Y(1:nLowTotal,:);
    YY_date = Y_date(1:nLowTotal,:);

    nTotal = nLowTotal - ylag;

    if nTotal <= 10
        error('The number of available supervised samples is too small. Please check ylag or the data length.');
    end

    nTrain = floor(splitRatio(1) * nTotal);
    nVal   = floor(splitRatio(2) * nTotal);
    nTest  = nTotal - nTrain - nVal;

    nTrainVal = nTrain + nVal;

    estStart = YY_date{ylag + 1};
    estEnd   = YY_date{ylag + nTrainVal};

    Output1 = AAAMIDAS(XX(:,1), YY(:,1), XX_date, YY_date, ...
                       xlag, ylag, horizon, estStart, estEnd);

    TrainValX1 = Output1.TrainX';
    TestX1     = Output1.TestX';

    TrainValYL = Output1.TrainYL';
    TestYL     = Output1.TestYL';

    Output2 = AAAMIDAS(XXX(:,1), YY(:,1), XX_date, YY_date, ...
                       xlag, ylag, horizon, estStart, estEnd);

    TrainValX2 = Output2.TrainX';
    TestX2     = Output2.TestX';

    TrainValX = [TrainValX1; TrainValX2; TrainValYL];
    TestX     = [TestX1;     TestX2;     TestYL];

    if size(TrainValX,2) < nTrainVal
        error('Insufficient TrainValX samples: required %d, but got %d.', ...
            nTrainVal, size(TrainValX,2));
    end

    if size(TestX,2) < nTest
        error('Insufficient TestX samples: required %d, but got %d.', ...
            nTest, size(TestX,2));
    end

    TrainValX = TrainValX(:, 1:nTrainVal);
    TestX     = TestX(:, 1:nTest);

    TrainX = TrainValX(:, 1:nTrain);
    ValX   = TrainValX(:, nTrain+1:nTrain+nVal);

    TrainY = Y(ylag+1 : ylag+nTrain)';
    ValY   = Y(ylag+nTrain+1 : ylag+nTrain+nVal)';
    TestY  = Y(ylag+nTrain+nVal+1 : ylag+nTrain+nVal+nTest)';

    if size(TrainX,2) ~= numel(TrainY)
        error('The number of TrainX samples does not match the length of TrainY.');
    end

    if size(ValX,2) ~= numel(ValY)
        error('The number of ValX samples does not match the length of ValY.');
    end

    if size(TestX,2) ~= numel(TestY)
        error('The number of TestX samples does not match the length of TestY.');
    end

    Data = struct();
    Data.TrainX = TrainX;
    Data.ValX   = ValX;
    Data.TestX  = TestX;
    Data.TrainY = TrainY;
    Data.ValY   = ValY;
    Data.TestY  = TestY;

    SplitInfo = struct();
    SplitInfo.nLowTotal = nLowTotal;
    SplitInfo.nTotal = nTotal;
    SplitInfo.nTrain = nTrain;
    SplitInfo.nVal = nVal;
    SplitInfo.nTest = nTest;
    SplitInfo.nTrainVal = nTrainVal;
    SplitInfo.estStart = estStart;
    SplitInfo.estEnd = estEnd;
    SplitInfo.xlag = xlag;
    SplitInfo.ylag = ylag;
    SplitInfo.horizon = horizon;
end
