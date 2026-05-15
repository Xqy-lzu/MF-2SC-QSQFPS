function layers = sequenceBackbone(inputSize, outDim, varargin)

    p = inputParser;
    addParameter(p, 'hiddenUnitsVec', [16, 8]);
    addParameter(p, 'drop', 0.1);
    parse(p, varargin{:});

    hiddenUnitsVec = p.Results.hiddenUnitsVec;
    drop           = p.Results.drop;

    numSeqLayers = numel(hiddenUnitsVec);

    layers = [
        sequenceInputLayer(inputSize, 'Normalization','none', 'Name','input')
    ];

    for ell = 1:numSeqLayers

        if ell < numSeqLayers
            outputMode = 'sequence';
        else
            outputMode = 'last';
        end

        layers = [
            layers
            lstmLayer(hiddenUnitsVec(ell), ...
                'OutputMode', outputMode, ...
                'Name', sprintf('seq%d', ell))
            dropoutLayer(drop, ...
                'Name', sprintf('drop_seq%d', ell))
        ];
    end

    layers = [
        layers
        fullyConnectedLayer(128, 'Name','fc0')
        reluLayer('Name','relu0')
        dropoutLayer(drop, 'Name','drop0')
        fullyConnectedLayer(128, 'Name','fc1')
        reluLayer('Name','relu1')
        dropoutLayer(drop, 'Name','drop1')
        fullyConnectedLayer(outDim, 'Name','fc_out')
    ];
end