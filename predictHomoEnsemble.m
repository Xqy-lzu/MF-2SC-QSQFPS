function [QEns_norm, QMembers_norm] = predictHomoEnsemble(ensembleModel, X, alphas)

    models = ensembleModel.models;
    tau_m = ensembleModel.tau_m;
    Nm = numel(models);

    needMembers = nargout > 1;

    if needMembers
        QMembers_norm = cell(Nm,1);
    end

    QEns_norm = [];

    for m = 1:Nm

        Qm = predictSingleQSQFModel(models{m}, X, alphas);

        if isempty(QEns_norm)
            QEns_norm = zeros(size(Qm), 'like', Qm);
        end

        QEns_norm = QEns_norm + tau_m(m) * Qm;

        if needMembers
            QMembers_norm{m} = Qm;
        end
    end

    QEns_norm = cummax(QEns_norm, 2);
end
