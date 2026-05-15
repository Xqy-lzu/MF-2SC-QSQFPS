function Qnorm = predictSingleQSQFModel(model, X, alphas)

    Xtmp = single(X);
    Xb3D = permute(Xtmp, [3 2 1]);    % [1 x N x seqLen]
    Xb   = dlarray(Xb3D, 'CBT');

    Ydl  = predict(model.net, Xb);
    Ydl  = stripdims(Ydl);
    Yout = permute(Ydl, [2 1]);
    Yout = gather(extractdata(Yout));

    [beta_0, beta_2k, sigma, gamma] = unpackParams(Yout, model.k);

    Qnorm = Qpredict_batch(beta_0, beta_2k, sigma, gamma, alphas);
    Qnorm = cummax(Qnorm, 2);
end
