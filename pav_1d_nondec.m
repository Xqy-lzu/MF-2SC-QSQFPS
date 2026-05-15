function y = pav_1d_nondec(x)

    K = numel(x);

    block_mean = zeros(1, K);
    block_weight = zeros(1, K);

    nBlocks = 0;

    for j = 1:K

        nBlocks = nBlocks + 1;
        block_mean(nBlocks) = x(j);
        block_weight(nBlocks) = 1;

        while nBlocks > 1 && block_mean(nBlocks-1) > block_mean(nBlocks)

            new_weight = block_weight(nBlocks-1) + block_weight(nBlocks);

            block_mean(nBlocks-1) = ...
                (block_weight(nBlocks-1) * block_mean(nBlocks-1) + ...
                 block_weight(nBlocks)   * block_mean(nBlocks)) / new_weight;

            block_weight(nBlocks-1) = new_weight;

            block_mean(nBlocks:end-1) = block_mean(nBlocks+1:end);
            block_weight(nBlocks:end-1) = block_weight(nBlocks+1:end);

            nBlocks = nBlocks - 1;
        end
    end

    y = zeros(1, K);

    idx = 1;

    for j = 1:nBlocks
        y(idx:idx+block_weight(j)-1) = block_mean(j);
        idx = idx + block_weight(j);
    end
end