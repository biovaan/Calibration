function GPC_5x2CV( Path, Name, Approx )
    % Path: path to data files
    % Name: name used in the data files (Name_S1_1.csv etc.)
    % Approx: the algorithm to use to approximate the GP (Laplace,
    % EP (expectation propagation, or MCMC (Markov chain Monte Carlo)
    
    % we will do a 5x2CV
    for i = 1 : 5
        fprintf("Replication %i/5\n%s\n", i, datetime)
        % read the data files
        S1_fn = join([Path, '/', Name, '_S1_', int2str(i), '.csv'], '');
        S1 = csvread(S1_fn);
        S2_fn = join([Path, '/', Name, '_S2_', int2str(i), '.csv'], '');
        S2 = csvread(S2_fn);
        
        % set up GP
        meanfunc = {@meanZero}; hyp.mean = [];
        ell = 1; sf = 1;
        covfunc = {@covSEiso}; hyp.cov = [log(ell) log(sf)];
        likfunc = {@likLogistic}; hyp.lik = [];

        % fold 1, use S1 as training data and S2 as test data
        columns = size(S1, 2);
        x = S1(:, 1:(columns-1));
        y = S1(:, columns);
        t = S2(:, 1:(columns-1));
        ty = S2(:, columns);
        
        % time the operation
        tic;
        if (strcmp(Approx, 'EP'))
            hyp2 = minimize(hyp, @gp, -100, @infEP, meanfunc, covfunc, likfunc, x, y);
            [a, b, c, d, lp] = gp(hyp2, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(length(t),1));
        elseif (strcmp(Approx, 'Laplace'))
            hyp2 = minimize(hyp, @gp, -100, @infLaplace, meanfunc, covfunc, likfunc, x, y);
            [a, b, c, d, lp] = gp(hyp2, @infLaplace, meanfunc, covfunc, likfunc, x, y, t, ones(length(t),1));
        elseif (strcmp(Approx, 'MCMC'))
            par.sampler = 'hmc'; par.Nsample = 1000; par.Nskip=100; par.Nburnin = 100;
            [post, nlZ, dnlZ,] = infMCMC(hyp, meanfunc, covfunc, likfunc, x, y, par);
            [a, b, c, d, lp] = gp(hyp, @infMCMC, meanfunc, covfunc, likfunc, x, post, t, ones(length(t), 1));
        end
        time1(i) = toc;
        
        posteriors = exp(lp);
        % class 1 if posterior is over 0.5, otherwise -1
        cl = (posteriors >= 0.5) * 1 + (posteriors < 0.5) * (-1);
        % classification rate
        cr1(i) = sum(cl == ty) / length(ty);
        % mean squared error
        % transform class labels to {0,1}
        ty2 = ty > 0;
        mse1(i) = mean((posteriors - ty2) .^2);
        % logarithmic loss
        eps = 1e-15;
        logloss1(i) = -sum((ty > 0) .* log(max(posteriors, eps)) + (ty < 0) .* log(max((1 - posteriors), eps))) / length(ty);

        fprintf("Fold one took %f minutes\nLogloss was %f\n", time1(i)/60, logloss1(i))

        % fold 2: switch S1 and S2 roles

        x = S2(:, 1:(columns-1));
        y = S2(:, columns);
        t = S1(:, 1:(columns-1));
        ty = S1(:, columns);

        % time the operation
        tic;
        if (strcmp(Approx, 'EP'))
            hyp2 = minimize(hyp, @gp, -100, @infEP, meanfunc, covfunc, likfunc, x, y);
            [a, b, c, d, lp] = gp(hyp2, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(length(t),1));
        elseif (strcmp(Approx, 'Laplace'))
            hyp2 = minimize(hyp, @gp, -100, @infLaplace, meanfunc, covfunc, likfunc, x, y);
            [a, b, c, d, lp] = gp(hyp2, @infLaplace, meanfunc, covfunc, likfunc, x, y, t, ones(length(t),1));
        elseif (strcmp(Approx, 'MCMC'))
            par.sampler = 'hmc'; par.Nsample = 1000; par.Nskip=100; par.Nburnin = 100;
            [post, nlZ, dnlZ,] = infMCMC(hyp, meanfunc, covfunc, likfunc, x, y, par);
            [a, b, c, d, lp] = gp(hyp, @infMCMC, meanfunc, covfunc, likfunc, x, post, t, ones(length(t), 1));
        end
        time2(i) = toc;

        posteriors = exp(lp);
        % class 1 if posterior is over 0.5, otherwise -1
        cl = (posteriors >= 0.5) * 1 + (posteriors < 0.5) * (-1);
        % classification rate
        cr2(i) = sum(cl == ty) / length(ty);
        % mean squared error
        % transform class labels to {0,1}
        ty2 = ty > 0;
        mse2(i) = mean((posteriors - ty2) .^2);
        % logarithmic loss
        eps = 1e-15;
        logloss2(i) = -sum((ty > 0) .* log(max(posteriors, eps)) + (ty < 0) .* log(max((1 - posteriors), eps))) / length(ty);
        
        fprintf("Fold two took %f minutes\nLogloss was %f\n", time2(i)/60, logloss2(i))
    end
    
    % write the results in csv files
    fn = join([Path, '/', Name, '_S1_GP_', Approx, '_results.csv'], '');
    writetable(table(cr1', mse1', logloss1', time1'), fn, 'Delimiter', ';', 'WriteVariableNames', true);

    fn = join([Path, '/', Name, '_S2_GP_', Approx, '_results.csv'], '');
    writetable(table(cr2', mse2', logloss2', time2'), fn, 'Delimiter', ';', 'WriteVariableNames', true);
end