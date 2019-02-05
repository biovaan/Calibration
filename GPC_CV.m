function GPC_CV( Path, Name, Approx )
    % Path: path to data files
    % Name: name used in the data files (Name_training_fold_1.csv etc.)
    % Approx: the algorithm to use to approximate the GP (Laplace,
    % EP (expectation propagation, or MCMC (Markov chain Monte Carlo)
    
    % we will do a 10-fold CV
    for i = 1 : 10
        fprintf("Replication %i/10\n%s\n", i, datetime)
        % read the data files
        training_fn = join([Path, '/', Name, '_training_fold_', int2str(i), '.csv'], '');
        training = csvread(training_fn);
        test_fn = join([Path, '/', Name, '_test_fold_', int2str(i), '.csv'], '');
        test = csvread(test_fn);
        
        % set up GP
        meanfunc = {@meanZero}; hyp.mean = [];
        ell = 1; sf = 1;
        covfunc = {@covSEiso}; hyp.cov = [log(ell) log(sf)];
        likfunc = {@likLogistic}; hyp.lik = [];

        columns = size(training, 2);
        x = training(:, 1:(columns-1));
        y = training(:, columns);
        t = test(:, 1:(columns-1));
        ty = test(:, columns);
        
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
        time(i) = toc;
        
        posteriors = exp(lp);
        % class 1 if posterior is over 0.5, otherwise -1
        cl = (posteriors >= 0.5) * 1 + (posteriors < 0.5) * (-1);
        % classification rate
        cr(i) = sum(cl == ty) / length(ty);
        % mean squared error
        % transform class labels to {0,1}
        ty2 = ty > 0;
        mse(i) = mean((posteriors - ty2) .^2);
        % logarithmic loss
        eps = 1e-15;
        logloss(i) = -sum((ty > 0) .* log(max(posteriors, eps)) + (ty < 0) .* log(max((1 - posteriors), eps))) / length(ty);

    end
    
    % write the results in csv files
    fn = join([Path, '/', Name, '_CV_GP_', Approx, '_results.csv'], '');
    writetable(table(cr', mse', logloss', time'), fn, 'Delimiter', ';', 'WriteVariableNames', true);

end