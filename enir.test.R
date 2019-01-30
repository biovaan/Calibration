enir.mccv.test <- function(S1, S2, name, classifier = "NB", control = FALSE, seed = 20190124, Nsamples = 5000, bin.size = 100) {
    # run tests that test the usefulness of DG and DGG algorithms for small data set calibration
    # input:  S1, S2: two-class data set split into two equal sized subsets, S1 and S2, where the last column is the label {0,1}
    #         name: a string to be used in the result filenames
    #         classifier: which classifier to use (implemented are NB, SVM, RF and NN)
    #         control: a boolean value indicating if BLR and GPR models should be run also as control cases
    #         seed: a seed for the random value generator to allow replication of the results
    #         Nsamples: number of data points to generate in DG
    #         bin.size: bin size to be used with DGG
    # output: csv files for classification rate, MSE, logloss and computation times

    if (!require(e1071)) {
        install.packages("e1071", dependencies = TRUE)
        require(e1071)
    }
    if (!require(kernlab)) {
        install.packages("kernlab", dependencies = TRUE)
        require(kernlab)
    }
    if (!require(doParallel)) {
        install.packages("doParallel", dependencies = TRUE)
        require(doParallel)
    }
    if (!require(foreach)) {
        install.packages("foreach")
        require(foreach)
    }
    if (!require(randomForest)) {
        install.packages("randomForest", dependencies = TRUE)
        require(randomForest)
    }
    if (!require(nnet)) {
        install.packages("nnet", dependencies = TRUE)
        require(nnet)
    }
    if (!require(arm)) {
        install.packages("arm", dependencies = TRUE)
        require(arm)
    }

	Label <- ncol(S1)
	N.var <- Label - 1
    names(S1)[Label] <- "Label"
    names(S2)[Label] <- "Label"

    source("enir.R")
    source("find.best.threshold.R")

    set.seed(seed)

    # the data set is divided to S1 and S2 data sets (5x2CV)
    # we will use the whole training data (S1) with ENIR full, DG, and DGG
    # a separate calibration data (S1.calibration) is split off for ENIR calibration 
    # and the rest (S1.training) is used for training the classifier
    
    cal.idx <- sample(1:nrow(S1), floor(nrow(S1)*0.1))
    S1.calibration <- S1[cal.idx, ]
    S1.training <- S1[-cal.idx, ]

    # make sure we don't have empty classes
    S1$Label <- factor(S1$Label)
    S1.training$Label <- factor(S1.training$Label)
    S2$Label <- factor(S2$Label)

    # parallel back end for hyperparameter tuning
    # svm has 14 different hyperparameters to test
    cl <- makeCluster(14)
    registerDoParallel(cl)
    clusterExport(cl, varlist = c("S1", "cross", "svm"), envir=environment())

    if (control) {
        # CASE 0: control

        # Bayesian logistic regression

        t.BLR <- system.time({
        mdl.BLR <- bayesglm(Label ~., data = S1, family = binomial(link = "logit"),
            drop.unused.levels = FALSE)
        # predict the training data so that we can find out the optimal threshold
        pred.BLR.tr <- data.frame(label = S1$Label, score = predict(mdl.BLR, S1,
            type = "response"))

        # raw scores for the test data
        prob.BLR <- predict(mdl.BLR, S2, type = "response")
        # find a threshold to maximize classification rate
        threshold.BLR <- find.best.threshold(pred.BLR.tr)
        # predict the test set with using that threshold
        pred.BLR <- ifelse(prob.BLR > threshold.BLR, levels(S2$Label)[2], levels(S2$Label)[1])
        levels(pred.BLR) <- levels(S2$Label) # make sure we have all levels in the predictions
        })

        # Gaussian process uses a Matlab implementation
    }

    # CASE 1: no calibration

    # train the classifier model
    # use the full training data

    if (classifier == "NB") {
        t.model.all <- system.time({
        mdl.raw <- naiveBayes(Label ~., data = S1)
        })
    } else if (classifier == "SVM") {
        t.model.all <- system.time({
        # tune SVM hyperparameters
		gammas <- sigest(Label ~., data = S1)

		accs <- foreach (cost = 2^(-2:11), combine = c) %dopar% {
			mdl <- svm(Label ~., data = S1, scale = FALSE, cost = cost, gamma = gammas[2], cross = 10)
			mdl$tot.accuracy
		}

		best.cost <- 2^(which.max(accs) - 3)

		mdl.raw <- svm(Label ~., data = S1, scale = FALSE, probability = TRUE, type = "C-classification",
            cost = best.cost, gamma = gammas[2])
        })

    } else if (classifier == "RF") {
        t.model.all <- system.time({
        # tune RF hyperparameters
		tune.res <- tuneRF(S1[, 1:N.var], S1[, (N.var+1)], ntreeTry = 500,
            plot = FALSE, trace = FALSE)
        mtry.tuned.raw <- tune.res[which.min(tune.res[, 2]), 1]

        # train the model, keep inbag info for oob calibration
		mdl.raw <- randomForest(Label ~., data = S1, mtry = mtry.tuned.raw, keep.inbag = TRUE)
        })

    } else if (classifier == "NN") {
        t.model.all <- system.time({
        # tune NN hyperparameters (10-fold cv = default)
        tune.obj.raw <- tune.nnet(Label~., data = S1, size = c(1, 3, 5, 7, 9),
            decay = c(0, 0.1, 0.01, 0.001, 0.0001), maxit = 200, trace = FALSE)

		mdl.raw <- tune.obj.raw$best.model
        })
    }

    # find a threshold to maximize classification rate
    if (classifier == "NB") {
        pred.raw.tr <- data.frame(label = S1$Label, score = predict(mdl.raw, S1,
            type = "raw")[, 2])

        # raw scores for the test data
        prob.raw <- predict(mdl.raw, S2, type = "raw")[, 2]

    } else if (classifier == "SVM") {
        p <- predict(mdl.raw, S1, probability = TRUE)
        
        # find which column of the prediction scores is related to the positive class
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(S1$Label)[2])

        pred.raw.tr <- data.frame(label = S1$Label, score = attr(p, "probabilities")[, col.pos])

        # raw scores for the test data
        prob.raw <- attr(predict(mdl.raw, S2, probability = TRUE), "probabilities")[, col.pos]
    } else if (classifier == "RF") {
        pred.raw.tr <- data.frame(label = S1$Label, score = predict(mdl.raw, S1,
            type = "prob")[, 2])

        # raw scores for the test data
        prob.raw <- predict(mdl.raw, S2, type = "prob")[, 2]
    } else if (classifier == "NN") {
        pred.raw.tr <- data.frame(label = S1$Label, score = predict(mdl.raw, S1)[, 1])

        # raw scores for the test data
        prob.raw <- predict(mdl.raw, S2)[, 1]
    }

    threshold.raw <- find.best.threshold(pred.raw.tr)
    # predict the test set with using that threshold
    pred.raw <- ifelse(prob.raw > threshold.raw, levels(S2$Label)[2], levels(S2$Label)[1])
    levels(pred.raw) <- levels(S2$Label) # make sure we have all levels in the predictions

    # CASE 2: ENIR calibration with a separate calibration data set

    # train the classifier model
    # use the training data minus the calibration data

    if (classifier == "NB") {
        t.model <- system.time({
        mdl <- naiveBayes(Label ~., data = S1.training)
        })
    } else if (classifier == "SVM") {
        t.model <- system.time({
        # tune SVM hyperparameters
		gammas <- sigest(Label ~., data = S1.training)

		accs <- foreach (cost = 2^(-2:11), combine = c) %dopar% {
			mdl <- svm(Label ~., data = S1.training, scale = FALSE, cost = cost, gamma = gammas[2], cross = 10)
			mdl$tot.accuracy
		}

		best.cost <- 2^(which.max(accs) - 3)

		mdl <- svm(Label ~., data = S1.training, scale = FALSE, probability = TRUE, type = "C-classification",
            cost = best.cost, gamma = gammas[2])
        })
    } else if (classifier == "RF") {
        t.model <- system.time({
        # tune RF hyperparameters
		tune.res <- tuneRF(S1.training[, 1:N.var], S1.training[, (N.var+1)], ntreeTry = 500,
            plot = FALSE, trace = FALSE)
        mtry.tuned <- tune.res[which.min(tune.res[, 2]), 1]

		mdl <- randomForest(Label ~., data = S1.training, mtry = mtry.tuned)
        })
    } else if (classifier == "NN") {
        t.model <- system.time({
        # tune NN hyperparameters (10-fold cv = default)
        tune.obj <- tune.nnet(Label~., data = S1.training, size = c(1, 3, 5, 7, 9),
            decay = c(0, 0.1, 0.01, 0.001, 0.0001), maxit = 200, trace = FALSE)

		mdl <- tune.obj$best.model
        })
    }

    # predict the calibration data, this is used to tune the calibration model
    if (classifier == "NB") {
        pred.enir.cal <- data.frame(label = S1.calibration$Label, score = predict(mdl, S1.calibration,
            type = "raw")[, 2])
    } else if (classifier == "SVM") {
        p <- predict(mdl, S1.calibration, probability = TRUE)

        # find which column of the prediction scores is related to the positive class:
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(S1.calibration$Label)[2])

        pred.enir.cal <- data.frame(label = S1.calibration$Label, score = attr(p, "probabilities")[, col.pos])
    } else if (classifier == "RF") {
        pred.enir.cal <- data.frame(label = S1.calibration$Label, score = predict(mdl, S1.calibration,
            type = "prob")[, 2])
    } else if (classifier == "NN") {
        pred.enir.cal <- data.frame(label = S1.calibration$Label, score = predict(mdl, S1.calibration)[, 1])
    }

    # tune the calibration model
    t.ENIR <- system.time({
    mdl.enir <- enir.build(pred.enir.cal$score, pred.enir.cal$label)
    })

    # find a threshold to maximize classification rate
    if (classifier == "NB") {
        pred.enir.tr.cal <- data.frame(label = S1.training$Label,
            score = enir.predict(mdl.enir, predict(mdl, S1.training, type = "raw")[, 2]))

        # raw scores for the test data
        prob.enir.raw <- predict(mdl, S2, type = "raw")[, 2]

    } else if (classifier == "SVM") {
        p <- predict(mdl, S1.training, probability = TRUE)

        # find which column of the prediction scores is related to the positive class:
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(S1.training$Label)[2])

        pred.enir.tr.cal <- data.frame(label = S1.training$Label,
            score = enir.predict(mdl.enir, attr(p, "probabilities")[, col.pos]))

        # raw scores for the test data
        prob.enir.raw <- attr(predict(mdl, S2, probability = TRUE), "probabilities")[, col.pos]
    } else if (classifier == "RF") {
        pred.enir.tr.cal <- data.frame(label = S1.training$Label,
            score = enir.predict(mdl.enir, predict(mdl, S1.training, type = "prob")[, 2]))

        # raw scores for the test data
        prob.enir.raw <- predict(mdl, S2, type = "prob")[, 2]
    } else if (classifier == "ELM") {
        pred.enir.tr.cal <- data.frame(label = S1.training$Label,
            score = enir.predict(mdl.enir, elm_predict(mdl, as.matrix(scaled.S1.training[, 1:N.var]),
            normalize = TRUE)[, 2]))

        # raw scores for the test data
        scaled.S2 <- scale.features(S2, scaling.params)
        prob.enir.raw <- elm_predict(mdl, as.matrix(scaled.S2[, 1:N.var]), normalize = TRUE)[, 2]
    } else if (classifier == "NN") {
        pred.enir.tr.cal <- data.frame(label = S1.training$Label,
            score = enir.predict(mdl.enir, predict(mdl, S1.training)[, 1]))

        # raw scores for the test data
        prob.enir.raw <- predict(mdl, S2)[, 1]
    }

    threshold.enir <- find.best.threshold(pred.enir.tr.cal)
    # calibrate the raw prediction scores
    prob.enir.cal <- enir.predict(mdl.enir, prob.enir.raw)
    # predict the test set with using that threshold
    pred.enir <- ifelse(prob.enir.cal > threshold.enir, levels(S2$Label)[2], levels(S2$Label)[1])

    levels(pred.enir) <- levels(S2$Label) # make sure we have all levels in the predictions

    # we don't use the parallel back end anymore
    stopCluster(cl)

    # CASE 2b: ENIR calibration without separate calibration data (use the same data for classifier training and calibration)

    # we'll be using the mdl.raw trained above

    # calibrate the model using the full training data set
    if (classifier == "NB") {
        pred.enir.full.cal <- data.frame(label = S1$Label,
            score = predict(mdl.raw, S1, type = "raw")[, 2])
    } else if (classifier == "SVM") {
        p <- predict(mdl.raw, S1, probability = TRUE)

        # find which column of the prediction scores is related to the positive class:
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(S1$Label)[2])

        pred.enir.full.cal <- data.frame(label = S1$Label, score = attr(p, "probabilities")[, col.pos])
    } else if (classifier == "RF") {
        pred.enir.full.cal <- data.frame(label = S1$Label,
            score = predict(mdl.raw, S1, type = "prob")[, 2])
    } else if (classifier == "NN") {
        pred.enir.full.cal <- data.frame(label = S1$Label,
            score = predict(mdl.raw, S1)[, 1])
    }

    # tune the calibration model
    t.ENIR.full <- system.time({
    mdl.enir.full <- enir.build(pred.enir.full.cal$score, pred.enir.full.cal$label)
    })

    # custom threshold
    if (classifier == "NB") {
        pred.enir.tr.full.cal <- data.frame(label = S1$Label,
            score = enir.predict(mdl.enir.full, predict(mdl.raw, S1, type = "raw")[, 2]))

        # raw scores for the test data
        prob.enir.full.raw <- predict(mdl.raw, S2, type = "raw")[, 2]

    } else if (classifier == "SVM") {
        p <- predict(mdl.raw, S1, probability = TRUE)

        # find which column of the prediction scores is related to the positive class:
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(S1$Label)[2])

        pred.enir.tr.full.cal <- data.frame(label = S1$Label,
            score = enir.predict(mdl.enir.full, attr(p, "probabilities")[, col.pos]))

        # raw scores for the test data
        prob.enir.full.raw <- attr(predict(mdl.raw, S2, probability = TRUE), "probabilities")[, col.pos]
    } else if (classifier == "RF") {
        pred.enir.tr.full.cal <- data.frame(label = S1$Label,
            score = enir.predict(mdl.enir.full, predict(mdl.raw, S1, type = "prob")[, 2]))

        # raw scores for the test data
        prob.enir.full.raw <- predict(mdl.raw, S2, type = "prob")[, 2]
    } else if (classifier == "NN") {
        pred.enir.tr.full.cal <- data.frame(label = S1$Label,
            score = enir.predict(mdl.enir.full, predict(mdl.raw, S1)[, 1]))

        # raw scores for the test data
        prob.enir.full.raw <- predict(mdl.raw, S2)[, 1]
    }

    threshold.enir.full <- find.best.threshold(pred.enir.tr.full.cal)
    # calibrate the raw prediction scores
    prob.enir.full.cal <- enir.predict(mdl.enir.full, prob.enir.full.raw)
    # predict the test set with using that threshold
    pred.enir.full <- ifelse(prob.enir.full.cal > threshold.enir.full, levels(S2$Label)[2],
        levels(S2$Label)[1])

    levels(pred.enir.full) <- levels(S2$Label) # make sure we have all levels in the predictions

    # CASE 3: DG + ENIR calibration

    # use mdl.raw trained above

    t.DG <- system.time({

    # generate calibration data with Monte Carlo CV
    pred.tr <- factor()
	tst.Y <- factor()
	posteriors <- numeric()

    # we need "iter" iterations to get at least Nsamples of calibrations data
	iter <- ceiling(Nsamples / floor(nrow(S1)*0.3))

	for (i in 1:iter) {
		tr <- sample(1:nrow(S1), floor(nrow(S1)*0.7))
		trset <- S1[tr, ]
		tstset <- S1[-tr, ]

		if (classifier == "NB") {
            mdl.DG <- naiveBayes(Label ~., data = trset)
            pred.probs.tr <- predict(mdl.DG, tstset[,1:N.var], type = "raw")[, 2]
            pred.tr <- unlist(list(pred.tr, predict(mdl.DG, tstset[, 1:N.var])))
		} else if (classifier == "SVM") {
			# use the same hyperparameters as with the whole data set
			mdl.DG <- svm(Label ~., data = trset, scale = FALSE, probability = TRUE, type = "C-classification", cost = best.cost,
                gamma = gammas[2])
			p <- predict(mdl.DG, tstset, probability = TRUE)
			pred.tr <- unlist(list(pred.tr, p))

            # find which column of the prediction scores is related to the positive class:
		    col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(trset$Label)[2])

			pred.probs.tr <- attr(p, "probabilities")[, col.pos]
		} else if (classifier == "RF") {
			# use the same tuning parameters as with the whole data set
			mdl.DG <- randomForest(Label ~., data = trset, mtry = mtry.tuned.raw)
			pred.probs.tr <- predict(mdl.DG, tstset, type = "prob")[, 2]
			pred.tr <- unlist(list(pred.tr, predict(mdl.DG, tstset)))
		} else if (classifier == "NN") {
			# use the same tuning parameters as with the whole data set
			mdl.DG <- nnet(Label ~., data = trset, size = tune.obj.raw$best.parameters$size,
                decay = tune.obj.raw$best.parameters$decay, maxit = 200)
			pred.probs.tr <- predict(mdl.DG, tstset)[, 1]
            pred.tr.new <- factor(ifelse(pred.probs.tr > 0.5, 2, 1))
            levels(pred.tr.new) <- levels(tstset$Label)
			pred.tr <- unlist(list(pred.tr, pred.tr.new))
		}

        # true labels
		tst.Y <- unlist(list(tst.Y, tstset$Label))

		# positive class posteriors
		posteriors <- c(posteriors, pred.probs.tr)
	}

    })

    # tune the calibration model
    t.ENIR.DG <- system.time({
    mdl.enir.DG <- enir.build(posteriors, tst.Y)
    })
    
    # custom threshold
    if (classifier == "NB") {
        pred.enir.tr.DG.cal <- data.frame(label = S1$Label,
            score = enir.predict(mdl.enir.DG, predict(mdl.raw, S1, type = "raw")[, 2]))

        # raw scores for the test data
        prob.enir.DG.raw <- predict(mdl.raw, S2, type = "raw")[, 2]

    } else if (classifier == "SVM") {
        p <- predict(mdl.raw, S1, probability = TRUE)

        # find which column of the prediction scores is related to the positive class:
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(S1$Label)[2])

        pred.enir.tr.DG.cal <- data.frame(label = S1$Label,
            score = enir.predict(mdl.enir.DG, attr(p, "probabilities")[, col.pos]))

        # raw scores for the test data
        prob.enir.DG.raw <- attr(predict(mdl.raw, S2, probability = TRUE), "probabilities")[, col.pos]
    } else if (classifier == "RF") {
        pred.enir.tr.DG.cal <- data.frame(label = S1$Label,
            score = enir.predict(mdl.enir.DG, predict(mdl.raw, S1, type = "prob")[, 2]))

        # raw scores for the test data
        prob.enir.DG.raw <- predict(mdl.raw, S2, type = "prob")[, 2]
    } else if (classifier == "NN") {
        pred.enir.tr.DG.cal <- data.frame(label = S1$Label,
            score = enir.predict(mdl.enir.DG, predict(mdl.raw, S1)[, 1]))

        # raw scores for the test data
        prob.enir.DG.raw <- predict(mdl.raw, S2)[, 1]
    }

    threshold.enir.DG <- find.best.threshold(pred.enir.tr.DG.cal)
    # calibrate the raw prediction scores
    prob.enir.DG.cal <- enir.predict(mdl.enir.DG, prob.enir.DG.raw)
    # predict the test set with using that threshold
    pred.enir.DG <- ifelse(prob.enir.DG.cal > threshold.enir.DG, levels(S2$Label)[2], levels(S2$Label)[1])

    levels(pred.enir.DG) <- levels(S2$Label) # make sure we have all levels in the predictions    

    # CASE 4: DGG + ENIR calibration

    t.DGG <- system.time({

    # do the grouping

    df <- data.frame(Posterior = posteriors, pred = pred.tr, true = tst.Y)
	df <- df[with(df, order(Posterior)),] # increasing order
	N <- Nsamples / bin.size
	step <- as.integer(nrow(df) / N)
	y <- numeric()
	x <- numeric()
	for (i in 0:(N-1)) {
		true <- df$true[(i*step+1):((i+1)*step)]

        if (class(true) == "factor") {
            true <- as.integer(true) - 1
        }

        y <- c(y, sum(true) / length(true))
		x <- c(x, mean(df$Posterior[(i*step+1):((i+1)*step)]))
	}

    })

    # tune the calibration model
    t.ENIR.DGG <- system.time({
    mdl.enir.DGG <- enir.build(x, y)
    })

    # custom threshold
    if (classifier == "NB") {
        pred.enir.tr.DGG.cal <- data.frame(label = S1$Label,
            score = enir.predict(mdl.enir.DGG, predict(mdl.raw, S1, type = "raw")[, 2]))

        # raw scores for the test data
        prob.enir.DGG.raw <- predict(mdl.raw, S2, type = "raw")[, 2]

    } else if (classifier == "SVM") {
        p <- predict(mdl.raw, S1, probability = TRUE)

        # find which column of the prediction scores is related to the positive class:
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(S1$Label)[2])

        pred.enir.tr.DGG.cal <- data.frame(label = S1$Label,
            score = enir.predict(mdl.enir.DGG, attr(p, "probabilities")[, col.pos]))

        # raw scores for the test data
        prob.enir.DGG.raw <- attr(predict(mdl.raw, S2, probability = TRUE), "probabilities")[, col.pos]
    } else if (classifier == "RF") {
        pred.enir.tr.DGG.cal <- data.frame(label = S1$Label,
            score = enir.predict(mdl.enir.DGG, predict(mdl.raw, S1, type = "prob")[, 2]))

        # raw scores for the test data
        prob.enir.DGG.raw <- predict(mdl.raw, S2, type = "prob")[, 2]
    } else if (classifier == "NN") {
        pred.enir.tr.DGG.cal <- data.frame(label = S1$Label,
            score = enir.predict(mdl.enir.DGG, predict(mdl.raw, S1)[, 1]))

        # raw scores for the test data
        prob.enir.DGG.raw <- predict(mdl.raw, S2)[, 1]
    }

    threshold.enir.DGG <- find.best.threshold(pred.enir.tr.DGG.cal)
    # calibrate the raw prediction scores
    prob.enir.DGG.cal <- enir.predict(mdl.enir.DGG, prob.enir.DGG.raw)
    # predict the test set with using that threshold
    pred.enir.DGG <- ifelse(prob.enir.DGG.cal > threshold.enir.DGG, levels(S2$Label)[2], levels(S2$Label)[1])

    levels(pred.enir.DGG) <- levels(S2$Label) # make sure we have all levels in the predictions    


    # CASE 5: Platt scaling for SVM
    if (classifier == "SVM") {

    }

    # CASE 6: using out-of-bag samples for random forest ENIR calibration
    if (classifier == "RF") {
        # predict the training set with all individual trees so we can use the oob samples for calibration
        pred.enir.oob <- predict(mdl.raw, S1, predict.all = TRUE)

        # individual tree predictions are here: pred.enir.oob$individual
        # and inbag info here (saved when training the model): mdl.raw$inbag
        # calculate the fraction of trees predicting positive class for out-of-bag samples, i.e. predicted probability:
        prob.enir.oob.tr <- rowSums(ifelse(mdl.raw$inbag > 0, 0,
            ifelse(pred.enir.oob$individual == levels(S1$Label)[2], 1, 0))) / mdl.raw$oob.times

        # tune the calibration model
        t.ENIR.oob <- system.time({
        mdl.enir.oob <- enir.build(prob.enir.oob.tr, pred.enir.oob$aggregate)
        })

        # predict the raw scores for the test set
        prob.enir.oob.raw <- predict(mdl.raw, S2, type = "prob")[, 2]

        # find the optimal threshold
        pred.enir.tr.oob.cal <- data.frame(label = S1$Label,
            score = enir.predict(mdl.enir.oob, predict(mdl.raw, S1, type = "prob")[, 2]))

        threshold.enir.oob <- find.best.threshold(pred.enir.tr.oob.cal)

        # calibrate the raw prediction scores
        prob.enir.oob.cal <- enir.predict(mdl.enir.oob, prob.enir.oob.raw)
        # predict the test set with using that threshold
        pred.enir.oob <- ifelse(prob.enir.oob.cal > threshold.enir.oob, levels(S2$Label)[2], levels(S2$Label)[1])

        levels(pred.enir.oob) <- levels(S2$Label) # make sure we have all levels in the predictions
    }


    # combine the results into data frames that we can then write into csv files
    # append if the files exists so that we can run several simulation rounds

    # accuracies (classification rates)
    df.acc <- data.frame(seed = seed,
            CR.raw = 100 * sum(pred.raw == S2$Label) / nrow(S2),
            CR.ENIR = 100 * sum(pred.enir == S2$Label) / nrow(S2),
            CR.ENIR.full = 100 * sum(pred.enir.full == S2$Label) / nrow(S2),
            CR.DG = 100 * sum(pred.enir.DG == S2$Label) / nrow(S2),
            CR.DGG = 100 * sum(pred.enir.DGG == S2$Label) / nrow(S2))
    if (classifier == "RF") {
        df.acc <- cbind(df.acc, data.frame(CR.ENIR.oob = 100 * sum(pred.enir.oob == S2$Label) / nrow(S2)))
    }
    if (control) {
        df.acc <- cbind(df.acc, data.frame(CR.BLR = 100 * sum(pred.BLR == S2$Label) / nrow(S2)))
    }

    filename <- paste(name, "_", classifier, "_accuracies.csv", sep = "")
	if (!file.exists(filename)) {
		write.table(df.acc, file = filename, sep = ";", dec = ",", col.names = TRUE, row.names = FALSE)
	} else {
		write.table(df.acc, file = filename, append = TRUE, sep = ";", dec = ",", col.names = FALSE, row.names = FALSE)
	}

    # mean squared errors
    df.MSE <- data.frame(seed = seed,
            MSE.raw = MSE(prob.raw, S2$Label),
            MSE.ENIR = MSE(prob.enir.cal, S2$Label),
            MSE.ENIR.full = MSE(prob.enir.full.cal, S2$Label),
            MSE.DG = MSE(prob.enir.DG.cal, S2$Label),
            MSE.DGG = MSE(prob.enir.DGG.cal, S2$Label))
    if (classifier == "RF") {
        df.MSE <- cbind(df.MSE, data.frame(MSE.ENIR.oob = MSE(prob.enir.oob.cal, S2$Label)))
    }
    if (control) {
        df.MSE <- cbind(df.MSE, data.frame(MSE.BLR = MSE(prob.BLR, S2$Label)))
    }

    filename <- paste(name, "_", classifier, "_MSEs.csv", sep = "")
	if (!file.exists(filename)) {
		write.table(df.MSE, file = filename, sep = ";", dec = ",", col.names = TRUE, row.names = FALSE)
	} else {
		write.table(df.MSE, file = filename, append = TRUE, sep = ";", dec = ",", col.names = FALSE, row.names = FALSE)
	}

    # logarithmic losses
    df.logloss <- data.frame(seed = seed,
            logloss.raw = MultiLogLoss(prob.raw, S2$Label),
            logloss.ENIR = MultiLogLoss(prob.enir.cal, S2$Label),
            logloss.ENIR.full = MultiLogLoss(prob.enir.full.cal, S2$Label),
            logloss.DG = MultiLogLoss(prob.enir.DG.cal, S2$Label),
            logloss.DGG = MultiLogLoss(prob.enir.DGG.cal, S2$Label))
    if (classifier == "RF") {
        df.logloss <- cbind(df.logloss, data.frame(logloss.ENIR.oob = MultiLogLoss(prob.enir.oob.cal, S2$Label)))
    }
    if (control) {
        df.logloss <- cbind(df.logloss, data.frame(logloss.BLR = MultiLogLoss(prob.BLR, S2$Label)))
    }
            

    filename <- paste(name, "_", classifier, "_loglosses.csv", sep = "")
	if (!file.exists(filename)) {
		write.table(df.logloss, file = filename, sep = ";", dec = ",", col.names = TRUE, row.names = FALSE)
	} else {
		write.table(df.logloss, file = filename, append = TRUE, sep = ";", dec = ",", col.names = FALSE, row.names = FALSE)
	}

    # computation times
    df.times <- data.frame(seed = seed,
            time.ENIR = t.ENIR[[3]],
            time.ENIR.full = t.ENIR.full[[3]],
            time.DG = t.DG[[3]] + t.ENIR.DG[[3]],
            time.DGG = t.DG[[3]] + t.DGG[[3]] + t.ENIR.DGG[[3]],
            time.model.all = t.model.all[[3]],
            time.model = t.model[[3]]
            )
    if (classifier == "RF") {
        df.times <- cbind(df.times, data.frame(time.ENIR.oob = t.ENIR.oob[[3]]))
    }
    if (control) {
        df.times <- cbind(df.times, data.frame(time.BLR = t.BLR[[3]]))
    }
            

    filename <- paste(name, "_", classifier, "_times.csv", sep = "")
	if (!file.exists(filename)) {
		write.table(df.times, file = filename, sep = ";", dec = ",", col.names = TRUE, row.names = FALSE)
	} else {
		write.table(df.times, file = filename, append = TRUE, sep = ";", dec = ",", col.names = FALSE, row.names = FALSE)
	}
}