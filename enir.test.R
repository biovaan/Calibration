enir.mccv.test <- function(data.set, name, classifier = "NB", control = FALSE, seed = 20180822, Nsamples = 5000, bin.size = 100) {
    # run tests that test the usefulness of DG and DGG algorithms for small data set calibration
    # input:  data.set: two-class data set where the last column is the label {0,1}
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
    if (!require(arm)) {
        install.packages("arm", dependencies = TRUE)
        require(arm)
    }
    if (!require(nnet)) {
        install.packages("nnet", dependencies = TRUE)
        require(nnet)
    }

	Label <- ncol(data.set)
	N.var <- Label - 1
    names(data.set)[Label] <- "Label"

    source("enir.R")
    source("find.best.threshold.R")

    set.seed(seed)

    # divide the data set to training, test, and calibration data sets
    # we will use the whole training data (training.data.raw) with ENIR full, DG, and DGG
    # a separate calibration data is split off for ENIR calibration 
    # and the rest (training.data) is used for training the classifier
    tr.idx <- sample(1:nrow(data.set), floor(nrow(data.set)*0.7))
	training.data.raw <- data.set[tr.idx, ]
    cal.idx <- sample(1:nrow(training.data.raw), floor(nrow(training.data.raw)*0.1))
    calibration.data <- training.data.raw[cal.idx, ]
    training.data <- training.data.raw[-cal.idx, ]
	test.data <- data.set[-tr.idx, ]

    # make sure we don't have empty classes
    training.data.raw$Label <- factor(training.data.raw$Label)
    training.data$Label <- factor(training.data$Label)
    test.data$Label <- factor(test.data$Label)

    # parallel back end for hyperparameter tuning
    # svm has 14 different hyperparameters to test
    cl <- makeCluster(14)
    registerDoParallel(cl)
    clusterExport(cl, varlist = c("training.data.raw", "gausspr", "cross", "svm"), envir=environment())

    if (control) {
        # CASE 0: control

        # Bayesian logistic regression

        t.BGLM <- system.time({
        mdl.bglm <- bayesglm(Label ~., data = training.data.raw, family = binomial(link = "logit"),
            drop.unused.levels = FALSE)
        # predict the training data so that we can find out the optimal threshold
        pred.bglm.tr <- data.frame(label = training.data.raw$Label, score = predict(mdl.bglm, training.data.raw,
            type = "response"))

        # raw scores for the test data
        prob.bglm <- predict(mdl.bglm, test.data, type = "response")
        # find a threshold to maximize classification rate
        threshold.bglm <- find.best.threshold(pred.bglm.tr)
        # predict the test set with using that threshold
        pred.bglm <- ifelse(prob.bglm > threshold.bglm, levels(test.data$Label)[2], levels(test.data$Label)[1])
        levels(pred.bglm) <- levels(test.data$Label) # make sure we have all levels in the predictions
        })

        # Gaussian process
        # tune kernel spread
        t.GPR <- system.time({
        errs <- foreach(sig = 10^(-2:7), combine = c) %dopar% {
            mdl <- gausspr(Label ~., data = training.data.raw, kernel = "rbfdot", kpar = list(sigma = sig), cross = 10)
            cross(mdl)
        }

        best.sigma <- 10^(which.min(errs) - 3)
        mdl.gpr <- gausspr(Label ~., data = training.data.raw, kernel = "rbfdot", kpar = list(sigma = best.sigma))
        # predict the training data so that we can find out the optimal threshold
        pred.gpr.tr <- data.frame(label = training.data.raw$Label, score = predict(mdl.gpr, training.data.raw,
            type = "probabilities")[, 2])

        # raw scores for the test data
        prob.gpr <- predict(mdl.gpr, test.data, type = "probabilities")[, 2]
        # find a threshold to maximize classification rate
        threshold.gpr <- find.best.threshold(pred.gpr.tr)
        # predict the test set with using that threshold
        pred.gpr <- ifelse(prob.gpr > threshold.gpr, levels(test.data$Label)[2], levels(test.data$Label)[1])
        levels(pred.gpr) <- levels(test.data$Label) # make sure we have all levels in the predictions
        })
    }

    # CASE 1: no calibration

    # train the classifier model
    # use the full training data

    if (classifier == "NB") {
        t.model.all <- system.time({
        mdl.raw <- naiveBayes(Label ~., data = training.data.raw)
        })
    } else if (classifier == "SVM") {
        t.model.all <- system.time({
        # tune SVM hyperparameters
		gammas <- sigest(Label ~., data = training.data.raw)

		accs <- foreach (cost = 2^(-2:11), combine = c) %dopar% {
			mdl <- svm(Label ~., data = training.data.raw, cost = cost, gamma = gammas[2], cross = 10)
			mdl$tot.accuracy
		}

		best.cost <- 2^(which.max(accs) - 3)

		mdl.raw <- svm(Label ~., data = training.data.raw, probability = TRUE, type = "C-classification",
            cost = best.cost, gamma = gammas[2])
        })

    } else if (classifier == "RF") {
        t.model.all <- system.time({
        # tune RF hyperparameters
		tune.res <- tuneRF(training.data.raw[, 1:N.var], training.data.raw[, (N.var+1)], ntreeTry = 500,
            plot = FALSE, trace = FALSE)
        mtry.tuned.raw <- tune.res[which.min(tune.res[, 2]), 1]

		mdl.raw <- randomForest(Label ~., data = training.data.raw, mtry = mtry.tuned.raw)
        })

    } else if (classifier == "NN") {
        t.model.all <- system.time({
        # tune NN hyperparameters (10-fold cv = default)
        tune.obj.raw <- tune.nnet(Label~., data = training.data.raw, size = c(1, 3, 5, 7, 9),
            decay = c(0, 0.1, 0.01, 0.001, 0.0001), maxit = 200, trace = FALSE)

		mdl.raw <- tune.obj.raw$best.model
        })
    }

    # find a threshold to maximize classification rate
    if (classifier == "NB") {
        pred.raw.tr <- data.frame(label = training.data.raw$Label, score = predict(mdl.raw, training.data.raw,
            type = "raw")[, 2])

        # raw scores for the test data
        prob.raw <- predict(mdl.raw, test.data, type = "raw")[, 2]

    } else if (classifier == "SVM") {
        p <- predict(mdl.raw, training.data.raw, probability = TRUE)
        
        # find which column of the prediction scores is related to the positive class
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(training.data.raw$Label)[2])

        pred.raw.tr <- data.frame(label = training.data.raw$Label, score = attr(p, "probabilities")[, col.pos])

        # raw scores for the test data
        prob.raw <- attr(predict(mdl.raw, test.data, probability = TRUE), "probabilities")[, col.pos]
    } else if (classifier == "RF") {
        pred.raw.tr <- data.frame(label = training.data.raw$Label, score = predict(mdl.raw, training.data.raw,
            type = "prob")[, 2])

        # raw scores for the test data
        prob.raw <- predict(mdl.raw, test.data, type = "prob")[, 2]
    } else if (classifier == "NN") {
        pred.raw.tr <- data.frame(label = training.data.raw$Label, score = predict(mdl.raw, training.data.raw)[, 1])

        # raw scores for the test data
        prob.raw <- predict(mdl.raw, test.data)[, 1]
    }

    threshold.raw <- find.best.threshold(pred.raw.tr)
    # predict the test set with using that threshold
    pred.raw <- ifelse(prob.raw > threshold.raw, levels(test.data$Label)[2], levels(test.data$Label)[1])
    levels(pred.raw) <- levels(test.data$Label) # make sure we have all levels in the predictions

    # CASE 2: ENIR calibration with a separate calibration data set

    # train the classifier model
    # use the training data minus the calibration data

    if (classifier == "NB") {
        t.model <- system.time({
        mdl <- naiveBayes(Label ~., data = training.data)
        })
    } else if (classifier == "SVM") {
        t.model <- system.time({
        # tune SVM hyperparameters
		gammas <- sigest(Label ~., data = training.data)

		accs <- foreach (cost = 2^(-2:11), combine = c) %dopar% {
			mdl <- svm(Label ~., data = training.data, cost = cost, gamma = gammas[2], cross = 10)
			mdl$tot.accuracy
		}

		best.cost <- 2^(which.max(accs) - 3)

		mdl <- svm(Label ~., data = training.data, probability = TRUE, type = "C-classification",
            cost = best.cost, gamma = gammas[2])
        })
    } else if (classifier == "RF") {
        t.model <- system.time({
        # tune RF hyperparameters
		tune.res <- tuneRF(training.data[, 1:N.var], training.data[, (N.var+1)], ntreeTry = 500,
            plot = FALSE, trace = FALSE)
        mtry.tuned <- tune.res[which.min(tune.res[, 2]), 1]

		mdl <- randomForest(Label ~., data = training.data, mtry = mtry.tuned)
        })
    } else if (classifier == "NN") {
        t.model <- system.time({
        # tune NN hyperparameters (10-fold cv = default)
        tune.obj <- tune.nnet(Label~., data = training.data, size = c(1, 3, 5, 7, 9),
            decay = c(0, 0.1, 0.01, 0.001, 0.0001), maxit = 200, trace = FALSE)

		mdl <- tune.obj$best.model
        })
    }

    # predict the calibration data, this is used to tune the calibration model
    if (classifier == "NB") {
        pred.enir.cal <- data.frame(label = calibration.data$Label, score = predict(mdl, calibration.data,
            type = "raw")[, 2])
    } else if (classifier == "SVM") {
        p <- predict(mdl, calibration.data, probability = TRUE)

        # find which column of the prediction scores is related to the positive class:
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(calibration.data$Label)[2])

        pred.enir.cal <- data.frame(label = calibration.data$Label, score = attr(p, "probabilities")[, col.pos])
    } else if (classifier == "RF") {
        pred.enir.cal <- data.frame(label = calibration.data$Label, score = predict(mdl, calibration.data,
            type = "prob")[, 2])
    } else if (classifier == "NN") {
        pred.enir.cal <- data.frame(label = calibration.data$Label, score = predict(mdl, calibration.data)[, 1])
    }

    # tune the calibration model
    t.ENIR <- system.time({
    mdl.enir <- enir.build(pred.enir.cal$score, pred.enir.cal$label)
    })

    # find a threshold to maximize classification rate
    if (classifier == "NB") {
        pred.enir.tr.cal <- data.frame(label = training.data$Label,
            score = enir.predict(mdl.enir, predict(mdl, training.data, type = "raw")[, 2]))

        # raw scores for the test data
        prob.enir.raw <- predict(mdl, test.data, type = "raw")[, 2]

    } else if (classifier == "SVM") {
        p <- predict(mdl, training.data, probability = TRUE)

        # find which column of the prediction scores is related to the positive class:
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(training.data$Label)[2])

        pred.enir.tr.cal <- data.frame(label = training.data$Label,
            score = enir.predict(mdl.enir, attr(p, "probabilities")[, col.pos]))

        # raw scores for the test data
        prob.enir.raw <- attr(predict(mdl, test.data, probability = TRUE), "probabilities")[, col.pos]
    } else if (classifier == "RF") {
        pred.enir.tr.cal <- data.frame(label = training.data$Label,
            score = enir.predict(mdl.enir, predict(mdl, training.data, type = "prob")[, 2]))

        # raw scores for the test data
        prob.enir.raw <- predict(mdl, test.data, type = "prob")[, 2]
    } else if (classifier == "ELM") {
        pred.enir.tr.cal <- data.frame(label = training.data$Label,
            score = enir.predict(mdl.enir, elm_predict(mdl, as.matrix(scaled.training.data[, 1:N.var]),
            normalize = TRUE)[, 2]))

        # raw scores for the test data
        scaled.test.data <- scale.features(test.data, scaling.params)
        prob.enir.raw <- elm_predict(mdl, as.matrix(scaled.test.data[, 1:N.var]), normalize = TRUE)[, 2]
    } else if (classifier == "NN") {
        pred.enir.tr.cal <- data.frame(label = training.data$Label,
            score = enir.predict(mdl.enir, predict(mdl, training.data)[, 1]))

        # raw scores for the test data
        prob.enir.raw <- predict(mdl, test.data)[, 1]
    }

    threshold.enir <- find.best.threshold(pred.enir.tr.cal)
    # calibrate the raw prediction scores
    prob.enir.cal <- enir.predict(mdl.enir, prob.enir.raw)
    # predict the test set with using that threshold
    pred.enir <- ifelse(prob.enir.cal > threshold.enir, levels(test.data$Label)[2], levels(test.data$Label)[1])

    levels(pred.enir) <- levels(test.data$Label) # make sure we have all levels in the predictions

    # we don't use the parallel back end anymore
    stopCluster(cl)

    # CASE 2b: ENIR calibration without separate calibration data (use the same data for classifier training and calibration)

    # we'll be using the mdl.raw trained above

    # calibrate the model using the full training data set
    if (classifier == "NB") {
        pred.enir.full.cal <- data.frame(label = training.data.raw$Label,
            score = predict(mdl.raw, training.data.raw, type = "raw")[, 2])
    } else if (classifier == "SVM") {
        p <- predict(mdl.raw, training.data.raw, probability = TRUE)

        # find which column of the prediction scores is related to the positive class:
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(training.data.raw$Label)[2])

        pred.enir.full.cal <- data.frame(label = training.data.raw$Label, score = attr(p, "probabilities")[, col.pos])
    } else if (classifier == "RF") {
        pred.enir.full.cal <- data.frame(label = training.data.raw$Label,
            score = predict(mdl.raw, training.data.raw, type = "prob")[, 2])
    } else if (classifier == "NN") {
        pred.enir.full.cal <- data.frame(label = training.data.raw$Label,
            score = predict(mdl.raw, training.data.raw)[, 1])
    }

    # tune the calibration model
    t.ENIR.full <- system.time({
    mdl.enir.full <- enir.build(pred.enir.full.cal$score, pred.enir.full.cal$label)
    })

    # custom threshold
    if (classifier == "NB") {
        pred.enir.tr.full.cal <- data.frame(label = training.data.raw$Label,
            score = enir.predict(mdl.enir.full, predict(mdl.raw, training.data.raw, type = "raw")[, 2]))

        # raw scores for the test data
        prob.enir.full.raw <- predict(mdl.raw, test.data, type = "raw")[, 2]

    } else if (classifier == "SVM") {
        p <- predict(mdl.raw, training.data.raw, probability = TRUE)

        # find which column of the prediction scores is related to the positive class:
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(training.data.raw$Label)[2])

        pred.enir.tr.full.cal <- data.frame(label = training.data.raw$Label,
            score = enir.predict(mdl.enir.full, attr(p, "probabilities")[, col.pos]))

        # raw scores for the test data
        prob.enir.full.raw <- attr(predict(mdl.raw, test.data, probability = TRUE), "probabilities")[, col.pos]
    } else if (classifier == "RF") {
        pred.enir.tr.full.cal <- data.frame(label = training.data.raw$Label,
            score = enir.predict(mdl.enir.full, predict(mdl.raw, training.data.raw, type = "prob")[, 2]))

        # raw scores for the test data
        prob.enir.full.raw <- predict(mdl.raw, test.data, type = "prob")[, 2]
    } else if (classifier == "NN") {
        pred.enir.tr.full.cal <- data.frame(label = training.data.raw$Label,
            score = enir.predict(mdl.enir.full, predict(mdl.raw, training.data.raw)[, 1]))

        # raw scores for the test data
        prob.enir.full.raw <- predict(mdl.raw, test.data)[, 1]
    }

    threshold.enir.full <- find.best.threshold(pred.enir.tr.full.cal)
    # calibrate the raw prediction scores
    prob.enir.full.cal <- enir.predict(mdl.enir.full, prob.enir.full.raw)
    # predict the test set with using that threshold
    pred.enir.full <- ifelse(prob.enir.full.cal > threshold.enir.full, levels(test.data$Label)[2],
        levels(test.data$Label)[1])

    levels(pred.enir.full) <- levels(test.data$Label) # make sure we have all levels in the predictions

    # CASE 3: DG + ENIR calibration

    # use mdl.raw trained above

    t.DG <- system.time({

    # generate calibration data with Monte Carlo CV
    pred.tr <- factor()
	tst.Y <- factor()
	posteriors <- numeric()

    # we need "iter" iterations to get at least Nsamples of calibrations data
	iter <- ceiling(Nsamples / floor(nrow(training.data.raw)*0.3))

	for (i in 1:iter) {
		tr <- sample(1:nrow(training.data.raw), floor(nrow(training.data.raw)*0.7))
		trset <- training.data.raw[tr, ]
		tstset <- training.data.raw[-tr, ]

		if (classifier == "NB") {
            mdl.DG <- naiveBayes(Label ~., data = trset)
            pred.probs.tr <- predict(mdl.DG, tstset[,1:N.var], type = "raw")[, 2]
            pred.tr <- unlist(list(pred.tr, predict(mdl.DG, tstset[, 1:N.var])))
		} else if (classifier == "SVM") {
			# use the same hyperparameters as with the whole data set
			mdl.DG <- svm(Label ~., data = trset, probability = TRUE, type = "C-classification", cost = best.cost,
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
        pred.enir.tr.DG.cal <- data.frame(label = training.data.raw$Label,
            score = enir.predict(mdl.enir.DG, predict(mdl.raw, training.data.raw, type = "raw")[, 2]))

        # raw scores for the test data
        prob.enir.DG.raw <- predict(mdl.raw, test.data, type = "raw")[, 2]

    } else if (classifier == "SVM") {
        p <- predict(mdl.raw, training.data.raw, probability = TRUE)

        # find which column of the prediction scores is related to the positive class:
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(training.data.raw$Label)[2])

        pred.enir.tr.DG.cal <- data.frame(label = training.data.raw$Label,
            score = enir.predict(mdl.enir.DG, attr(p, "probabilities")[, col.pos]))

        # raw scores for the test data
        prob.enir.DG.raw <- attr(predict(mdl.raw, test.data, probability = TRUE), "probabilities")[, col.pos]
    } else if (classifier == "RF") {
        pred.enir.tr.DG.cal <- data.frame(label = training.data.raw$Label,
            score = enir.predict(mdl.enir.DG, predict(mdl.raw, training.data.raw, type = "prob")[, 2]))

        # raw scores for the test data
        prob.enir.DG.raw <- predict(mdl.raw, test.data, type = "prob")[, 2]
    } else if (classifier == "NN") {
        pred.enir.tr.DG.cal <- data.frame(label = training.data.raw$Label,
            score = enir.predict(mdl.enir.DG, predict(mdl.raw, training.data.raw)[, 1]))

        # raw scores for the test data
        prob.enir.DG.raw <- predict(mdl.raw, test.data)[, 1]
    }

    threshold.enir.DG <- find.best.threshold(pred.enir.tr.DG.cal)
    # calibrate the raw prediction scores
    prob.enir.DG.cal <- enir.predict(mdl.enir.DG, prob.enir.DG.raw)
    # predict the test set with using that threshold
    pred.enir.DG <- ifelse(prob.enir.DG.cal > threshold.enir.DG, levels(test.data$Label)[2], levels(test.data$Label)[1])

    levels(pred.enir.DG) <- levels(test.data$Label) # make sure we have all levels in the predictions    

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
        pred.enir.tr.DGG.cal <- data.frame(label = training.data.raw$Label,
            score = enir.predict(mdl.enir.DGG, predict(mdl.raw, training.data.raw, type = "raw")[, 2]))

        # raw scores for the test data
        prob.enir.DGG.raw <- predict(mdl.raw, test.data, type = "raw")[, 2]

    } else if (classifier == "SVM") {
        p <- predict(mdl.raw, training.data.raw, probability = TRUE)

        # find which column of the prediction scores is related to the positive class:
		col.pos <- which(labels(attr(p, "probabilities"))[[2]] == levels(training.data.raw$Label)[2])

        pred.enir.tr.DGG.cal <- data.frame(label = training.data.raw$Label,
            score = enir.predict(mdl.enir.DGG, attr(p, "probabilities")[, col.pos]))

        # raw scores for the test data
        prob.enir.DGG.raw <- attr(predict(mdl.raw, test.data, probability = TRUE), "probabilities")[, col.pos]
    } else if (classifier == "RF") {
        pred.enir.tr.DGG.cal <- data.frame(label = training.data.raw$Label,
            score = enir.predict(mdl.enir.DGG, predict(mdl.raw, training.data.raw, type = "prob")[, 2]))

        # raw scores for the test data
        prob.enir.DGG.raw <- predict(mdl.raw, test.data, type = "prob")[, 2]
    } else if (classifier == "NN") {
        pred.enir.tr.DGG.cal <- data.frame(label = training.data.raw$Label,
            score = enir.predict(mdl.enir.DGG, predict(mdl.raw, training.data.raw)[, 1]))

        # raw scores for the test data
        prob.enir.DGG.raw <- predict(mdl.raw, test.data)[, 1]
    }

    threshold.enir.DGG <- find.best.threshold(pred.enir.tr.DGG.cal)
    # calibrate the raw prediction scores
    prob.enir.DGG.cal <- enir.predict(mdl.enir.DGG, prob.enir.DGG.raw)
    # predict the test set with using that threshold
    pred.enir.DGG <- ifelse(prob.enir.DGG.cal > threshold.enir.DGG, levels(test.data$Label)[2], levels(test.data$Label)[1])

    levels(pred.enir.DGG) <- levels(test.data$Label) # make sure we have all levels in the predictions    


    # compare the results!

    #cat(sprintf("%s\nAccuracy\nRaw: %.2f\n", classifier, 100 * sum(pred.raw == test.data$Label) / nrow(test.data)))
    #cat(sprintf("ENIR: %.2f\n", 100 * sum(pred.enir == test.data$Label) / nrow(test.data)))
    #cat(sprintf("ENIR full: %.2f\n", 100 * sum(pred.enir.full == test.data$Label) / nrow(test.data)))
    #cat(sprintf("ENIR DG: %.2f\n", 100 * sum(pred.enir.DG == test.data$Label) / nrow(test.data)))
    #cat(sprintf("ENIR DGG: %.2f\n", 100 * sum(pred.enir.DGG == test.data$Label) / nrow(test.data)))

    #cat(sprintf("MSE\nRaw: %.3f\n", MSE(prob.raw, test.data$Label)))
    #cat(sprintf("ENIR: %.3f\n", MSE(prob.enir.cal, test.data$Label)))
    #cat(sprintf("ENIR full: %.3f\n", MSE(prob.enir.full.cal, test.data$Label)))
    #cat(sprintf("ENIR DG: %.3f\n", MSE(prob.enir.DG.cal, test.data$Label)))
    #cat(sprintf("ENIR DGG: %.3f\n", MSE(prob.enir.DGG.cal, test.data$Label)))

    #cat(sprintf("Logloss\nRaw: %.3f\n", MultiLogLoss(prob.raw, test.data$Label)))
    #cat(sprintf("ENIR: %.3f\n", MultiLogLoss(prob.enir.cal, test.data$Label)))
    #cat(sprintf("ENIR full: %.3f\n", MultiLogLoss(prob.enir.full.cal, test.data$Label)))
    #cat(sprintf("ENIR DG: %.3f\n", MultiLogLoss(prob.enir.DG.cal, test.data$Label)))
    #cat(sprintf("ENIR DGG: %.3f\n", MultiLogLoss(prob.enir.DGG.cal, test.data$Label)))

    # combine the results into data frames that we can then write into csv files
    # append if the files exists so that we can run several simulation rounds

    # accuracies (classification rates)
    df.acc <- data.frame(seed = seed,
            CR.raw = 100 * sum(pred.raw == test.data$Label) / nrow(test.data),
            CR.ENIR = 100 * sum(pred.enir == test.data$Label) / nrow(test.data),
            CR.ENIR.full = 100 * sum(pred.enir.full == test.data$Label) / nrow(test.data),
            CR.DG = 100 * sum(pred.enir.DG == test.data$Label) / nrow(test.data),
            CR.DGG = 100 * sum(pred.enir.DGG == test.data$Label) / nrow(test.data))
    if (control) {
        df.acc <- cbind(df.acc, data.frame(CR.BGLM = 100 * sum(pred.bglm == test.data$Label) / nrow(test.data),
            CR.GPR = 100 * sum(pred.gpr == test.data$Label) / nrow(test.data)))
    }

    filename <- paste(name, "_", classifier, "_accuracies.csv", sep = "")
	if (!file.exists(filename)) {
		write.table(df.acc, file = filename, sep = ";", dec = ",", col.names = TRUE, row.names = FALSE)
	} else {
		write.table(df.acc, file = filename, append = TRUE, sep = ";", dec = ",", col.names = FALSE, row.names = FALSE)
	}

    # mean squared errors
    df.MSE <- data.frame(seed = seed,
            MSE.raw = MSE(prob.raw, test.data$Label),
            MSE.ENIR = MSE(prob.enir.cal, test.data$Label),
            MSE.ENIR.full = MSE(prob.enir.full.cal, test.data$Label),
            MSE.DG = MSE(prob.enir.DG.cal, test.data$Label),
            MSE.DGG = MSE(prob.enir.DGG.cal, test.data$Label))
    if (control) {
        df.MSE <- cbind(df.MSE, data.frame(MSE.BGLM = MSE(prob.bglm, test.data$Label),
            MSE.GPR = MSE(prob.gpr, test.data$Label)))
    }

    filename <- paste(name, "_", classifier, "_MSEs.csv", sep = "")
	if (!file.exists(filename)) {
		write.table(df.MSE, file = filename, sep = ";", dec = ",", col.names = TRUE, row.names = FALSE)
	} else {
		write.table(df.MSE, file = filename, append = TRUE, sep = ";", dec = ",", col.names = FALSE, row.names = FALSE)
	}

    # logarithmic losses
    df.logloss <- data.frame(seed = seed,
            logloss.raw = MultiLogLoss(prob.raw, test.data$Label),
            logloss.ENIR = MultiLogLoss(prob.enir.cal, test.data$Label),
            logloss.ENIR.full = MultiLogLoss(prob.enir.full.cal, test.data$Label),
            logloss.DG = MultiLogLoss(prob.enir.DG.cal, test.data$Label),
            logloss.DGG = MultiLogLoss(prob.enir.DGG.cal, test.data$Label))
    if (control) {
        df.logloss <- cbind(df.logloss, data.frame(logloss.BGLM = MultiLogLoss(prob.bglm, test.data$Label),
            logloss.GPR = MultiLogLoss(prob.gpr, test.data$Label)))
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
    if (control) {
        df.times <- cbind(df.times, data.frame(time.BGLM = t.BGLM[[3]], time.GPR = t.GPR[[3]]))
    }
            

    filename <- paste(name, "_", classifier, "_times.csv", sep = "")
	if (!file.exists(filename)) {
		write.table(df.times, file = filename, sep = ";", dec = ",", col.names = TRUE, row.names = FALSE)
	} else {
		write.table(df.times, file = filename, append = TRUE, sep = ";", dec = ",", col.names = FALSE, row.names = FALSE)
	}
}