enir.build <- function(posteriors, lbl) {
    # build the ENIR calibration model
    # input:  posteriors: a vector of raw prediction scores
    #         labels: class labels, factor levels 1 and 2 or integers {0,1}
	# output: ENIR model object

    # convert factor levels to integers
    if (class(lbl) == "factor") {
        lbl <- as.integer(lbl) - 1
    }
	df.cal <- data.frame(score = posteriors, label = lbl)

	# remove duplicate entries
	idx <- duplicated(df.cal[,1])
	df.cal <- data.frame(score = df.cal[!idx, 1], label = df.cal[!idx, 2])

	# order in ascending order
	df.cal <- df.cal[with(df.cal, order(score)),]

    # NOTE!
    # this package needs to be the modified version made by Naeini et al. which is included in this repository also
    if(!require(neariso)) {
        install.packages("neariso_1.0.tar.gz", repos = NULL, type = "source")
        require(neariso)
    }

    # build the near isotonic regression model(s)
	mdl.niso <- neariso(df.cal$label, maxBreaks = nrow(df.cal), lambda = NULL) 

	# keep track of the original scores (in the right order) as neariso assumes even distribution
	mdl.niso$scores <- df.cal$score
	mdl.niso$labels <- df.cal$label

	# drop the overfit model (lambda == 0) unless it's the only one
    if (length(mdl.niso$lambda) > 1) {
        mdl.niso$beta <- mdl.niso$beta[, mdl.niso$lambda != 0]
        mdl.niso$df <- mdl.niso$df[mdl.niso$lambda != 0]
        mdl.niso$lambda <- mdl.niso$lambda[mdl.niso$lambda != 0]
    }

    # correct the degrees of freedom estimate(s) to be the number of joined pieces
    if (length(mdl.niso$lambda) == 1) {
        mdl.niso$df <- sum(diff(mdl.niso$beta) != 0) + 1
    } else {
	    mdl.niso$df <- colSums(diff(mdl.niso$beta) != 0) + 1
    }

	# calculate BIC scores for the remaining model(s)
	BICs <- c()
    if (length(mdl.niso$lambda) > 1) {
        for(i in 1:ncol(mdl.niso$beta)) {
            likelihood <- mdl.niso$beta[, i]
            likelihood[mdl.niso$labels == 0] <- 1 - likelihood[mdl.niso$labels == 0] # | prob - correct |
            BICs <- c(BICs, mdl.niso$df[i] * log(nrow(mdl.niso$beta)) - 2 * sum(log(likelihood)))
        }
    } else {
        likelihood <- mdl.niso$beta
        likelihood[mdl.niso$labels == 0] <- 1 - likelihood[mdl.niso$labels == 0] # | prob - correct |
        BICs <- c(BICs, mdl.niso$df * log(length(mdl.niso$beta)) - 2 * sum(log(likelihood)))
    }

	# calculate the relative likelihood of the models
	mdl.niso$relative.likelihood <- exp((min(BICs)-BICs)/2)

    return(mdl.niso)
}

enir.predict <- function(mdl, p) {
    # transform raw prediction scores into probabilities using the ENIR model
    # input:  mdl: near isotonic regression model
    #         p: a vector of raw prediction scores to be calibrated
    # output: a vector of calibrated prediction scores

    pred <- c()

    for (i in 1:length(p)) {
        # is the score to be predicted less than any of the scores in calibration data?
        if (sum(p[i] >= mdl$scores) == 0) {
            # weighted sum of the models' predictions: use the smallest values in the model
            if (length(mdl$lambda) > 1) {
                pred <- c(pred, sum(mdl$beta[1, ] * (mdl$relative.likelihood / sum(mdl$relative.likelihood))))
            } else { # only one model
                pred <- c(pred, mdl$beta[1])
            }

        # or larger than any of the scores in calibration data? use the largest values in the model
        } else if (sum(p[i] >= mdl$scores) == length(mdl$scores)) {
            # weighted sum of the models' predictions
            if (length(mdl$lambda) > 1) {
                pred <- c(pred, sum(mdl$beta[nrow(mdl$beta), ] * (mdl$relative.likelihood / sum(mdl$relative.likelihood))))
            } else { # only one model
                pred <- c(pred, mdl$beta[length(mdl$beta)])
            }

        # otherwise interpolate between calibration training data points
        } else {
            # find the adjacent data points from the model
            id.lower <- max(which(p[i] >= mdl$scores))
            id.upper <- id.lower + 1

            # interpolate: y = a + b*x
            if (length(mdl$lambda) > 1) {
                # weighted sum of the models' predictions
                a <- mdl$beta[id.lower, ]
                b <- (mdl$beta[id.upper, ] - mdl$beta[id.lower, ]) / (mdl$scores[id.upper] - mdl$scores[id.lower])
                pred <- c(pred, sum((a + b * (p[i] - mdl$scores[id.lower])) * (mdl$relative.likelihood / sum(mdl$relative.likelihood))))
            } else { # only one model
                a <- mdl$beta[id.lower]
                b <- (mdl$beta[id.upper] - mdl$beta[id.lower]) / (mdl$scores[id.upper] - mdl$scores[id.lower])
                pred <- c(pred, a + b * (p[i] - mdl$scores[id.lower]))
            }
        }
    }

    return(pred)
}

MSE <- function(predicted, actual) {
    # calculate the mean squared error
	# input:  actual: integers (0 or 1) or factors (with levels 1 or 2)
    #         predicted: predicted probability of the positive class
	# output: MSE

    if (class(actual) == "factor") {
        actual <- as.integer(actual) - 1
    }

    sum(ifelse(actual == 0, predicted^2, (1 - predicted)^2)) / length(predicted)
}

MultiLogLoss <- function(predicted, actual) {
    # calculate the logarithmic loss
    # input:  actual: integers (0 or 1) or factors (with levels 1 or 2)
    #         predicted: predicted probability of the positive class
	# output: logloss

    if (class(actual) == "factor") {
        actual <- as.integer(actual) - 1
    }

    # code modified from kaggle wiki

    predicted <- cbind(1 - predicted, predicted)
    actual <- cbind(ifelse(actual == 0, 1, 0), ifelse(actual == 1, 1, 0))

	eps = 1e-15;
	nr <- nrow(predicted)
	predicted = matrix(sapply( predicted, function(x) max(eps,x)), nrow = nr)      
	predicted = matrix(sapply( predicted, function(x) min(1-eps,x)), nrow = nr)
	ll = sum(actual*log(predicted) + (1-actual)*log(1-predicted))
	ll = ll * -1/(nrow(actual))      
	return(ll);
}