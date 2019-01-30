test.calibration <- function(data.set, name) {
    source("enir.test.R")
    for (i in 1:5) {
        # using the caret package for preprocessing
        if (!require(caret)) {
            install.packages("caret", dependencies = TRUE)
            require(caret)
        }

        # we are doing 5x2CV
        set.seed(20190126 + i)
        S1.idx <- sample(1:nrow(data.set), floor(nrow(data.set)*0.5))
	    S1 <- data.set[S1.idx, ]
        S2 <- data.set[-S1.idx, ]

        # preprocess the features to zero mean and unit variance, removing near-zero variance predictors
        pp <- preProcess(data.set, method = c("scale", "center", "nzv"))
        S1 <- predict(pp, S1)
        S2 <- predict(pp, S2)

        for (classifier in c("SVM", "RF", "NN", "NB")) {
            if (classifier == "NB") {
                # we only need to calculate the control case (Bayesian logistic regression) once
                # 5x2CV fold 1
                enir.mccv.test(S1, S2, paste(name, "_S1", sep = ""), classifier = classifier, control = TRUE, seed = 20190126 + i)
                # and fold 2
                enir.mccv.test(S2, S1, paste(name, "_S2", sep = ""), classifier = classifier, control = TRUE, seed = 20190126 + i)
            } else {
                # 5x2CV fold 1
                enir.mccv.test(S1, S2, paste(name, "_S1", sep = ""), classifier = classifier, control = FALSE, seed = 20190126 + i)
                # and fold 2
                enir.mccv.test(S2, S1, paste(name, "_S2", sep = ""), classifier = classifier, control = FALSE, seed = 20190126 + i)
            }
        }

        # we will use a Matlab implementation for GPC, save the data sets for later use
        S1[, ncol(S1)] <- ifelse(as.integer(S1[, ncol(S1)]) == 2, 1, -1)
        S2[, ncol(S2)] <- ifelse(as.integer(S2[, ncol(S2)]) == 2, 1, -1)
        filename <- paste(name, "_S1_", i, ".csv", sep = "")
        write.table(S1, filename, dec = ".", sep = ",", row.names = FALSE, col.names = FALSE)
        filename <- paste(name, "_S2_", i, ".csv", sep = "")
        write.table(S2, filename, dec = ".", sep = ",", row.names = FALSE, col.names = FALSE)
    }
}