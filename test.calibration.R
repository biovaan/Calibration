test.calibration <- function(data.set, name, method = "CV") {
    # run tests to compare different calibration scenarios
    # input:  data.set: a data set in a data.frame format, last column as the class variable (factor)
    #         name: a string to be used for the result csv files
    #         method: sampling method, 10-fold cross validation ("CV") or 5x2CV ("5x2CV")
    # output: test results in csv files and data set splits in scv files to be used with Matlab implementation of GPC

    source("enir.test.R")

    # using the caret package for preprocessing
    if (!require(caret)) {
        install.packages("caret", dependencies = TRUE)
        require(caret)
    }

    if (method == "CV") {
        # 10-fold CV
        set.seed(20190126)
        folds <- createFolds(data.set[, ncol(data.set)], k = 10)

        for (i in 1:10) {
            S1 <- data.set[-folds[[i]], ]
            S2 <- data.set[folds[[i]], ]

            # preprocess the features to zero mean and unit variance, removing near-zero variance predictors
            pp <- preProcess(data.set, method = c("scale", "center", "nzv"))
            S1 <- predict(pp, S1)
            S2 <- predict(pp, S2)

            for (classifier in c("SVM", "RF", "NN", "NB")) {
                if (classifier == "NB") {
                    # we only need to calculate the control case (Bayesian logistic regression) once
                    enir.mccv.test(S1, S2, paste(name, "_CV", sep = ""), classifier = classifier, control = TRUE, seed = 20190126 + i)
                } else {
                    enir.mccv.test(S1, S2, paste(name, "_CV", sep = ""), classifier = classifier, control = FALSE, seed = 20190126 + i)
                }
            }

            # we will use a Matlab implementation for GPC, save the data sets for later use
            # transform class labels to {-1,1}
            S1[, ncol(S1)] <- ifelse(as.integer(S1[, ncol(S1)]) == 2, 1, -1)
            S2[, ncol(S2)] <- ifelse(as.integer(S2[, ncol(S2)]) == 2, 1, -1)
            filename <- paste(name, "_training_fold_", i, ".csv", sep = "")
            write.table(S1, filename, dec = ".", sep = ",", row.names = FALSE, col.names = FALSE)
            filename <- paste(name, "_test_fold_", i, ".csv", sep = "")
            write.table(S2, filename, dec = ".", sep = ",", row.names = FALSE, col.names = FALSE)
        }
    }

    if (method == "5x2CV") {
        for (i in 1:5) {
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
            # transform class labels to {-1,1}
            S1[, ncol(S1)] <- ifelse(as.integer(S1[, ncol(S1)]) == 2, 1, -1)
            S2[, ncol(S2)] <- ifelse(as.integer(S2[, ncol(S2)]) == 2, 1, -1)
            filename <- paste(name, "_S1_", i, ".csv", sep = "")
            write.table(S1, filename, dec = ".", sep = ",", row.names = FALSE, col.names = FALSE)
            filename <- paste(name, "_S2_", i, ".csv", sep = "")
            write.table(S2, filename, dec = ".", sep = ",", row.names = FALSE, col.names = FALSE)
        }
    }
}