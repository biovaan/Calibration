test.calibration <- function(dataset, name) {
    source("enir.test.R")
    for (classifier in c("SVM", "RF", "NN", "NB")) {
        for (i in 1:10) {
            if (classifier == "NB") {
                # we only need to calculate the control case once
                enir.mccv.test(dataset, name, classifier = classifier, control = TRUE, seed = i)
            } else {
                enir.mccv.test(dataset, name, classifier = classifier, control = FALSE, seed = i)
            }
        }
    }
}