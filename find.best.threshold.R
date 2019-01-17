find.best.threshold <- function(predictions) {
	# find a threshold value that maximizes classification rate
	# implemented based on Lachiche & Flach (2003)
	# input:  predictions data.frame(label (true), score (positive class))
	# output: best.threshold
	
	# initialize optimum as if all samples were predicted to negative class
	# negative class is factor level 1
	# positive class is factor level 2
	optimum <- sum(as.integer(predictions$label) == 1) / nrow(predictions)
	current <- optimum
	
	predictions <- predictions[with(predictions, order(score, decreasing = TRUE)),]
	
	# initialize best.threshold as the highest score
	best.threshold <- predictions$score[1]
	
	for (i in 2:(nrow(predictions))) {
		# classification rate if all scores > current score are classified as positive class
		current <- sum(as.integer(predictions$label) == ifelse(predictions$score > predictions$score[i], 2, 1)) / nrow(predictions)
		# check if classification rate improved by changing the threshold
		if (current > optimum) {
			optimum <- current
			# choose the threshold between adjacent scores
			best.threshold <- mean(c(predictions$score[i], predictions$score[i-1]))
		}
	}
	
	return(best.threshold)
}