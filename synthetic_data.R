# define the normal distribution to draw samples from
mu1 <- c(3.3, 5.2)
s1 <- diag(2) * 0.4
mu2 <- c(6.3, 6.5)
s2 <- diag(2) * 0.3
mu3 <- c(5.2, 4.5)
s3 <- diag(2) * 2

# draw a random sample from the distributions
x1 <- rmvnorm(50, mean = mu1, sigma = s1)
x2 <- rmvnorm(50, mean = mu2, sigma = s2)
x3 <- rmvnorm(100, mean = mu3, sigma = s3)

# visualize
# plot(x3, xlim = c(0,10), ylim = c(0,10), xlab = "X1", ylab = "X2", col = "green")
# points(x1, xlim = c(0,10), ylim = c(0,10), xlab = "X1", ylab = "X2", col = "red")
# points(x2, xlim = c(0,10), ylim = c(0,10), xlab = "X1", ylab = "X2", col = "red")

bin_synth <- rbind(data.frame(x1 = x1[, 1], x2 = x1[, 2], y = 1),
    data.frame(x1 = x2[, 1], x2 = x2[, 2], y = 1),
    data.frame(x1 = x3[, 1], x2 = x3[, 2], y = 2)
    )
bin_synth$y <- factor(bin_synth$y, labels = c("Neg", "Pos"))

# add engineered features
bin_synth$x3 <- sin(bin_synth$x1 * pi)
bin_synth$x4 <- sin(bin_synth$x2 * pi)
bin_synth$x5 <- (bin_synth$x1 + 1)^2
bin_synth$x6 <- (bin_synth$x2 + 1)^2
bin_synth$x7 <- 1 / (1 + bin_synth$x1)
bin_synth$x8 <- 1 / (1 + bin_synth$x2)

# calculate the true probabilities for each instance in the sample
probs <- dmvnorm(bin_synth[, c(1,2)], mu3, s3) / (dmvnorm(bin_synth[, c(1,2)], mu1, s1) + dmvnorm(bin_synth[, c(1,2)], mu2, s2) + dmvnorm(bin_synth[, c(1,2)], mu3, s3))

# remove original features and reorder the columns so that class label is last
bin_synth <- bin_synth[, c(4,5,6,7,8,9,3)]