# ======================
## HW1 -- Problem 1 (Comparison of Several KDEs with histogram)
# ======================

# ==========================================
# 1. Data Generation
# ==========================================
set.seed(123)
n <- 500
data <- rnorm(n, mean = 0, sd = 1)

# Print Sample Statistics
cat("Mean:", mean(data), "(True: 0)\n")
cat("Std Dev:", sd(data), "(True: 1)\n\n")

# ==========================================
# 2. Calculate Kernel Density Estimates (KDE)
# ==========================================
# R's base density() function easily handles different kernels
kde_gauss <- density(data, kernel = "gaussian")
kde_epan  <- density(data, kernel = "epanechnikov")
kde_rect  <- density(data, kernel = "rectangular")
kde_tria  <- density(data, kernel = "triangular")

# ==========================================
# 3. Plotting the Results (2x3 Grid)
# ==========================================
# Set up a 2-row, 3-column plot layout
par(mfrow = c(2, 3))

# Helper function to plot histogram + KDE overlay cleanly
plot_kde <- function(kde_obj, title, col_line) {
  hist(data, probability = TRUE, breaks = 30, col = "gray90", border = "white",
       main = title, xlab = "Value", ylim = c(0, 0.5))
  lines(kde_obj, col = col_line, lwd = 2)
  curve(dnorm(x), add = TRUE, col = "black", lwd = 2, lty = 2) # True Normal
}

# Panel 1: Histogram Only
hist(data, probability = TRUE, breaks = 30, col = "gray90", border = "white",
     main = "Histogram Only", xlab = "Value", ylim = c(0, 0.5))
curve(dnorm(x), add = TRUE, col = "black", lwd = 2, lty = 2)

# Panels 2-5: Individual Kernels
plot_kde(kde_gauss, "Kernel: gaussian", "blue")
plot_kde(kde_epan, "Kernel: epanechnikov", "purple")
plot_kde(kde_rect, "Kernel: rectangular", "orange")
plot_kde(kde_tria, "Kernel: triangular", "brown")

# Panel 6: All Kernels Combined
hist(data, probability = TRUE, breaks = 30, col = "gray90", border = "white",
     main = "All Kernels Combined", xlab = "Value", ylim = c(0, 0.5))
lines(kde_gauss, col = "blue", lwd = 2)
lines(kde_epan, col = "purple", lwd = 2)
lines(kde_rect, col = "orange", lwd = 2)
lines(kde_tria, col = "brown", lwd = 2)
curve(dnorm(x), add = TRUE, col = "black", lwd = 2, lty = 2)

# ==========================================
# 4. Calculate Mean Integrated Squared Error (MISE)
# ==========================================
# Helper function to approximate the integral of (f_hat - f)^2
calc_mise_kde <- function(kde_obj) {
  dx <- kde_obj$x[2] - kde_obj$x[1]           # width of intervals
  true_y <- dnorm(kde_obj$x)                  # True density values
  sq_error <- (kde_obj$y - true_y)^2
  sum(sq_error * dx)                          # Riemann sum approximation
}

# Histogram MISE requires building a step-function first
h <- hist(data, breaks = 30, plot = FALSE)
hist_fun <- stepfun(h$breaks, c(0, h$density, 0))
x_val <- seq(min(data) - 1, max(data) + 1, length.out = 2000)
dx_hist <- x_val[2] - x_val[1]
mise_hist <- sum((hist_fun(x_val) - dnorm(x_val))^2 * dx_hist)

# Build and Print the Table
mise_results <- data.frame(
  Method = c("epanechnikov", "triangular", "gaussian", "rectangular", "Histogram"),
  MISE = c(
    calc_mise_kde(kde_epan),
    calc_mise_kde(kde_tria),
    calc_mise_kde(kde_gauss),
    calc_mise_kde(kde_rect),
    mise_hist
  )
)

print("MISE Comparison Table:")
print(mise_results)

par(mfrow = c(1, 1))

# ======================
## HW1 -- Problem 2 (Two errors from Laplace and Normal, NWs and local quantile estimators performances)
# ======================
# ==========================================
# 1. Generate Samples
# ==========================================
library(ggplot2)
library(dplyr)
library(tidyr)
library(knitr)

set.seed(42) # Adding seed for reproducibility

x <- runif(100)

# Y_1: when epsilon follows N(0,1)
eps_norm <- rnorm(100)
y1 <- exp(x) + eps_norm

# Y_2: when epsilon follows Laplace(0,1)
# A Laplace variable can be generated as the difference of two Exponentials
eps_laplace <- rexp(100, rate = 1) - rexp(100, rate = 1) 
y2 <- exp(x) + eps_laplace

df <- data.frame(
  x = rep(x, 2),
  y = c(y1, y2),
  dist = factor(rep(c("Normal", "Laplace"), each = 100))
)

# ==========================================
# 2. Local Linear Smoother Function
# ==========================================
local_linear_smooth <- function(x_eval, x_data, y_data, h) {
  preds <- sapply(x_eval, function(x0) {
    weights <- dnorm((x_data - x0) / h)
    tryCatch({
      fit <- lm(y_data ~ I(x_data - x0), weights = weights)
      return(coef(fit)[1])
    }, error = function(e) NA)
  })
  return(preds)
}

# ==========================================
# 3. Model Fitting
# ==========================================
# Bandwidth selection
h <- 0.1

# Nadaraya-Watson (NW) Estimates using ksmooth
fit_norm <- ksmooth(x, y1, kernel = "normal", bandwidth = h, x.points = x)
fit_lapl <- ksmooth(x, y2, kernel = "normal", bandwidth = h, x.points = x)

preds_df <- data.frame(
  x_grid = c(fit_norm$x, fit_lapl$x),
  y_hat = c(fit_norm$y, fit_lapl$y),
  dist = rep(c("Normal", "Laplace"), each = length(fit_norm$x))
)

# Local Linear Estimates
grid_points <- seq(0, 1, length.out = 200)

y_hat_norm_grid <- local_linear_smooth(grid_points, x, y1, h)
y_hat_norm_orig <- local_linear_smooth(x, x, y1, h)

y_hat_lapl_grid <- local_linear_smooth(grid_points, x, y2, h)
y_hat_lapl_orig <- local_linear_smooth(x, x, y2, h)

preds_df_linear <- data.frame(
  x_grid = rep(grid_points, 2),
  y_hat = c(y_hat_norm_grid, y_hat_lapl_grid),
  dist = rep(c("Normal", "Laplace"), each = length(grid_points))
)

# ==========================================
# 4. MSE Calculation for NW Estimator
# ==========================================
calc_mse <- function(x_orig, y_est_x, y_est_y) {
  y_hat_aligned <- approx(y_est_x, y_est_y, xout = x_orig)$y
  mean((y_hat_aligned - exp(x_orig))^2)
}

mse_norm_nw <- calc_mse(x, fit_norm$x, fit_norm$y)
mse_lapl_nw <- calc_mse(x, fit_lapl$x, fit_lapl$y)

results_nw <- data.frame(
  Distribution = c("Standard Normal", "Standard Laplace"),
  Variance = c("1.0", "2.0"),
  MSE = c(mse_norm_nw, mse_lapl_nw)
)

print("NW Estimator Results:")
print(results_nw)

# ==========================================
# 5. MSE Calculation for Local Linear Estimator
# ==========================================
# Calculate MSE directly using the predictions at observed X points
mse_norm_ll <- mean((y_hat_norm_orig - exp(x))^2)
mse_lapl_ll <- mean((y_hat_lapl_orig - exp(x))^2)

results_ll <- data.frame(
  Distribution = c("Standard Normal", "Standard Laplace"),
  Variance = c("1.0", "2.0"),
  MSE = c(mse_norm_ll, mse_lapl_ll)
)

print("Local Linear Estimator Results:")
print(results_ll)
