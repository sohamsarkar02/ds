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




# ======================
## HW 2-- Problem 1 (regression quantile lines for various choices of quantile indexes)
# ======================

# install.packages(c("quantreg", "ggplot2", "dplyr", "tidyr"))

library(quantreg)
library(ggplot2)
library(dplyr)
library(tidyr)

# 1. Input the mock exam dataset (replaces read.csv)
data <- data.frame(
  X1 = c(1.0, 2.0, 1.5, 3.0, 2.5, 2.0, 10.0),
  X2 = c(2.5, 2.0, 3.0, 2.5, 4.0, 2.5, 12.0),
  Y  = c(5.2, 6.1, 7.4, 8.8, 11.1, 25.0, 10.5)
)

data_long <- data %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "value")

boxplot <- ggplot(data_long, aes(x = feature, y = value)) +
  geom_boxplot(fill = "tomato", outlier.colour = "black", outlier.shape = 16) +
  facet_wrap(~feature, scales = "free") +
  theme_minimal() +
  theme(axis.text.x = element_blank()) +
  labs(title = "Boxplots for Outlier Identification", y = "Value", x = "")
boxplot

# OLS regression model
ols_model <- lm(Y ~ ., data = data)

# Quantile Regression for various quantile indexes
taus <- c(0.10, 0.50, 0.90)

qr_models <- list()
for (tau in taus) {
  qr_models[[paste0("tau_", tau)]] <- rq(
    Y ~ .,
    tau  = tau,
    data = data
  )
}

# Build coefficients table
coef_df <- data.frame(Variable = names(coef(qr_models[[1]])))
for (tau in taus) {
  coef_df <- cbind(coef_df, round(coef(qr_models[[paste0("tau_", tau)]]), 4))
}
names(coef_df)[2:ncol(coef_df)] <- paste0("tau=", taus)

# Plot: Quantile regression lines: X1 vs Y

new_data <- data.frame(
  X1 = seq(min(data$X1), max(data$X1), length.out = 200),
  X2 = median(data$X2) # Hold the second predictor at its median
)

predictions <- data.frame()
for (tau in taus) {
  pred <- predict(qr_models[[paste0("tau_", tau)]], newdata = new_data)
  predictions <- rbind(predictions,
                       data.frame(X1  = new_data$X1,
                                  Y   = pred,
                                  tau = factor(tau)))
}

p1 <- ggplot() +
  geom_point(data = data,
             aes(x = X1, y = Y),
             color = "black", size = 2.5) + 
  geom_line(data = predictions,
            aes(x = X1, y = Y, color = tau),
            linewidth = 1) +
  scale_color_viridis_d(name = "Quantile (tau)") +
  labs(title    = "Regression Quantile Lines",
       subtitle = "X2 held at its median",
       x = "X1",
       y = "Response (Y)") +
  theme_minimal() +
  theme(legend.position = "right",
        plot.title = element_text(face = "bold"))

# Plot: OLS vs Median Quantile Regression

ols_pred    <- predict(ols_model, newdata = new_data)
median_pred <- predict(qr_models$tau_0.5, newdata = new_data)

comparison <- data.frame(
  X1        = new_data$X1,
  OLS       = ols_pred,
  Median_QR = median_pred
) %>% pivot_longer(-X1, names_to = "Method", values_to = "Y")

p2 <- ggplot() +
  geom_point(data = data,
             aes(x = X1, y = Y),
             color = "black", size = 2.5) +
  geom_line(data = comparison,
            aes(x = X1, y = Y, color = Method),
            linewidth = 1.2) +
  scale_color_manual(values = c("OLS" = "red", "Median_QR" = "blue")) +
  labs(title    = "OLS vs Median Quantile Regression",
       subtitle = "OLS is pulled by outliers; Median QR is more robust",
       x = "X1",
       y = "Response (Y)") +
  theme_minimal() +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold"))

# Plot: Simple bivariate quantile lines (X1 only)

p3 <- ggplot(data, aes(x = X1, y = Y)) +
  geom_point(color = "black", size = 2.5) +
  geom_quantile(quantiles = taus, formula = y ~ x,
                aes(color = after_stat(quantile)),
                linewidth = 0.9) +
  scale_color_viridis_c(name = "Quantile") +
  geom_smooth(method = "lm", se = FALSE, color = "red",
              linetype = "dashed", linewidth = 1) +
  labs(title    = "Simple Quantile Regression Lines",
       subtitle = "Red dashed = OLS; coloured lines = quantile regression at various tau",
       x = "X1",
       y = "Response (Y)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# Display plots
print(p1)
print(p2)
print(p3)


# ====================
## HW 2-- Problem 2 (regression quantile curves for various choices of quantile indexes)
# ====================
# install.packages(c("quantreg", "ggplot2", "dplyr", "tidyr"))

library(quantreg)
library(ggplot2)
library(dplyr)
library(tidyr)
library(knitr)

# 1. Input the mock exam dataset
data <- data.frame(
  X = c(0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
  Y = c(15.2, 8.5, 3.1, 1.2, 0.8, 2.5, 6.8, 14.1)
)

# Long format for boxplots (numeric columns only)
data_long <- data %>%
  pivot_longer(cols = everything(), names_to = "feature", values_to = "value")

# Boxplot
boxplot_q2 <- ggplot(data_long, aes(x = feature, y = value)) +
  geom_boxplot(fill = "steelblue", outlier.colour = "black", outlier.shape = 16) +
  facet_wrap(~feature, scales = "free") +
  theme_minimal() +
  theme(axis.text.x = element_blank()) +
  labs(title = "Boxplots for Outlier Identification",
       y = "Value", x = "")
boxplot_q2
# Non-parametric Quantile Regression using rqss (smoothing splines)
# Taus updated to match exam prompt: 0.25, 0.50, 0.75
taus_np <- c(0.25, 0.50, 0.75)

# Fit non-parametric quantile regression models
# Reduced lambda to 0.5 to prevent over-smoothing on a small 8-point dataset
qrss_models <- list()
for (tau in taus_np) {
  qrss_models[[paste0("tau_", tau)]] <- rqss(
    Y ~ qss(X, lambda = 0.5),
    tau  = tau,
    data = data
  )
}

# Predictions on a fine grid
new_X <- data.frame(X = seq(min(data$X), max(data$X), length.out = 200))

np_predictions <- data.frame()
for (tau in taus_np) {
  pred <- predict(qrss_models[[paste0("tau_", tau)]], newdata = new_X)
  np_predictions <- rbind(np_predictions,
                          data.frame(X   = new_X$X,
                                     Y   = as.vector(pred),
                                     tau = factor(tau)))
}

# Plot: Non-parametric quantile curves
p_np <- ggplot() +
  geom_point(data = data, aes(x = X, y = Y),
             color = "black", size = 3) +
  geom_line(data = np_predictions,
            aes(x = X, y = Y, color = tau),
            linewidth = 1) +
  scale_color_viridis_d(name = "Quantile (tau)") +
  labs(title    = "Non-parametric Regression Quantile Curves",
       subtitle = "Y vs X fitted with quantile smoothing splines (rqss)",
       x = "Predictor (X)",
       y = "Response (Y)") +
  theme_minimal() +
  theme(legend.position = "right",
        plot.title = element_text(face = "bold"))
p_np
# Comparison: LOESS (mean) vs Median Non-parametric QR
loess_fit  <- loess(Y ~ X, data = data, span = 0.6)
loess_pred <- predict(loess_fit, newdata = new_X)

median_np <- np_predictions %>% filter(tau == 0.5)

p_compare <- ggplot() +
  geom_point(data = data, aes(x = X, y = Y),
             color = "black", size = 3) +
  geom_line(data = data.frame(X = new_X$X, Y = loess_pred),
            aes(x = X, y = Y, color = "LOESS (Mean)"),
            linewidth = 1.2, linetype = "dashed") +
  geom_line(data = median_np,
            aes(x = X, y = Y, color = "Median QRSS"),
            linewidth = 1.2) +
  scale_color_manual(name = "Method",
                     values = c("LOESS (Mean)" = "red", "Median QRSS" = "blue")) +
  labs(title = "LOESS vs Non-parametric Median Quantile Regression",
       x = "Predictor (X)",
       y = "Response (Y)") +
  theme_minimal() +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold"))
p_compare
# Quantile bands (prediction intervals)
np_wide <- np_predictions %>%
  pivot_wider(names_from = tau, values_from = Y, names_prefix = "tau_")

p_bands <- ggplot() +
  geom_ribbon(data = np_wide,
              aes(x = X, ymin = tau_0.25, ymax = tau_0.75),
              fill = "steelblue", alpha = 0.5) +
  geom_point(data = data, aes(x = X, y = Y),
             color = "black", size = 3) +
  geom_line(data = np_predictions %>% filter(tau == 0.5),
            aes(x = X, y = Y),
            color = "darkblue", linewidth = 1.2) +
  labs(title    = "Non-parametric Quantile Bands",
       subtitle = "Blue band: 50% interval (25th–75th quantiles)",
       x = "Predictor (X)",
       y = "Response (Y)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# Display plots
print(boxplot_q2)
print(p_np)
print(p_compare)
print(p_bands)



# ======================
## HW 2-- Problem 3 (data depth and multivariate quantile contours for sym and asym data)
# ======================

# install.packages(c("ddalpha", "ggplot2", "MASS", "gridExtra"))

library(ddalpha)
library(ggplot2)
library(MASS)
library(gridExtra)

# 1. Input Symmetric Data (Group A)
df_sym <- data.frame(
  x = c(-0.5, 0.8, -1.2, 1.1, 0.1),
  y = c(0.2, -0.6, -1.0, 1.5, -0.1)
)
df_sym$type <- "Symmetric"

# 2. Input Asymmetric Data (Group B)
df_asym <- data.frame(
  x = c(0.1, 0.4, 1.2, 3.5, 7.2),
  y = c(0.2, 0.8, 2.5, 8.1, 18.5)
)
df_asym$type <- "Asymmetric"

# Function to calculate depth grid for visualization
calculate_depth_grid <- function(data_df, depth_method = "Mahalanobis") {
  
  # Create a grid covering the data range
  x_seq <- seq(min(data_df$x)-1, max(data_df$x)+1, length.out = 50)
  y_seq <- seq(min(data_df$y)-1, max(data_df$y)+1, length.out = 50)
  grid_df <- expand.grid(x = x_seq, y = y_seq)
  
  # Calculate depth based on the chosen method
  if (depth_method == "Mahalanobis") {
    grid_df$depth <- depth.Mahalanobis(x = grid_df, data = data_df[,1:2])
  } else if (depth_method == "Projection") {
    # Projection depth is an approximation of Half-space depth
    grid_df$depth <- depth.halfspace(x = grid_df, data = data_df[,1:2])
  }
  
  return(grid_df)
}

# Compute grids for Symmetric Data
grid_sym_mah <- calculate_depth_grid(df_sym, "Mahalanobis")
grid_sym_hs <- calculate_depth_grid(df_sym, "Projection")

# Compute grids for Asymmetric Data
grid_asym_mah <- calculate_depth_grid(df_asym, "Mahalanobis")
grid_asym_hs <- calculate_depth_grid(df_asym, "Projection")

# Function to plot data with contours
plot_contours <- function(data_df, grid_df, title) {
  ggplot() +
    # Plot original data points (Adjusted size/color for small dataset)
    geom_point(data = data_df, aes(x = x, y = y), color = "black", size = 3) +
    # Plot depth contours (Multivariate Quantiles)
    geom_contour(data = grid_df, aes(x = x, y = y, z = depth, color = after_stat(level)), bins = 10, linewidth = 1) +
    scale_color_viridis_c(name = "Depth") +
    ggtitle(title) +
    theme_minimal() +
    theme(legend.position = "bottom")
}

p1 <- plot_contours(df_sym, grid_sym_mah, "Symmetric: Mahalanobis Depth")
p2 <- plot_contours(df_sym, grid_sym_hs,  "Symmetric: Half-Space Depth")
p3 <- plot_contours(df_asym, grid_asym_mah, "Asymmetric: Mahalanobis Depth")
p4 <- plot_contours(df_asym, grid_asym_hs,  "Asymmetric: Half-Space Depth")

grid.arrange(p1, p2, p3, p4, ncol = 2)

# ==========================================
# 3. Calculate Depth for Exam Target Point
# ==========================================
target_pt <- data.frame(x = 2.0, y = 2.0)

depth_mah_sym <- depth.Mahalanobis(x = target_pt, data = df_sym[,1:2])
depth_mah_asym <- depth.Mahalanobis(x = target_pt, data = df_asym[,1:2])

cat("\n--- Exam Question Answer ---\n")
cat("Mahalanobis Depth of (2.0, 2.0) wrt Group A (Symmetric):", depth_mah_sym, "\n")
cat("Mahalanobis Depth of (2.0, 2.0) wrt Group B (Asymmetric):", depth_mah_asym, "\n")


# ======================================
## HW2 -- Problem 3 (original problem)
# ======================================

library(ddalpha)
library(ggplot2)
library(MASS)
library(gridExtra)

set.seed(67)

# 1. Generate Symmetric Data (Bivariate Normal)
# Mean at (0,0), Covariance matrix with correlation 0.5
sigma_sym <- matrix(c(1, 0.5, 0.5, 1), nrow = 2)
data_sym <- mvrnorm(n = 500, mu = c(0, 0), Sigma = sigma_sym)
df_sym <- as.data.frame(data_sym)
colnames(df_sym) <- c("x", "y")
df_sym$type <- "Symmetric"

# 2. Generate Asymmetric Data (Mixture of Normals - Bimodal)
# Group A: Center at (-2, -2)
# Group B: Center at (2, 2) with higher variance
data_asym_1 <- mvrnorm(n = 300, mu = c(-2, -2), Sigma = matrix(c(1, 0.2, 0.2, 1), 2))
data_asym_2 <- mvrnorm(n = 200, mu = c(2, 2), Sigma = matrix(c(2, -0.5, -0.5, 2), 2))
data_asym <- rbind(data_asym_1, data_asym_2)
df_asym <- as.data.frame(data_asym)
colnames(df_asym) <- c("x", "y")
df_asym$type <- "Asymmetric"

# Function to calculate depth grid for visualization
calculate_depth_grid <- function(data_df, depth_method = "Mahalanobis") {
  
  # Create a grid covering the data range
  x_seq <- seq(min(data_df$x)-1, max(data_df$x)+1, length.out = 50)
  y_seq <- seq(min(data_df$y)-1, max(data_df$y)+1, length.out = 50)
  grid_df <- expand.grid(x = x_seq, y = y_seq)
  
  # Calculate depth based on the chosen method
  if (depth_method == "Mahalanobis") {
    grid_df$depth <- depth.Mahalanobis(x = grid_df, data = data_df[,1:2])
  } else if (depth_method == "Projection") {
    # Projection depth is an approximation of Half-space depth
    grid_df$depth <- depth.halfspace(x = grid_df, data = data_df[,1:2])
  }
  
  return(grid_df)
}

# Compute grids for Symmetric Data
grid_sym_mah <- calculate_depth_grid(df_sym, "Mahalanobis")
grid_sym_hs <- calculate_depth_grid(df_sym, "Projection")

# Compute grids for Asymmetric Data
grid_asym_mah <- calculate_depth_grid(df_asym, "Mahalanobis")
grid_asym_hs <- calculate_depth_grid(df_asym, "Projection")


# Function to plot data with contours
plot_contours <- function(data_df, grid_df, title) {
  ggplot() +
    # Plot original data points
    geom_point(data = data_df, aes(x = x, y = y), color = "grey40", alpha = 0.5) +
    # Plot depth contours (Multivariate Quantiles)
    geom_contour(data = grid_df, aes(x = x, y = y, z = depth, color = after_stat(level)), bins = 10, linewidth = 1) +
    scale_color_viridis_c(name = "Depth") +
    ggtitle(title) +
    theme_minimal() +
    theme(legend.position = "bottom")
}

p1 <- plot_contours(df_sym, grid_sym_mah, "Symmetric: Mahalanobis Depth")
p2 <- plot_contours(df_sym, grid_sym_hs,  "Symmetric: Half-Space Depth")
p3 <- plot_contours(df_asym, grid_asym_mah, "Asymmetric: Mahalanobis Depth")
p4 <- plot_contours(df_asym, grid_asym_hs,  "Asymmetric: Half-Space Depth")

grid.arrange(p1, p2, p3, p4, ncol = 2)


# ===================
## HW 3--Problem 1 (Ckeecing MVN using DD)
# ===================

if (!require("mvnTest")) install.packages("mvnTest")
if (!require("fasano.franceschini.test")) install.packages("fasano.franceschini.test")
if (!require("ddalpha")) install.packages("ddalpha")
if (!require("ks")) install.packages("ks")
if (!require("ggplot2")) install.packages("ggplot2")

library(MASS)
library(mvnTest)
library(fasano.franceschini.test)
library(ddalpha)
library(ks)
library(ggplot2)

# 1. Input the mock exam dataset (replaces wine.csv)
mock_data <- data.frame(
  X1 = c(4.0, 4.2, 3.9, 4.5, 4.1, 8.5),
  X2 = c(2.1, 2.5, 2.0, 2.8, 2.2, 6.0)
)
X <- as.matrix(mock_data)

cat(" Sample size =", nrow(X), "\n","Dimension =", ncol(X), "\n")

# Estimate parameters
mu_hat <- colMeans(X)
Sigma_hat <- cov(X)

# Simulate fitted multivariate normal sample
set.seed(123)
Y_norm <- MASS::mvrnorm(n = nrow(X), mu = mu_hat, Sigma = Sigma_hat)

# 1. Multivariate KS-type test (Fasano & Franceschini)
ks_res <- fasano.franceschini.test(X, Y_norm)
print(ks_res)

# 2. Multivariate Cramer-von Mises test
# Note: CM.test requires n >= 20. Wrapped in try() to avoid crashing on toy data.
cat("\nRunning CM.test (will likely throw an error due to n=6):\n")
try(cvm_res <- CM.test(X, qqplot = FALSE))
try(print(cvm_res))

Z <- rbind(X, Y_norm)

# Calculate depths
dX_mah <- depth.Mahalanobis(x = Z, data = X)
dY_mah <- depth.Mahalanobis(x = Z, data = Y_norm)

# Set exact = TRUE for small exam datasets
dX_half <- depth.halfspace(x = Z, data = X, exact = TRUE)
dY_half <- depth.halfspace(x = Z, data = Y_norm, exact = TRUE)

dX_proj <- depth.projection(x = Z, data = X)
dY_proj <- depth.projection(x = Z, data = Y_norm)

par(mfrow = c(1, 3))
# Mahalanobis
plot(dX_mah, dY_mah, xlab = "Depth wrt Observed Data", ylab = "Depth wrt Fitted MVN",
     main = "DD Plot: Mahalanobis", pch = 19, col = "blue", cex = 1.5)
abline(0, 1, lty = 2, col = "red", lwd = 2)

# Halfspace
plot(dX_half, dY_half, xlab = "Depth wrt Observed Data", ylab = "Depth wrt Fitted MVN",
     main = "DD Plot: Halfspace", pch = 19, col = "darkgreen", cex = 1.5)
abline(0, 1, lty = 2, col = "red", lwd = 2)

# Projection
plot(dX_proj, dY_proj, xlab = "Depth wrt Observed Data", ylab = "Depth wrt Fitted MVN",
     main = "DD Plot: Projection", pch = 19, col = "purple", cex = 1.5)
abline(0, 1, lty = 2, col = "red", lwd = 2)

par(mfrow = c(1, 1))
cat("\nCorrelation of DD coordinates:\n",
    "Mahalanobis depth :", round(cor(dX_mah, dY_mah), 4), "\n",
    "Halfspace depth   :", round(cor(dX_half, dY_half), 4), "\n",
    "Projection depth  :", round(cor(dX_proj, dY_proj), 4), "\n")



# ===================
## HW 3--Problem 2
# =================== 

# ===============================================
# HW3 - Depth Based + KDE Based Classification
# on Mini-Iris Exam Data
# ===============================================

if (!require("ddalpha")) install.packages("ddalpha")
if (!require("ks"))      install.packages("ks")
if (!require("ggplot2")) install.packages("ggplot2")

library(ddalpha)
library(ks)
library(ggplot2)

# ===============================================
# Load and Split Data (Mock Exam Dataset)
# ===============================================

# Training Data
X_train <- matrix(c(5.1, 3.5, 
                    4.9, 3.0, 
                    4.7, 3.2, 
                    7.0, 3.2, 
                    6.4, 3.2, 
                    6.9, 3.1), ncol = 2, byrow = TRUE)
y_train <- factor(c(rep("Class1", 3), rep("Class2", 3)))

# Test Data (Adding true labels based on obvious visual proximity for evaluation)
X_test <- matrix(c(5.0, 3.4, 
                   6.8, 3.0), ncol = 2, byrow = TRUE)
y_test <- factor(c("Class1", "Class2"))

# Create a combined mock_iris dataframe for later ggplot visualizations
mock_iris <- data.frame(rbind(X_train, X_test))
colnames(mock_iris) <- c("Sepal.Length", "Sepal.Width")
mock_iris$Species <- factor(c(as.character(y_train), as.character(y_test)))

# Split into class-specific matrices
class_1 <- X_train[y_train == "Class1", ]
class_2 <- X_train[y_train == "Class2", ]

class_list  <- list(class_1, class_2)
class_names <- c("Class1", "Class2")

# ===============================================
# PART A: Depth-Based Classification
# ===============================================

# Note: simplicial and halfspace removed because n=3 is too small for geometric depth in 2D
depth_methods <- c("Mahalanobis", "spatial") 

depth_funcs <- list(
  Mahalanobis = depth.Mahalanobis,
  spatial     = depth.spatial
)

results <- list()

for (m in depth_methods) {
  
  cat("Classifying with", m, "depth ...\n")
  predicted <- character(nrow(X_test))
  
  for (i in 1:nrow(X_test)) {
    x <- matrix(X_test[i, ], nrow = 1)
    d <- numeric(2) # Reduced to 2 classes
    for (j in 1:2) {
      d[j] <- depth_funcs[[m]](x, class_list[[j]])
    }
    predicted[i] <- class_names[which.max(d)]
  }
  
  predicted <- factor(predicted, levels = levels(mock_iris$Species))
  
  results[[m]] <- list(
    predicted     = predicted,
    accuracy      = mean(predicted == y_test) * 100,
    error_rate    = mean(predicted != y_test),
    misclassified = sum(predicted != y_test),
    conf_matrix   = table(Predicted = predicted, Actual = y_test)
  )
}

# ===============================================
# PART B: KDE-Based Classification
# ===============================================

# --- Method 1: KDE with Normal Scale Bandwidth ---
cat("Classifying with KDE (Normal Scale bandwidth) ...\n")

H_ns <- list(Hns(class_1), Hns(class_2))

kde_ns <- list(
  kde(class_1, H = H_ns[[1]]),
  kde(class_2, H = H_ns[[2]])
)

predicted_kde_ns <- character(nrow(X_test))
for (i in 1:nrow(X_test)) {
  x <- X_test[i, , drop = FALSE]
  dens <- numeric(2)
  for (j in 1:2) {
    dens[j] <- predict(kde_ns[[j]], x = x)
  }
  predicted_kde_ns[i] <- class_names[which.max(dens)]
}
predicted_kde_ns <- factor(predicted_kde_ns, levels = levels(mock_iris$Species))

results[["KDE_NormalScale"]] <- list(
  predicted     = predicted_kde_ns,
  accuracy      = mean(predicted_kde_ns == y_test) * 100,
  error_rate    = mean(predicted_kde_ns != y_test),
  misclassified = sum(predicted_kde_ns != y_test),
  conf_matrix   = table(Predicted = predicted_kde_ns, Actual = y_test)
)

# --- Method 2: KDE with Plug-In Bandwidth ---
cat("Classifying with KDE (Plug-In bandwidth) ...\n")

H_pi <- list(Hpi(class_1), Hpi(class_2))

kde_pi <- list(
  kde(class_1, H = H_pi[[1]]),
  kde(class_2, H = H_pi[[2]])
)

predicted_kde_pi <- character(nrow(X_test))
for (i in 1:nrow(X_test)) {
  x <- X_test[i, , drop = FALSE]
  dens <- numeric(2)
  for (j in 1:2) {
    dens[j] <- predict(kde_pi[[j]], x = x)
  }
  predicted_kde_pi[i] <- class_names[which.max(dens)]
}
predicted_kde_pi <- factor(predicted_kde_pi, levels = levels(mock_iris$Species))

results[["KDE_PlugIn"]] <- list(
  predicted     = predicted_kde_pi,
  accuracy      = mean(predicted_kde_pi == y_test) * 100,
  error_rate    = mean(predicted_kde_pi != y_test),
  misclassified = sum(predicted_kde_pi != y_test),
  conf_matrix   = table(Predicted = predicted_kde_pi, Actual = y_test)
)

# ===============================================
# Print All Confusion Matrices
# ===============================================

all_methods <- c(depth_methods, "KDE_NormalScale", "KDE_PlugIn")

for (m in all_methods) {
  cat("\n================================================\n")
  cat("Method:", m, "\n")
  cat("================================================\n")
  print(results[[m]]$conf_matrix)
  cat("Accuracy:", round(results[[m]]$accuracy, 2), "%\n")
  cat("Misclassified:", results[[m]]$misclassified, "out of", nrow(X_test), "\n")
}

# ===============================================
# Comparison Table
# ===============================================

cat("\n     COMPARISON: DEPTH-BASED vs KDE-BASED CLASSIFIERS    \n")

comparison <- data.frame(
  Method        = c(paste("Depth:", depth_methods),
                    "KDE: Normal Scale", "KDE: Plug-In"),
  Type          = c(rep("Depth-Based", 2), rep("KDE-Based", 2)),
  Accuracy      = sapply(all_methods, function(m) 
    paste0(round(results[[m]]$accuracy, 2), "%")),
  Misclassified = sapply(all_methods, function(m) 
    results[[m]]$misclassified),
  Error_Rate    = sapply(all_methods, function(m) 
    round(results[[m]]$error_rate, 4)),
  row.names = NULL
)
print(comparison)

# ===============================================
# Visualization 1: Scatter Plot
# ===============================================

p1 <- ggplot(mock_iris, aes(Sepal.Length, Sepal.Width, color = Species)) +
  geom_point(size = 4) +
  scale_color_manual(values = c("red", "blue")) +
  ggtitle("Mini-Iris Dataset: Sepal Features") +
  theme_minimal()
print(p1)

# ===============================================
# Visualization 2: PCA Plot
# ===============================================

pca <- prcomp(mock_iris[, 1:2], scale. = TRUE)
pca_df <- data.frame(pca$x[, 1:2], Species = mock_iris$Species)

p2 <- ggplot(pca_df, aes(PC1, PC2, color = Species)) +
  geom_point(size = 4) +
  scale_color_manual(values = c("red", "blue")) +
  ggtitle("PCA Projection of Mini-Iris Dataset") +
  theme_minimal()
print(p2)

# ===============================================
# Visualization 3: Depth Distribution
# ===============================================

depth_1 <- depth.Mahalanobis(class_1, class_1)
depth_2 <- depth.Mahalanobis(class_2, class_2)

depth_df <- data.frame(
  Depth = c(depth_1, depth_2),
  Class = factor(c(rep("Class1", nrow(class_1)),
                   rep("Class2", nrow(class_2))),
                 levels = levels(mock_iris$Species))
)

p3 <- ggplot(depth_df, aes(x = Depth, fill = Class)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("red", "blue")) +
  ggtitle("Depth Distribution by Class (Mahalanobis)") +
  theme_minimal()
print(p3)

# ===============================================
# Visualization 4: DD-Plots
# ===============================================

DDplot <- function(classA, classB, labelA, labelB) {
  combined <- rbind(classA, classB)
  depths_A <- depth.Mahalanobis(combined, classA)
  depths_B <- depth.Mahalanobis(combined, classB)
  
  df <- data.frame(
    depthA = depths_A,
    depthB = depths_B,
    class  = factor(c(rep(labelA, nrow(classA)),
                      rep(labelB, nrow(classB))))
  )
  
  p <- ggplot(df, aes(depthA, depthB, color = class)) +
    geom_point(size = 4) +
    scale_color_manual(values = c("red", "blue")) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    labs(x     = paste("Depth w.r.t", labelA),
         y     = paste("Depth w.r.t", labelB),
         title = paste("DD Plot:", labelA, "vs", labelB)) +
    theme_minimal()
  print(p)
}

DDplot(class_1, class_2, "Class1", "Class2")

# ===============================================
# Visualization 5: KDE 1D Density per Feature
# ===============================================

par(mfrow = c(1, 2))
feature_names <- colnames(mock_iris)[1:2]

for (f in 1:2) {
  
  d_1 <- density(class_1[, f])
  d_2 <- density(class_2[, f])
  
  yr <- range(c(d_1$y, d_2$y))
  xr <- range(c(d_1$x, d_2$x))
  
  plot(d_1, col = "red", lwd = 2,
       xlim = xr, ylim = yr,
       main = paste("KDE:", feature_names[f]),
       xlab = feature_names[f], ylab = "Density")
  lines(d_2, col = "blue", lwd = 2)
  
  legend("topright",
         legend = class_names,
         col = c("red", "blue"),
         lwd = 2, cex = 0.7)
}
par(mfrow = c(1, 1))

# ===============================================
# Visualization 6: Classification Comparison
# ===============================================

acc_df <- data.frame(
  Method   = factor(c(paste("Depth:", depth_methods), 
                      "KDE: Normal Scale", "KDE: Plug-In"),
                    levels = c(paste("Depth:", depth_methods), 
                               "KDE: Normal Scale", "KDE: Plug-In")),
  Type     = c(rep("Depth-Based", 2), rep("KDE-Based", 2)),
  Accuracy = sapply(all_methods, function(m) results[[m]]$accuracy)
)

p6 <- ggplot(acc_df, aes(x = Method, y = Accuracy, fill = Type)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(round(Accuracy, 1), "%")),
            vjust = -0.5, size = 3.5) +
  ylim(0, 105) +
  labs(title = "Classification Accuracy: Depth vs KDE",
       y = "Accuracy (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))
print(p6)

# ====================
## HW3--Problem 2(original dataset)

# ===============================================
# HW3 - Depth Based + KDE Based Classification
# on Iris Data
# ===============================================

if (!require("ddalpha")) install.packages("ddalpha")
if (!require("ks"))      install.packages("ks")
if (!require("ggplot2")) install.packages("ggplot2")

library(ddalpha)
library(ks)
library(ggplot2)

# ===============================================
# Load and Split Data
# ===============================================

data(iris)

train_index <- c(1:25, 51:75, 101:125)
test_index  <- c(26:50, 76:100, 126:150)

train_data <- iris[train_index, ]
test_data  <- iris[test_index, ]

X_train <- as.matrix(train_data[, 1:4])
y_train <- train_data$Species

X_test <- as.matrix(test_data[, 1:4])
y_test <- test_data$Species

class_setosa     <- X_train[y_train == "setosa", ]
class_versicolor <- X_train[y_train == "versicolor", ]
class_virginica  <- X_train[y_train == "virginica", ]

class_list  <- list(class_setosa, class_versicolor, class_virginica)
class_names <- c("setosa", "versicolor", "virginica")

# ===============================================
# PART A: Depth-Based Classification (4 depths)
# ===============================================

depth_methods <- c("Mahalanobis", "halfspace", "simplicial", "spatial")

depth_funcs <- list(
  Mahalanobis = depth.Mahalanobis,
  halfspace   = depth.halfspace,
  simplicial  = depth.simplicial,
  spatial     = depth.spatial
)

results <- list()

for (m in depth_methods) {
  
  cat("Classifying with", m, "depth ...\n")
  
  predicted <- character(nrow(X_test))
  
  for (i in 1:nrow(X_test)) {
    
    x <- matrix(X_test[i, ], nrow = 1)
    
    d <- numeric(3)
    for (j in 1:3) {
      d[j] <- depth_funcs[[m]](x, class_list[[j]])
    }
    
    predicted[i] <- class_names[which.max(d)]
  }
  
  predicted <- factor(predicted, levels = levels(iris$Species))
  
  results[[m]] <- list(
    predicted     = predicted,
    accuracy      = mean(predicted == y_test) * 100,
    error_rate    = mean(predicted != y_test),
    misclassified = sum(predicted != y_test),
    conf_matrix   = table(Predicted = predicted, Actual = y_test)
  )
}

# ===============================================
# PART B: KDE-Based Classification
# ===============================================

# --- Method 1: KDE with Normal Scale Bandwidth ---
cat("Classifying with KDE (Normal Scale bandwidth) ...\n")

H_ns <- list(Hns(class_setosa), Hns(class_versicolor), Hns(class_virginica))

kde_ns <- list(
  kde(class_setosa,     H = H_ns[[1]]),
  kde(class_versicolor, H = H_ns[[2]]),
  kde(class_virginica,  H = H_ns[[3]])
)

predicted_kde_ns <- character(nrow(X_test))
for (i in 1:nrow(X_test)) {
  x <- X_test[i, , drop = FALSE]
  dens <- numeric(3)
  for (j in 1:3) {
    dens[j] <- predict(kde_ns[[j]], x = x)
  }
  predicted_kde_ns[i] <- class_names[which.max(dens)]
}
predicted_kde_ns <- factor(predicted_kde_ns, levels = levels(iris$Species))

results[["KDE_NormalScale"]] <- list(
  predicted     = predicted_kde_ns,
  accuracy      = mean(predicted_kde_ns == y_test) * 100,
  error_rate    = mean(predicted_kde_ns != y_test),
  misclassified = sum(predicted_kde_ns != y_test),
  conf_matrix   = table(Predicted = predicted_kde_ns, Actual = y_test)
)

# --- Method 2: KDE with Plug-In Bandwidth ---
cat("Classifying with KDE (Plug-In bandwidth) ...\n")

H_pi <- list(Hpi(class_setosa), Hpi(class_versicolor), Hpi(class_virginica))

kde_pi <- list(
  kde(class_setosa,     H = H_pi[[1]]),
  kde(class_versicolor, H = H_pi[[2]]),
  kde(class_virginica,  H = H_pi[[3]])
)

predicted_kde_pi <- character(nrow(X_test))
for (i in 1:nrow(X_test)) {
  x <- X_test[i, , drop = FALSE]
  dens <- numeric(3)
  for (j in 1:3) {
    dens[j] <- predict(kde_pi[[j]], x = x)
  }
  predicted_kde_pi[i] <- class_names[which.max(dens)]
}
predicted_kde_pi <- factor(predicted_kde_pi, levels = levels(iris$Species))

results[["KDE_PlugIn"]] <- list(
  predicted     = predicted_kde_pi,
  accuracy      = mean(predicted_kde_pi == y_test) * 100,
  error_rate    = mean(predicted_kde_pi != y_test),
  misclassified = sum(predicted_kde_pi != y_test),
  conf_matrix   = table(Predicted = predicted_kde_pi, Actual = y_test)
)

# ===============================================
# Print All Confusion Matrices
# ===============================================

all_methods <- c(depth_methods, "KDE_NormalScale", "KDE_PlugIn")

for (m in all_methods) {
  cat("\n================================================\n")
  cat("Method:", m, "\n")
  cat("================================================\n")
  print(results[[m]]$conf_matrix)
  cat("Accuracy:", round(results[[m]]$accuracy, 2), "%\n")
  cat("Misclassified:", results[[m]]$misclassified, "out of", nrow(X_test), "\n")
}

# ===============================================
# Comparison Table
# ===============================================


cat("     COMPARISON: DEPTH-BASED vs KDE-BASED CLASSIFIERS    \n")


comparison <- data.frame(
  Method        = c(paste("Depth:", depth_methods),
                    "KDE: Normal Scale", "KDE: Plug-In"),
  Type          = c(rep("Depth-Based", 4), rep("KDE-Based", 2)),
  Accuracy      = sapply(all_methods, function(m) 
    paste0(round(results[[m]]$accuracy, 2), "%")),
  Misclassified = sapply(all_methods, function(m) 
    results[[m]]$misclassified),
  Error_Rate    = sapply(all_methods, function(m) 
    round(results[[m]]$error_rate, 4)),
  row.names = NULL
)
print(comparison)

best <- all_methods[which.max(sapply(all_methods, function(m) results[[m]]$accuracy))]
cat("\nBest performing method:", best, "\n")

# ===============================================
# Per-Class Accuracy
# ===============================================


cat("              PER-CLASS ACCURACY                          \n")

for (m in all_methods) {
  cat("\n---", m, "---\n")
  for (cl in class_names) {
    idx     <- y_test == cl
    correct <- sum(results[[m]]$predicted[idx] == y_test[idx])
    total   <- sum(idx)
    cat(sprintf("  %-12s: %d / %d correct (%.1f%%)\n",
                cl, correct, total, correct / total * 100))
  }
}

# ===============================================
# Visualization 1: Scatter Plot
# ===============================================

ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) +
  geom_point(size = 3) +
  ggtitle("Iris Dataset: Sepal Features") +
  theme_minimal()

# ===============================================
# Visualization 2: PCA Plot
# ===============================================

pca <- prcomp(iris[, 1:4], scale. = TRUE)
pca_df <- data.frame(pca$x[, 1:2], Species = iris$Species)

ggplot(pca_df, aes(PC1, PC2, color = Species)) +
  geom_point(size = 3) +
  ggtitle("PCA Projection of Iris Dataset") +
  theme_minimal()

# ===============================================
# Visualization 3: Depth Distribution
# ===============================================

depth_setosa     <- depth.Mahalanobis(class_setosa,     class_setosa)
depth_versicolor <- depth.Mahalanobis(class_versicolor, class_versicolor)
depth_virginica  <- depth.Mahalanobis(class_virginica,  class_virginica)

depth_df <- data.frame(
  Depth = c(depth_setosa, depth_versicolor, depth_virginica),
  Class = factor(c(rep("setosa", 25),
                   rep("versicolor", 25),
                   rep("virginica", 25)),
                 levels = levels(iris$Species))
)

ggplot(depth_df, aes(x = Depth, fill = Class)) +
  geom_density(alpha = 0.5) +
  ggtitle("Depth Distribution by Class (Mahalanobis)") +
  theme_minimal()

# ===============================================
# Visualization 4: DD-Plots
# ===============================================

DDplot <- function(classA, classB, labelA, labelB) {
  
  combined <- rbind(classA, classB)
  
  depths_A <- depth.Mahalanobis(combined, classA)
  depths_B <- depth.Mahalanobis(combined, classB)
  
  df <- data.frame(
    depthA = depths_A,
    depthB = depths_B,
    class  = factor(c(rep(labelA, nrow(classA)),
                      rep(labelB, nrow(classB))))
  )
  
  ggplot(df, aes(depthA, depthB, color = class)) +
    geom_point(size = 3) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    labs(x     = paste("Depth w.r.t", labelA),
         y     = paste("Depth w.r.t", labelB),
         title = paste("DD Plot:", labelA, "vs", labelB)) +
    theme_minimal()
}

setosa     <- as.matrix(iris[iris$Species == "setosa", 1:4])
versicolor <- as.matrix(iris[iris$Species == "versicolor", 1:4])
virginica  <- as.matrix(iris[iris$Species == "virginica", 1:4])

DDplot(setosa, versicolor, "setosa", "versicolor")
DDplot(setosa, virginica,  "setosa", "virginica")
DDplot(versicolor, virginica, "versicolor", "virginica")

# ===============================================
# Visualization 5: Depth Contours in PCA Space
# ===============================================

pca_data <- data.frame(pca$x[, 1:2], Species = iris$Species)

draw_ellipse <- function(data, color) {
  mu <- colMeans(data)
  S  <- cov(data)
  angles  <- seq(0, 2 * pi, length = 100)
  circle  <- cbind(cos(angles), sin(angles))
  ellipse <- t(mu + t(circle %*% chol(S)))
  lines(ellipse, col = color, lwd = 2)
}

plot(pca_data$PC1, pca_data$PC2,
     col = as.numeric(pca_data$Species),
     pch = 19, xlab = "PC1", ylab = "PC2",
     main = "Depth-like Contours in PCA Space")

legend("topright", legend = levels(pca_data$Species), col = 1:3, pch = 19)

draw_ellipse(pca_data[pca_data$Species == "setosa",     1:2], "black")
draw_ellipse(pca_data[pca_data$Species == "versicolor", 1:2], "red")
draw_ellipse(pca_data[pca_data$Species == "virginica",  1:2], "green3")

# ===============================================
# Visualization 6: KDE Contour Plot in PCA Space
# ===============================================

# Project training data to PCA space for 2D KDE visualization
pca_train <- predict(pca, X_train)[, 1:2]
pca_test  <- predict(pca, X_test)[, 1:2]

pca_setosa     <- pca_train[y_train == "setosa", ]
pca_versicolor <- pca_train[y_train == "versicolor", ]
pca_virginica  <- pca_train[y_train == "virginica", ]

# Fit 2D KDE for each class in PCA space
kde2d_setosa     <- kde(pca_setosa,     H = Hpi(pca_setosa))
kde2d_versicolor <- kde(pca_versicolor, H = Hpi(pca_versicolor))
kde2d_virginica  <- kde(pca_virginica,  H = Hpi(pca_virginica))

# Plot KDE contours
plot(kde2d_setosa, display = "filled.contour",
     cont = c(25, 50, 75),
     col = c(rgb(1, 0, 0, 0), rgb(1, 0, 0, 0.15), 
             rgb(1, 0, 0, 0.3), rgb(1, 0, 0, 0.45)),
     xlab = "PC1", ylab = "PC2",
     xlim = range(pca_train[, 1]) * 1.2,
     ylim = range(pca_train[, 2]) * 1.2,
     main = "KDE Density Contours in PCA Space (Plug-In Bandwidth)")

plot(kde2d_versicolor, display = "filled.contour",
     cont = c(25, 50, 75),
     col = c(rgb(0, 0, 1, 0), rgb(0, 0, 1, 0.15), 
             rgb(0, 0, 1, 0.3), rgb(0, 0, 1, 0.45)),
     add = TRUE)

plot(kde2d_virginica, display = "filled.contour",
     cont = c(25, 50, 75),
     col = c(rgb(0, 0.6, 0, 0), rgb(0, 0.6, 0, 0.15), 
             rgb(0, 0.6, 0, 0.3), rgb(0, 0.6, 0, 0.45)),
     add = TRUE)

points(pca_train[, 1], pca_train[, 2],
       col = as.numeric(y_train), pch = 19, cex = 0.8)

legend("topright",
       legend = c("setosa", "versicolor", "virginica"),
       col = c("red", "blue", "green3"),
       pch = 19, fill = c(rgb(1,0,0,0.3), rgb(0,0,1,0.3), rgb(0,0.6,0,0.3)),
       border = NA, cex = 0.8)

# ===============================================
# Visualization 7: KDE 1D Density per Feature
# ===============================================

par(mfrow = c(2, 2))
feature_names <- colnames(iris)[1:4]

for (f in 1:4) {
  
  d_s <- density(class_setosa[, f])
  d_v <- density(class_versicolor[, f])
  d_g <- density(class_virginica[, f])
  
  yr <- range(c(d_s$y, d_v$y, d_g$y))
  xr <- range(c(d_s$x, d_v$x, d_g$x))
  
  plot(d_s, col = "red", lwd = 2,
       xlim = xr, ylim = yr,
       main = paste("KDE:", feature_names[f]),
       xlab = feature_names[f], ylab = "Density")
  lines(d_v, col = "blue", lwd = 2)
  lines(d_g, col = "green3", lwd = 2)
  
  legend("topright",
         legend = class_names,
         col = c("red", "blue", "green3"),
         lwd = 2, cex = 0.7)
}
par(mfrow = c(1, 1))

# ===============================================
# Visualization 8: Classification Comparison
# ===============================================

# Bar chart of accuracy across all methods
acc_df <- data.frame(
  Method   = factor(c(paste("Depth:", depth_methods), 
                      "KDE: Normal Scale", "KDE: Plug-In"),
                    levels = c(paste("Depth:", depth_methods), 
                               "KDE: Normal Scale", "KDE: Plug-In")),
  Type     = c(rep("Depth-Based", 4), rep("KDE-Based", 2)),
  Accuracy = sapply(all_methods, function(m) results[[m]]$accuracy)
)

ggplot(acc_df, aes(x = Method, y = Accuracy, fill = Type)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = paste0(round(Accuracy, 1), "%")),
            vjust = -0.5, size = 3.5) +
  ylim(0, 105) +
  labs(title = "Classification Accuracy: Depth-Based vs KDE-Based",
       y = "Accuracy (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))


# ============================================
# HW4 -- Problem 1
# K-MEANS + DATA DEPTH (MAHALANOBIS)
# ============================================

# -------------------------------
# 1. Load Libraries
# -------------------------------
required_pkgs <- c("ggplot2", "cluster", "factoextra", "fpc",
                   "aricode", "reshape2", "gridExtra")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
}

library(ggplot2)
library(cluster)
library(factoextra)
library(fpc)
library(aricode)
library(reshape2)
library(gridExtra)

# -------------------------------
# 2. Load Dataset (Mock Exam Data)
# -------------------------------
data <- data.frame(
  X1 = c(1.1, 1.4, 0.9, 2.5, 8.1, 8.5, 9.0, 8.8),
  X2 = c(1.5, 1.2, 1.4, 3.0, 8.5, 8.0, 8.2, 8.6)
)

true_labels <- NULL # Unlabelled data
cat("Dataset dimensions:", nrow(data), "x", ncol(data), "\n")
cat("No true labels available.\n\n")

# -------------------------------
# 3. Exploratory Data Analysis
# -------------------------------
cat("===== Data Summary =====\n")
print(summary(data))
cat("\n")

# Correlation heatmap
cor_mat <- cor(data)
cor_melted <- melt(cor_mat)
p_cor <- ggplot(cor_melted, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "steelblue", mid = "white", high = "firebrick",
                       midpoint = 0, limit = c(-1, 1), name = "Correlation") +
  geom_text(aes(label = round(value, 2)), size = 3) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Feature Correlation Heatmap") +
  xlab("") + ylab("")
print(p_cor)

# -------------------------------
# 4. Standardize Data
# -------------------------------
data_scaled <- scale(data)

n <- nrow(data_scaled)
p <- ncol(data_scaled)
K <- 2   # Adjusted to 2 clusters for the exam prompt

# ============================================
# 5. DETERMINE OPTIMAL K
# ============================================

# 5a) Elbow Method (Within-cluster Sum of Squares)
# Reduced max k to 5 to avoid crashing on an 8-point dataset
wss <- sapply(1:5, function(k) {
  kmeans(data_scaled, centers = k, nstart = 10)$tot.withinss
})

p_elbow <- ggplot(data.frame(k = 1:5, WSS = wss), aes(k, WSS)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(size = 3, color = "steelblue") +
  geom_vline(xintercept = K, linetype = "dashed", color = "red") +
  annotate("text", x = K + 0.2, y = max(wss) * 0.9,
           label = paste0("Chosen k = ", K), color = "red", hjust = 0) +
  scale_x_continuous(breaks = 1:5) +
  theme_minimal() +
  ggtitle("Elbow Method — Within-Cluster Sum of Squares") +
  xlab("Number of Clusters (k)") + ylab("Total WSS")
print(p_elbow)

# 5b) Gap Statistic
# Reduced K.max to 4 to avoid crashing on an 8-point dataset
set.seed(42)
gap_stat <- clusGap(data_scaled, FUN = kmeans, nstart = 10, K.max = 4, B = 50)
p_gap <- fviz_gap_stat(gap_stat) +
  ggtitle("Gap Statistic — Optimal Number of Clusters") +
  theme_minimal()
print(p_gap)

optimal_k <- maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"],
                   method = "Tibs2001SEmax")
cat("Optimal k from Gap statistic:", optimal_k, "\n\n")

# ============================================
# 6. K-MEANS CLUSTERING
# ============================================

set.seed(123)

kmeans_res <- kmeans(data_scaled, centers = K, nstart = 25)
k_labels <- kmeans_res$cluster

cat("===== K-Means Results =====\n")
cat("Cluster sizes:\n")
print(table(k_labels))
cat("Total WSS:", kmeans_res$tot.withinss, "\n")
cat("Between-SS / Total-SS:", round(kmeans_res$betweenss / kmeans_res$totss * 100, 1), "%\n\n")

# ============================================
# 7. MAHALANOBIS DEPTH FUNCTION
# ============================================

depth_function <- function(x, cluster_data) {
  mu <- colMeans(cluster_data)
  S  <- cov(cluster_data)
  
  # Regularization to avoid singular matrix
  if (det(S) == 0 || is.na(det(S))) {
    S <- S + diag(1e-6, ncol(S))
  }
  
  d <- mahalanobis(x, mu, S)
  return(1 / (1 + d))
}

# ============================================
# 8. DATA DEPTH BASED CLUSTERING
# ============================================

set.seed(123)
# Initialize using K-Means (avoids cluster collapse)
clusters <- kmeans(data_scaled, centers = K, nstart = 25)$cluster
max_iter <- 50

for (iter in 1:max_iter) {
  new_clusters <- clusters
  for (i in 1:n) {
    depths <- numeric(K)
    for (k in 1:K) {
      cluster_data <- data_scaled[clusters == k, , drop = FALSE]
      if (nrow(cluster_data) > p + 1) {
        depths[k] <- depth_function(data_scaled[i, ], cluster_data)
      } else {
        depths[k] <- 0
      }
    }
    new_clusters[i] <- which.max(depths)
  }
  
  if (all(new_clusters == clusters)) {
    cat("Depth clustering converged at iteration:", iter, "\n")
    break
  }
  clusters <- new_clusters
}

if (iter == max_iter) {
  cat("Depth clustering reached max iterations (", max_iter, ")\n")
}

depth_labels <- clusters

cat("===== Depth-Based Clustering Results =====\n")
cat("Cluster sizes:\n")
print(table(depth_labels))
cat("\n")

# ============================================
# 9. PCA VISUALIZATION — Side by Side
# ============================================

pca <- prcomp(data_scaled)
var_explained <- round(summary(pca)$importance[2, 1:2] * 100, 1)
x_lab <- paste0("PC1 (", var_explained[1], "%)")
y_lab <- paste0("PC2 (", var_explained[2], "%)")

make_pca_plot <- function(labels, title, palette = "Set1") {
  df <- data.frame(PC1 = pca$x[, 1], PC2 = pca$x[, 2],
                   Cluster = as.factor(labels))
  ggplot(df, aes(PC1, PC2, color = Cluster)) +
    # Increased size for small dataset visibility
    geom_point(size = 4, alpha = 1) + 
    # Warning: stat_ellipse might omit ellipses if a cluster has < 3 points
    stat_ellipse(level = 0.9, linetype = "dashed", linewidth = 0.8) +
    scale_color_brewer(palette = palette) +
    theme_minimal() +
    ggtitle(title) +
    xlab(x_lab) + ylab(y_lab)
}

p_km    <- make_pca_plot(k_labels, "K-Means Clustering")
p_depth <- make_pca_plot(depth_labels, "Depth-Based Clustering")

grid.arrange(p_km, p_depth, ncol = 2,
             top = "PCA Projections — Clustering Methods")

# ============================================
# 10. DEPTH DISTRIBUTION ANALYSIS
# ============================================

cluster_depths_km <- numeric(n)
for (i in 1:n) {
  cl <- k_labels[i]
  cluster_data <- data_scaled[k_labels == cl, , drop = FALSE]
  cluster_depths_km[i] <- depth_function(data_scaled[i, ], cluster_data)
}

cluster_depths_dp <- numeric(n)
for (i in 1:n) {
  cl <- depth_labels[i]
  cluster_data <- data_scaled[depth_labels == cl, , drop = FALSE]
  cluster_depths_dp[i] <- depth_function(data_scaled[i, ], cluster_data)
}

depth_df <- data.frame(
  Depth     = c(cluster_depths_km, cluster_depths_dp),
  Method    = rep(c("K-Means", "Depth-Based"), each = n),
  Cluster   = as.factor(c(k_labels, depth_labels))
)

p_depth_box <- ggplot(depth_df, aes(x = Cluster, y = Depth, fill = Method)) +
  geom_boxplot(alpha = 0.7, position = "dodge") +
  theme_minimal() +
  ggtitle("Within-Cluster Mahalanobis Depth Distribution") +
  xlab("Cluster") + ylab("Depth") +
  scale_fill_brewer(palette = "Set2")
print(p_depth_box)

p_depth_hist <- ggplot(depth_df, aes(x = Depth, fill = Method)) +
  geom_histogram(bins = 10, alpha = 0.6, position = "identity") +
  facet_wrap(~Method, ncol = 1) +
  theme_minimal() +
  ggtitle("Depth Distribution — K-Means vs Depth-Based") +
  xlab("Mahalanobis Depth") + ylab("Count") +
  scale_fill_brewer(palette = "Set2")
print(p_depth_hist)

# ============================================
# 11. SILHOUETTE ANALYSIS
# ============================================

dist_mat <- dist(data_scaled)

sil_k <- silhouette(k_labels, dist_mat)
sil_d <- silhouette(depth_labels, dist_mat)

p_sil_k <- fviz_silhouette(sil_k) +
  ggtitle("K-Means — Silhouette Plot") + theme_minimal()
p_sil_d <- fviz_silhouette(sil_d) +
  ggtitle("Depth-Based — Silhouette Plot") + theme_minimal()

print(p_sil_k)
print(p_sil_d)

# ============================================
# 12. COMPREHENSIVE EVALUATION METRICS
# ============================================

compute_metrics <- function(labels, true_lab, dist_m, method_name) {
  cs <- cluster.stats(dist_m, labels)
  
  safe_round <- function(x, digits = 4) {
    if (is.null(x) || !is.numeric(x)) return(NA)
    round(as.numeric(x), digits)
  }
  
  bss <- cs$between.cluster.ss
  if (is.null(bss)) {
    total_ss <- sum(scale(data_scaled, scale = FALSE)^2)
    bss <- total_ss - cs$within.cluster.ss
  }
  
  data.frame(
    Method            = method_name,
    Silhouette        = safe_round(cs$avg.silwidth),
    Dunn_Index        = safe_round(cs$dunn),
    Calinski_Harabasz = safe_round(cs$ch, 2),
    Within_SS         = safe_round(cs$within.cluster.ss, 2),
    Between_SS        = safe_round(bss, 2),
    ARI               = NA, # Hardcoded NA since no true labels
    NMI               = NA,
    stringsAsFactors  = FALSE
  )
}

metrics_km    <- compute_metrics(k_labels, true_labels, dist_mat, "K-Means")
metrics_depth <- compute_metrics(depth_labels, true_labels, dist_mat, "Depth-Based")

comparison <- rbind(metrics_km, metrics_depth)

cat("\n================================================================\n")
cat("       COMPREHENSIVE CLUSTERING COMPARISON TABLE                \n")
cat("================================================================\n")
print(comparison, row.names = FALSE, right = FALSE)
cat("================================================================\n\n")

# ============================================
# 14. BAR CHART COMPARISON OF KEY METRICS
# ============================================

metrics_long <- melt(comparison, id.vars = "Method",
                     variable.name = "Metric", value.name = "Value")

key_metrics <- c("Silhouette", "Dunn_Index") # Removed ARI/NMI since unlabelled
metrics_subset <- metrics_long[metrics_long$Metric %in% key_metrics, ]

p_bar <- ggplot(metrics_subset, aes(x = Metric, y = Value, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.85, width = 0.7) +
  geom_text(aes(label = round(Value, 3)),
            position = position_dodge(width = 0.7), vjust = -0.3, size = 3.5) +
  theme_minimal() +
  ggtitle("Clustering Quality — K-Means vs Depth-Based") +
  xlab("") + ylab("Score") +
  scale_fill_brewer(palette = "Set2") +
  theme(legend.position = "top")
print(p_bar)

# ============================================
# 15. WITHIN-CLUSTER VARIANCE COMPARISON
# ============================================

cat("===== Within-Cluster Sum of Squares =====\n\n")
cat("K-Means:\n")
cat("  Total WSS:", kmeans_res$tot.withinss, "\n")
for (cl in 1:K) {
  cl_data <- data_scaled[k_labels == cl, , drop = FALSE]
  wss_cl  <- sum(scale(cl_data, scale = FALSE)^2)
  cat("    Cluster", cl, ":", round(wss_cl, 2),
      " (n =", sum(k_labels == cl), ")\n")
}

cat("\nDepth-Based:\n")
total_wss_depth <- 0
for (cl in 1:K) {
  cl_data <- data_scaled[depth_labels == cl, , drop = FALSE]
  if (nrow(cl_data) > 0) {
    wss_cl  <- sum(scale(cl_data, scale = FALSE)^2)
    total_wss_depth <- total_wss_depth + wss_cl
  } else {
    wss_cl <- 0
  }
  cat("    Cluster", cl, ":", round(wss_cl, 2),
      " (n =", sum(depth_labels == cl), ")\n")
}
cat("  Total WSS:", round(total_wss_depth, 2), "\n\n")

cat("============================================================\n")
cat("           INTERPRETATION & DISCUSSION                      \n")
cat("============================================================\n\n")

cat("Note: For this toy dataset, since the clusters are small and\n")
cat("perfectly separable, K-Means and Depth-Based clustering will\n")
cat("likely converge to identical assignments.\n")


# ============================================
# HW4 -- Problem 2 (infinte d data with same measure or not)
# ============================================
# ============================================
# FUNCTIONAL TWO-SAMPLE TEST (MOCK EXAM DATA)
# ============================================

library(ggplot2)
library(patchwork)

# 1. Input the Mock Exam Dataset
t_pts <- c(0.2, 0.5, 0.8)

Sample_A <- matrix(c(3.2, 4.5, 3.8,
                     3.0, 4.2, 3.5,
                     3.5, 4.8, 4.0), nrow = 3, byrow = TRUE)

Sample_B <- matrix(c(6.1, 7.5, 6.8,
                     5.9, 7.0, 6.5,
                     6.4, 7.8, 7.0), nrow = 3, byrow = TRUE)

# Helper function to convert matrix to long format for ggplot
mat_to_long <- function(mat, t_seq, grp) {
  n <- nrow(mat); p <- ncol(mat)
  data.frame(
    id    = rep(seq_len(n), each = p),
    group = grp,
    time  = rep(t_seq, times = n),
    value = as.vector(t(mat))
  )
}

# ============================================
# 2. Plotting the Functional Data
# ============================================

df_mock <- rbind(mat_to_long(Sample_A, t_pts, "Sample A"),
                 mat_to_long(Sample_B, t_pts, "Sample B"))

mn_mock <- data.frame(
  time  = rep(t_pts, 2),
  value = c(colMeans(Sample_A), colMeans(Sample_B)),
  group = rep(c("Sample A", "Sample B"), each = length(t_pts))
)

p_traj <- ggplot(df_mock, aes(time, value, group = interaction(id, group), colour = group)) +
  geom_line(linewidth = 0.8, linetype = "dotted") +
  geom_line(data = mn_mock, aes(group = group), linewidth = 2) +
  geom_point(size = 3) + 
  scale_colour_manual(values = c("Sample A" = "#2166AC", "Sample B" = "#D6604D")) +
  labs(title = "Functional Data: Sample A vs Sample B",
       subtitle = "Dotted = individual curves; Solid Bold = group mean",
       x = "Time (t)", y = "Value X(t)", colour = NULL) +
  theme_minimal(base_size = 11) + theme(legend.position = "bottom")

print(p_traj)

# ============================================
# 3. Two-Sample Permutation Test Functions
# ============================================

cov_kernel <- function(X) {
  Xc <- scale(X, center = TRUE, scale = FALSE)
  (t(Xc) %*% Xc) / (nrow(X) - 1)
}

two_sample_test <- function(X1, X2, B = 999, seed = 42, delta = 1) {
  set.seed(seed)
  n1 <- nrow(X1); n2 <- nrow(X2); N <- n1 + n2
  X_pool <- rbind(X1, X2)
  
  Tmu <- function(A, Bm) delta       * sum((colMeans(A) - colMeans(Bm))^2)
  Tcv <- function(A, Bm) (delta^2)   * sum((cov_kernel(A) - cov_kernel(Bm))^2)
  
  obs_mu <- Tmu(X1, X2); obs_cv <- Tcv(X1, X2); obs_cb <- obs_mu + obs_cv
  
  null_mu <- null_cv <- null_cb <- numeric(B)
  for (b in seq_len(B)) {
    idx <- sample(N)
    A   <- X_pool[idx[seq_len(n1)], , drop = FALSE]
    Bm  <- X_pool[idx[(n1 + 1):N],  , drop = FALSE]
    null_mu[b] <- Tmu(A, Bm)
    null_cv[b] <- Tcv(A, Bm)
    null_cb[b] <- null_mu[b] + null_cv[b]
  }
  
  list(obs_mu  = obs_mu,  obs_cv  = obs_cv,  obs_cb  = obs_cb,
       null_mu = null_mu, null_cv = null_cv, null_cb = null_cb,
       pval_mu = mean(null_mu >= obs_mu),
       pval_cv = mean(null_cv >= obs_cv),
       pval_cb = mean(null_cb >= obs_cb),
       B = B, n1 = n1, n2 = n2)
}

# Execute Test (Using B=50 because N is extremely small in this toy dataset)
dt <- diff(range(t_pts)) / (length(t_pts) - 1)
res_mock <- two_sample_test(Sample_A, Sample_B, B = 50, delta = dt)

# ============================================
# 4. Results Table
# ============================================

pvals <- c(res_mock$pval_mu, res_mock$pval_cv, res_mock$pval_cb)

tab <- data.frame(
  Statistic  = c("T_mu (Mean)", "T_C (Covariance)", "T_comb (Combined)"),
  Observed   = round(c(res_mock$obs_mu, res_mock$obs_cv, res_mock$obs_cb), 6),
  pvalue     = pvals,
  Decision   = ifelse(pvals < 0.05, "Reject H0", "Fail to reject H0")
)

cat("\n--- Permutation Test Results (B = 50) ---\n")
print(tab)

# ============================================
# 5. Plotting Null Distributions
# ============================================

plot_null <- function(null, obs, pval, ttl, xlab) {
  ggplot(data.frame(s = null), aes(x = s)) +
    geom_histogram(bins = 15, fill = "#CECBF6", colour = "white", linewidth = 0.3) +
    geom_vline(xintercept = obs, colour = "#E24B4A",
               linewidth = 1.2, linetype = "dashed") +
    annotate("text", x = obs, y = Inf,
             label = paste0("obs\np = ", round(pval, 3)),
             hjust = -0.12, vjust = 1.6, size = 3.5, colour = "#A32D2D") +
    labs(title = ttl, x = xlab, y = "Count") +
    theme_minimal(base_size = 10)
}

p_dists <- (plot_null(res_mock$null_mu, res_mock$obs_mu, res_mock$pval_mu,
                      "Mock Data: T_mu", "T_mu") +
              plot_null(res_mock$null_cv, res_mock$obs_cv, res_mock$pval_cv,
                        "Mock Data: T_cov",  "T_cov") +
              plot_null(res_mock$null_cb, res_mock$obs_cb, res_mock$pval_cb,
                        "Mock Data: T_comb", "T_comb")) +
  patchwork::plot_annotation(
    title    = "Permutation Null Distributions (B = 50)",
    subtitle = "Red dashed line = observed statistic")

print(p_dists)

# ============================================
# 6. Pointwise Bands
# ============================================

pw_bands <- function(X1, X2, B = 50, seed = 7) {
  set.seed(seed)
  n1 <- nrow(X1); N <- n1 + nrow(X2)
  pool <- rbind(X1, X2)
  obs  <- colMeans(X1) - colMeans(X2)
  pd   <- t(replicate(B, {
    idx <- sample(N)
    colMeans(pool[idx[seq_len(n1)], , drop = FALSE]) - 
      colMeans(pool[idx[(n1 + 1):N], , drop = FALSE])
  }))
  list(diff = obs,
       lo   = apply(pd, 2, quantile, 0.025),
       hi   = apply(pd, 2, quantile, 0.975))
}

pw_mock <- pw_bands(Sample_A, Sample_B, B = 50)

p_bands <- ggplot(data.frame(t = t_pts, d = pw_mock$diff,
                             lo = pw_mock$lo, hi = pw_mock$hi), aes(x = t)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), fill = "#C1D3F0", alpha = 0.7) +
  geom_line(aes(y = d), colour = "#2166AC", linewidth = 1.5) +
  geom_point(aes(y = d), colour = "#2166AC", size = 3) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
  labs(title = "Pointwise Mean Difference with 95% Permutation Bands",
       subtitle = "Sample A - Sample B",
       x = "Time (t)",
       y = "Difference in Means") +
  theme_minimal(base_size = 11)

print(p_bands)

# ============================================
# HW4 -- Problem 3 (Infinite d with independence or not)
# ============================================
# ============================================
# FUNCTIONAL INDEPENDENCE TEST
# ============================================

library(fda)
library(ggplot2)
library(patchwork)

set.seed(42)

# 1. Input the Mock Exam Dataset
t_seq <- c(0, 1)
n     <- 4

X_mat <- matrix(c( 0.5,  1.2,
                   -0.2,  0.8,
                   1.1,  2.5,
                   0.8,  1.9), nrow = 4, byrow = TRUE)

Y_mat <- matrix(c(10.5, 15.2,
                  8.0, 12.1,
                  14.2, 20.5,
                  12.0, 18.0), nrow = 4, byrow = TRUE)

cat("Data dimensions (subjects × time points):", dim(X_mat), "\n\n")

# ============================================
# 2. Linear Interpolation Answer (Exam Specific)
# ============================================
# Prompt: "Compute the sample functional cross-covariance at t=0.5"

# Since t=0.5 is exactly halfway between t=0 and t=1, 
# linear interpolation is just the average of the two endpoints.
X_mid <- (X_mat[, 1] + X_mat[, 2]) / 2
Y_mid <- (Y_mat[, 1] + Y_mat[, 2]) / 2

# Compute sample covariance at t=0.5
cross_cov_05 <- cov(X_mid, Y_mid)

cat("=== EXAM QUESTION SPECIFIC ANSWER ===\n")
cat("Sample functional cross-covariance at t=0.5 (Linear Interpolation):", 
    round(cross_cov_05, 4), "\n\n")

# ============================================
# 3. Plotting the Functional Curves
# ============================================

make_long <- function(mat, component) {
  df           <- as.data.frame(mat)
  colnames(df) <- as.character(round(t_seq, 4))
  df$curve_id  <- seq_len(n)
  df_long      <- reshape(df,
                          varying   = setdiff(names(df), "curve_id"),
                          v.names   = "value",
                          timevar   = "time",
                          times     = as.character(round(t_seq, 4)),
                          direction = "long")
  df_long$time      <- as.numeric(as.character(df_long$time))
  df_long$component <- component
  df_long[, c("curve_id", "time", "value", "component")]
}

df_all <- rbind(
  make_long(X_mat, "Component X"),
  make_long(Y_mat, "Component Y")
)

p_curves <- ggplot(df_all, aes(x = time, y = value, group = curve_id, colour = component)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  facet_wrap(~component, scales = "free_y") +
  scale_colour_manual(values = c("Component X" = "#534AB7", "Component Y" = "#0F6E56")) +
  scale_x_continuous(breaks = c(0, 1)) +
  labs(title = "Bivariate Functional Curves (4 Subjects)",
       x = "Time (t)", y = "Value") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none", strip.text = element_text(size = 11, face = "bold"))

print(p_curves)

# ============================================
# 4. Hilbert-Schmidt Independence Test
# ============================================

hs_stat <- function(X1, X2) {
  n  <- nrow(X1)
  Tg <- ncol(X1)
  
  # Centre each component (remove the mean function)
  X1c <- scale(X1, center = TRUE, scale = FALSE)
  X2c <- scale(X2, center = TRUE, scale = FALSE)
  
  # Empirical cross-covariance matrix (T × T)
  C12 <- (t(X1c) %*% X2c) / n
  
  # Squared Hilbert-Schmidt norm (normalised by T²)
  sum(C12^2) / Tg^2
}

perm_test <- function(X1, X2, B = 24, seed = 1) {
  set.seed(seed)
  n    <- nrow(X1)
  T0   <- hs_stat(X1, X2)      # observed test statistic
  null <- numeric(B)
  
  for (b in seq_len(B)) {
    perm    <- sample(n)        # random permutation of subject indices
    null[b] <- hs_stat(X1, X2[perm, , drop=FALSE])
  }
  
  p_val <- mean(null >= T0)
  list(stat = T0, null = null, p_value = p_val, B = B)
}

# Run for the mock dataset (Using B=24 because 4 subjects = 4! = 24 permutations)
res_mock <- perm_test(X_mat, Y_mat, B = 24)

cat("=== PERMUTATION TEST RESULTS (B = 24) ===\n")
cat("  HS statistic :", round(res_mock$stat, 6), "\n")
cat("  p-value      :", round(res_mock$p_value, 4), "\n")
cat("  Decision     :", ifelse(res_mock$p_value < 0.05, "Reject H0 (Dependent)", "Fail to reject H0 (Independent)"), "\n\n")

# ============================================
# 5. Null Distribution Plot
# ============================================

plot_null <- function(res, title_text) {
  df <- data.frame(stat = res$null)
  ggplot(df, aes(x = stat)) +
    geom_histogram(bins = 10, fill = "#CECBF6", colour = "white", linewidth = 0.3) +
    geom_vline(xintercept = res$stat, colour = "#E24B4A",
               linewidth = 1.2, linetype = "dashed") +
    annotate("text", x = res$stat, y = Inf,
             label = paste0("obs\np = ", round(res$p_value, 3)),
             hjust = -0.15, vjust = 1.6, size = 4, colour = "#A32D2D") +
    labs(title = title_text,
         x = expression(paste("||", hat(C)[12], "||"[HS]^2, " (permuted)")),
         y = "Count") +
    theme_minimal(base_size = 11)
}

p_null <- plot_null(res_mock, "Null Distribution of HS Statistic (Mock Exam Data)")
print(p_null)



## SVM and DD based classifier comp.

# Load required libraries
library(e1071)
library(klaR)
library(ddalpha)
library(ks)

# 1. Input the generic dataset
df <- data.frame(
  X1 = c(2.5, 2.0, 3.1, 1.5, 2.2, 5.5, 6.0, 5.1, 4.5, 6.2),
  X2 = c(3.1, 2.8, 3.2, 2.0, 2.5, 6.1, 5.8, 6.2, 5.0, 5.5),
  Class = as.factor(c("Class0", "Class0", "Class0", "Class0", "Class0", 
                      "Class1", "Class1", "Class1", "Class1", "Class1"))
)

# Helper function to compute empirical misclassification probability
calc_misclass_prob <- function(actual, predicted) {
  mean(as.character(actual) != as.character(predicted))
}

cat("\n--- Classification Results (Generic Dataset) ---\n")

# ==========================================
# 2. Depth-Based Classifier
# ==========================================
# Separate the data into the two classes
class0_data <- df[df$Class == "Class0", c("X1", "X2")]
class1_data <- df[df$Class == "Class1", c("X1", "X2")]

pred_depth <- character(nrow(df))

# Loop through each point and calculate its depth in both classes
for(i in 1:nrow(df)) {
  pt <- df[i, c("X1", "X2")]
  
  # Calculate Mahalanobis depth
  d0 <- depth.Mahalanobis(pt, class0_data)
  d1 <- depth.Mahalanobis(pt, class1_data)
  
  # Assign to the class where depth is higher
  if(d0 > d1) {
    pred_depth[i] <- "Class0"
  } else {
    pred_depth[i] <- "Class1"
  }
}

cat("Depth (Mahalanobis) Misclass Prob  : ", calc_misclass_prob(df$Class, pred_depth), "\n")

# ==========================================
# 3. SVM Classifiers
# ==========================================
svm_linear <- svm(Class ~ ., data = df, kernel = "linear", scale = FALSE)
pred_linear <- predict(svm_linear, df)
cat("SVM (Linear) Misclass Prob         : ", calc_misclass_prob(df$Class, pred_linear), "\n")

svm_radial <- svm(Class ~ ., data = df, kernel = "radial", scale = FALSE)
pred_radial <- predict(svm_radial, df)
cat("SVM (Radial) Misclass Prob         : ", calc_misclass_prob(df$Class, pred_radial), "\n")

# ==========================================
# 4. KDE Method 1: Automated (klaR::NaiveBayes)
# ==========================================
suppressWarnings({
  kde_nb_clf <- NaiveBayes(Class ~ ., data = df, usekernel = TRUE)
  pred_kde_nb <- predict(kde_nb_clf, df)$class
})
cat("KDE (Method 1: klaR) Misclass Prob : ", calc_misclass_prob(df$Class, pred_kde_nb), "\n")

# ==========================================
# 5. KDE Method 2: Multivariate Engine (ks::kde)
# ==========================================
X_features <- df[, c("X1", "X2")]
classes <- levels(df$Class)
pred_kde_raw <- character(nrow(df))

for (i in 1:nrow(df)) {
  x_test <- as.matrix(X_features[i, , drop = FALSE])
  densities <- numeric(length(classes))
  
  for (j in seq_along(classes)) {
    class_data <- as.matrix(X_features[df$Class == classes[j], ])
    H_band <- tryCatch(Hns(class_data), error = function(e) diag(ncol(class_data)))
    class_kde <- kde(class_data, H = H_band)
    densities[j] <- predict(class_kde, x = x_test)
  }
  
  pred_kde_raw[i] <- classes[which.max(densities)]
}

pred_kde_raw <- factor(pred_kde_raw, levels = classes)
cat("KDE (Method 2: ks) Misclass Prob   : ", calc_misclass_prob(df$Class, pred_kde_raw), "\n")


## Help Snippet
# 1. Data Depth Classifier

library(ddalpha)

# Given test point 'pt' and two training sets 'class0_data', 'class1_data'
d0 <- depth.Mahalanobis(pt, class0_data)
d1 <- depth.Mahalanobis(pt, class1_data)

predicted_class <- ifelse(d0 > d1, "Class0", "Class1")

# 2. SVM Classifier with Different Kernels

library(e1071)

svm_linear <- svm(Class ~ ., data = df, kernel = "linear", scale = FALSE)
pred_linear <- predict(svm_linear, df)

svm_radial <- svm(Class ~ ., data = df, kernel = "radial", scale = FALSE)
pred_radial <- predict(svm_radial, df)

# 3. K-Means Cluster

data_scaled <- scale(data)

K <- 2
kmeans_res <- kmeans(data_scaled, centers = K, nstart = 25)

k_labels <- kmeans_res$cluster

# 4. Data Depth Cluster

depth_function <- function(x, cluster_data) {
  mu <- colMeans(cluster_data)
  S  <- cov(cluster_data)
  if (det(S) == 0 || is.na(det(S))) S <- S + diag(1e-6, ncol(S))
  return(1 / (1 + mahalanobis(x, mu, S)))
}

clusters <- kmeans(data_scaled, centers = K)$cluster

# Iterative depth-based reassignment
for (iter in 1:50) {
  new_clusters <- clusters
  for (i in 1:nrow(data_scaled)) {
    depths <- numeric(K)
    for (k in 1:K) {
      cluster_data <- data_scaled[clusters == k, , drop = FALSE]
      depths[k] <- ifelse(nrow(cluster_data) > ncol(data_scaled) + 1,
                          depth_function(data_scaled[i, ], cluster_data), 0)
    }
    new_clusters[i] <- which.max(depths)
  }
  if (all(new_clusters == clusters)) break # Convergence
  clusters <- new_clusters
}

# 5. Multivariate Normal Check Using Data Depth

library(ddalpha)
library(MASS)

mu_hat <- colMeans(X)
Sigma_hat <- cov(X)

# Simulate fitted multivariate normal sample
Y_norm <- mvrnorm(n = nrow(X), mu = mu_hat, Sigma = Sigma_hat)
Z <- rbind(X, Y_norm)

# Calculate depths for both datasets
dX_mah <- depth.Mahalanobis(x = Z, data = X)
dY_mah <- depth.Mahalanobis(x = Z, data = Y_norm)

# Plot DD-Plot to visually check alignment
plot(dX_mah, dY_mah, main = "DD Plot: Mahalanobis", xlab = "Observed", ylab = "Fitted MVN")
abline(0, 1, col = "red")

# 6. Regression Quantile

library(quantreg)

taus <- c(0.10, 0.50, 0.90)
qr_models <- list()

for (tau in taus) {
  qr_models[[paste0("tau_", tau)]] <- rq(Y ~ X1 + X2, tau = tau, data = data)
}

# 7. Non-Parametric Regression Quantile

library(quantreg)

# Fit non-parametric quantile regression using smoothing splines
# lambda controls the smoothness penalty
taus_np <- c(0.25, 0.50, 0.75)
qrss_models <- list()

for (tau in taus_np) {
  qrss_models[[paste0("tau_", tau)]] <- rqss(Y ~ qss(X, lambda = 0.5), tau = tau, data = data)
}

# 8. Kernel Estimation

kde_gauss <- density(data, kernel = "gaussian")
kde_epan  <- density(data, kernel = "epanechnikov")
kde_rect  <- density(data, kernel = "rectangular")

plot(kde_gauss, col = "blue", main = "Kernel Density Estimates")
lines(kde_epan, col = "purple")

# 9. Nadaraya-Watson Estimator

h <- 0.1 # Bandwidth
nw_fit <- ksmooth(x = data$x, y = data$y, kernel = "normal", bandwidth = h, x.points = data$x)

# Extract predicted values
predictions <- nw_fit$y

# 10. Draw Data Depth and Multivariate Contours

library(ggplot2)
library(ddalpha)

# Create a spatial grid over the data range
x_seq <- seq(min(df$x)-1, max(df$x)+1, length.out = 50)
y_seq <- seq(min(df$y)-1, max(df$y)+1, length.out = 50)
grid_df <- expand.grid(x = x_seq, y = y_seq)

# Compute depth for each grid point
grid_df$depth <- depth.Mahalanobis(x = grid_df, data = df[, c("x", "y")])

ggplot() +
  geom_point(data = df, aes(x = x, y = y)) +
  geom_contour(data = grid_df, aes(x = x, y = y, z = depth, color = after_stat(level))) +
  scale_color_viridis_c(name = "Depth") +
  theme_minimal()



