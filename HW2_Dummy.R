
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
