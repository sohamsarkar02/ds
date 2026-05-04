# ===================
## HW 3--Problem 1 (Ckeecing MVN using DD)
# ===================

if (!require("mvnTest")) install.packages("mvnTest")
if (!require("fasano.franceschini.test"))      install.packages("fasano.franceschini.test")
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


