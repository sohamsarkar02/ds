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
