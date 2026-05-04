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
