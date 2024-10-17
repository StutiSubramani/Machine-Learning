# Install and load necessary package
install.packages("mvtnorm")
library(mvtnorm)

#Coordinate descent algorithm for Lasso -------------------------------------------------------------
lasso_coordinate_descent <- function(X, y, lambda, max_iter = 500, tol = 1e-6) {
  # Get the number of observations (n) and predictors (p)
  n <- nrow(X)
  p <- ncol(X)
  
  # Initialize the coefficient vector (beta) with zeros
  beta <- rep(0, p)
  
  # Initialize the iteration counter
  iter <- 1
  
  # Main loop for coordinate descent
  while (iter <= max_iter) {
    # Save the current coefficient vector for convergence check
    beta_old <- beta
    
    # Loop through each predictor for coordinate updates
    for (j in 1:p) {
      # Compute partial residuals
      rij <- y - (X %*% beta - X[, j] * beta[j])
      
      # Compute the least squares coefficient of residuals on j-th predictor
      beta_star_j <- sum(X[, j] * rij) / n
      
      # Soft thresholding operation to update the j-th coefficient
      beta[j] <- sign(beta_star_j) * max(abs(beta_star_j) - lambda, 0)
    }
    
    # Check for convergence: if the maximum change in coefficients is below tolerance, exit the loop
    if (max(abs(beta - beta_old)) < tol) {
      break
    }
    
    # Increment the iteration counter
    iter <- iter + 1
  }
  
  # Return the final coefficient vector
  return(beta)
}


#Coordinate descent algorithm for Elastic net -----------------------------------------------------------------------
elastic_net_coordinate_descent <- function(X, y, lambda1, lambda2, max_iter = 500, tol = 1e-6) {
  # Get the number of observations (n) and predictors (p)
  n <- nrow(X)
  p <- ncol(X)
  
  # Initialize the coefficient vector (beta) with zeros
  beta <- rep(0, p)
  
  # Initialize the iteration counter
  iter <- 1
  
  # Main loop for coordinate descent
  while (iter <= max_iter) {
    # Save the current coefficient vector for convergence check
    beta_old <- beta
    
    # Loop through each predictor for coordinate updates
    for (j in 1:p) {
      # Compute partial residuals
      rij <- y - (X %*% beta - X[, j] * beta[j])
      
      # Compute the least squares coefficient of residuals on j-th predictor
      beta_star_j <- sum(X[, j] * rij) / n
      
      # Soft thresholding operation with elastic net penalty
      beta[j] <- (1 / (1 + 2 * lambda2)) * sign(beta_star_j) * max(abs(beta_star_j) - lambda1, 0)
    }
    
    # Check for convergence: if the maximum change in coefficients is below tolerance, exit the loop
    if (max(abs(beta - beta_old)) < tol) {
      break
    }
    
    # Increment the iteration counter
    iter <- iter + 1
  }
  
  # Return the final coefficient vector
  return(beta)
}


#Cross Validation for Lasso------------------------------------------------------------
cv_lasso <- function(lambda_max = 5, step_lambda = 0.1, n_folds = 2, y, X, one_stderr_rule = TRUE) {
  
  # Function to perform one pass of cross-validation for a given lambda
  cv.one_pass <- function(lambda) {
    errors <- numeric(n_folds)
    
    # Iterate over folds for cross-validation
    for (i in 1:n_folds) {
      Xtest <- X[X[, 1] != i, -1]
      ytest <- Y[Y[, 1] != i, -1]
      Xtrain <- X[X[, 1] == i, -1]
      ytrain <- Y[Y[, 1] == i, -1]
      
      # Fit Lasso regression on the training set
      beta_lasso <- lasso_coordinate_descent(Xtrain, ytrain, lambda)
      
      # Make predictions on the test set
      ypred <- Xtest %*% beta_lasso
      
      # Calculate mean squared error and store in errors vector
      errors[i] <- mean((ytest - ypred)^2)
    }
    
    # Return a list containing mean error and standard deviation of errors
    return(list(error = mean(errors), std = sd(errors)))
  }
  
  # Generate a sequence of lambda values
  lambdas <- seq(0, lambda_max, step_lambda)
  
  # Initialize a data frame to store results
  df <- data.frame(lambda = lambdas, estimated_error = numeric(length(lambdas)), std = numeric(length(lambdas)))
  
  # Iterate over lambda values to perform cross-validation
  for (l in 1:length(df$lambda)) {
    one_pass <- cv.one_pass(lambda = df$lambda[l])
    df$estimated_error[l] <- one_pass$error
    df$std[l] <- one_pass$std
  }
  
  # Calculate upper and lower bounds of the confidence interval
  df$ci_up <- df$estimated_error + df$std
  df$ci_down <- df$estimated_error - df$std
  
  # Determine the best lambda based on the one standard error rule or minimum estimated error
  if (!one_stderr_rule) {
    best_lambda <- df$lambda[which.min(df$estimated_error)]
  } else {
    thresh <- df$ci_up[which.min(df$estimated_error)]
    best_lambda <- max(df$lambda[df$estimated_error <= thresh])
  }
  
  # Mark the row corresponding to the best lambda
  df$is_best <- df$lambda == best_lambda
  
  # Return a list containing error data frame and the best lambda
  return(list(errors = df, best_lambda = best_lambda))
}


#Cross Validation for Elastic net-------------------------------------------------------
cv_EN <- function(lambda1_max = 10, step_lambda1 = .1, n_folds = 2, y, X, one_stderr_rule = TRUE) {
  
  # Function to perform one pass of cross-validation for a given lambda1 and lambda2
  cv.one_passEN <- function(lambda1, lambda2) {
    
    # Target errors vector
    errors <- numeric(n_folds)
    
    # Iterate over folds for cross-validation
    for (i in 1:n_folds) {
      Xtest <- X[X[, 1] != i, -1]
      ytest <- Y[Y[, 1] != i, -1]
      Xtrain <- X[X[, 1] == i, -1]
      ytrain <- Y[Y[, 1] == i, -1]
      
      # Fit Elastic Net regression on the training set
      beta_EN <- elastic_net_coordinate_descent(Xtrain, ytrain, lambda1, lambda2)
      
      # Make predictions on the test set
      ypred <- Xtest %*% beta_EN
      
      # Calculate mean squared error and store in errors vector
      errors[i] <- mean((ytest - ypred)^2)
    }
    
    # Return a list containing mean error and standard deviation of errors
    return(list(error = mean(errors), std = sd(errors)))
  }
  
  # Initialize grid of lambda1s and lambda2s
  lambdas_1 <- seq(0, lambda1_max, step_lambda1)
  lambdas_2 <- seq(0, 1, 0.1)
  
  # Initialize a data frame to store results
  df <- data.frame(lambda1 = lambdas_1,
                   lambda2 = numeric(length(lambdas_1)),
                   estimated_error = numeric(length(lambdas_1)),
                   std = numeric(length(lambdas_1)))
  
  # Iterate over lambda values to perform cross-validation
  for (i in 1:length(df$lambda1)) {
    
    # For each lambda, choose the best lambda2 which gives the minimum error
    # Store the data in lambda_best, error, and std, always store the minimum one
    # Initiate with lambda2 = 0
    lambda2_best <- 0
    current_pass <- cv.one_passEN(lambda1 = df$lambda1[i], lambda2 = lambda2_best)
    error <- current_pass$error
    std <- current_pass$std
    
    for (j in 2:length(lambdas_2)) {
      current_pass <- cv.one_passEN(lambda = df$lambda1[i], lambda2 = lambda2_best[j])
      
      if (error > current_pass$error) {
        error <- current_pass$error
        std <- current_pass$std
        lambda2_best <- lambdas_2[j]
      }
    }
    
    # Put the selected lambda2, error, and std in the df table
    df$estimated_error[i] <- error
    df$std[i] <- std
    df$lambda2[i] <- lambda2_best
  }
  
  # Calculate upper and lower bounds of the one standard error
  df$ci_up <- df$estimated_error + df$std
  df$ci_down <- df$estimated_error - df$std
  
  # Determine the best lambda based on the one standard error rule or minimum estimated error
  if (!one_stderr_rule) {
    best_lambda1 <- df$lambda1[which.min(df$estimated_error)]
  } else {
    thresh <- df$ci_up[which.min(df$estimated_error)]
    best_lambda1 <- max(df$lambda1[df$estimated_error <= thresh])
  }
  
  # Mark the row corresponding to the best lambda
  df$is_best <- df$lambda1 == best_lambda1
  
  # Return a list containing error data frame, the best lambda, and the best lambda
  return(list(errors = df, best_lambda1 = best_lambda1, best_lambda2 = lambda2_best))
}


#Data Simulation and Algorithm Implementation--------------------------------------------

# Set simulation parameters
n <- 240
p <- 8
b <- c(3, 1.5, 0, 0, 2, 0, 0, 0)
r <- 50
sd <- 3

# Create a data frame to store simulation results
final <- data.frame(lambda = numeric(r), lambda_1 = numeric(r),
                    lambda_2 = numeric(r), beta_ct_lasso = numeric(r),
                    beta_ct_en = numeric(r), mse_lasso = numeric(r),
                    mse_en = numeric(r))

#1. For uncorrelated Xi's-----------------------------------------------------------------

# Run r simulations
for (i in 1:r) {
  
  # Set seed for reproducibility
  set.seed(i)
  
  # Generate folds for cross-validation
  pop <- rep(seq(1, round(n/20)), 20)
  folds <- sample(pop, length(pop))
  
  # Simulate uncorrelated data
  Var_x <- diag(1, p, p)
  X <- rmvnorm(n, sigma = Var_x)
  epsilon <- rnorm(n, sd = sd)
  Y <- X %*% b + epsilon
  X <- cbind(folds, X)
  Y <- cbind(folds, Y)
  
  # Estimate hyperparameters using cross-validation
  final$lambda[i] <- cv_lasso(n_folds = length(unique(folds)),y = Y, X = X)$best_lambda
  final$lambda_1[i] <- cv_EN(n_folds = length(unique(folds)),y = Y, X = X)$best_lambda1
  final$lambda_2[i] <- cv_EN(n_folds = length(unique(folds)),y = Y, X = X)$best_lambda2
  
  # Split data into training and test sets
  train_data_ind <- X[X[, 1] == 1 | X[, 1] == 2, -1]
  train_data_resp <- Y[Y[, 1] == 1 | Y[, 1] == 2, -1]
  test_data_ind <- X[X[, 1] != 1 | X[, 1] != 2, -1]
  test_data_resp <- Y[Y[, 1] != 1 | Y[, 1] != 2, -1]
  
  # Lasso
  beta_lasso <- lasso_coordinate_descent(train_data_ind, train_data_resp, final$lambda[i])
  final$beta_ct_lasso[i] <- length(beta_lasso[beta_lasso != 0])
  pred_lasso <- test_data_ind %*% beta_lasso
  final$mse_lasso[i] <- mean((test_data_resp - pred_lasso)^2)
  
  # Elastic Net
  beta_en <- elastic_net_coordinate_descent(train_data_ind, train_data_resp,
                                            final$lambda_1[i], final$lambda_2[i])
  final$beta_ct_en[i] <- length(beta_en[beta_en != 0])
  pred_en <- test_data_ind %*% beta_en
  final$mse_en[i] <- mean((test_data_resp - pred_en)^2)
}

#2. For correlated Xi's--------------------------------------------------------------

# Run r simulations
for (i in 1:r) {
  
  # Set seed for reproducibility
  set.seed(i)
  
  # Generate folds for cross-validation
  pop <- rep(seq(1, round(n/20)), 20)
  folds <- sample(pop, length(pop))
  
  # Simulate correlated data
  Var_x <- diag(1, p, p)
  for (j in 1:p){
    for (k in 1:p) {
      Var_x[k,j] <- 0.5^abs(j-k)
      
    }
  }
  X <- rmvnorm(n, sigma = Var_x)
  epsilon <- rnorm(n, sd = sd)
  Y <- X %*% b + epsilon
  X <- cbind(folds, X)
  Y <- cbind(folds, Y)
  
  # Estimate hyperparameters using cross-validation
  final$lambda[i] <- cv_lasso(n_folds = length(unique(folds)),y = Y, X = X)$best_lambda
  final$lambda_1[i] <- cv_EN(n_folds = length(unique(folds)),y = Y, X = X)$best_lambda1
  final$lambda_2[i] <- cv_EN(n_folds = length(unique(folds)),y = Y, X = X)$best_lambda2
  
  # Split data into training and test sets
  train_data_ind <- X[X[, 1] == 1 | X[, 1] == 2, -1]
  train_data_resp <- Y[Y[, 1] == 1 | Y[, 1] == 2, -1]
  test_data_ind <- X[X[, 1] != 1 | X[, 1] != 2, -1]
  test_data_resp <- Y[Y[, 1] != 1 | Y[, 1] != 2, -1]
  
  # Lasso
  beta_lasso <- lasso_coordinate_descent(train_data_ind, train_data_resp, final$lambda[i])
  final$beta_ct_lasso[i] <- length(beta_lasso[beta_lasso != 0])
  pred_lasso <- test_data_ind %*% beta_lasso
  final$mse_lasso[i] <- mean((test_data_resp - pred_lasso)^2)
  
  # Elastic Net
  beta_en <- elastic_net_coordinate_descent(train_data_ind, train_data_resp,
                                            final$lambda_1[i], final$lambda_2[i])
  final$beta_ct_en[i] <- length(beta_en[beta_en != 0])
  pred_en <- test_data_ind %*% beta_en
  final$mse_en[i] <- mean((test_data_resp - pred_en)^2)
}

#Plotting the comparison of Lasso and Elastic Net for a given simulation

plot(seq(1,50),final$mse_lasso,type="l",col="blue", xlab = "Dataset Number",
     ylab = "MSE", main = "Comparing Lasso and Elastic Net for a given simulation")
lines(seq(1,50),final$mse_en,col="red")
legend("topright",legend = c("Lasso","Elastic Net"),col=c("Blue","Red"),cex=0.6,lwd=2)

#Extraction of lambda using minimum of MSE (sample code for data stored in x,y)---------------------------

l <- cv_lasso(y = Y, X = X)$errors[,1]
se <- cv_lasso(y = Y, X = X)$errors[,2]
l_final <- cv_lasso(y = Y, X = X)$best_lambda


plot(l,se,ylim=c(15,28),type="o", main="Lambda Selection for Lasso",xlab="Lambda",ylab="MSE",
     xlim=c(0.5,3), col=ifelse(l %in% l_final,'red','black'))

