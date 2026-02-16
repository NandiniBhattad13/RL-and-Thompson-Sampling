set.seed(123)
library(MASS)

# Parameters
T <- 1000    
K <- 5
d_context <- 4
d <- K * d_context
eta <- 1
sigma <- 1   # variance of noise
sigma0 <- 0.01 * eta # prior std dev

# LMC hyperparameters
h <- 1e-4
K_lmc <- 25

# True parameter vector
theta_star <- rnorm(d, mean = 0, sd = 1)

# Feature map
phi_map <- function(X, i) 
{
  vec <- rep(0, d)
  start_idx <- (i - 1) * d_context + 1
  vec[start_idx:(start_idx + d_context - 1)] <- X
  return(vec)
}

# Gradient 
grad_loss <- function(theta, A, b, sigma, sigma0) 
{
  grad <- theta / (sigma0^2)
  if (sum(A) > 0)
  {
    grad <- grad + (2 * eta * (A %*% theta - b))
  }
  return(as.vector(grad))
}

# Tracking
regrets <- numeric(T)
chosen_arms <- numeric(T)
optimal_arms <- numeric(T)
errors1 <- numeric(T)

# Initialize theta
theta_curr <- rep(0, d)

# Sufficient statistics for gradient
A <- matrix(0, d, d)   # Σ φ φ^T
b <- rep(0, d)         # Σ r φ

for (t in 1:T) 
{
  X_t <- rnorm(d_context, mean = 0, sd = 1)
  
  # LMC updates
  for (k in 1:K_lmc) 
  {
    noise <- rnorm(d)
    theta_curr <- theta_curr - h * grad_loss(theta_curr, A, b, sigma, sigma0) + sqrt(2 * h) * noise
  }
  
  # Action selection
  pred_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_curr))
  a_t <- which.max(pred_rewards)
  
  # Reward
  eps_t <- rnorm(1, mean = 0, sd = sigma)
  phi_t <- phi_map(X_t, a_t)
  r_t <- sum(phi_t * theta_star) + eps_t
  
  # Optimal reward
  opt_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_star))
  r_star_t <- max(opt_rewards)
  
  regrets[t] <- r_star_t - sum(phi_map(X_t, a_t) * theta_star)
  
  # Update sufficient statistics
  phi_t <- phi_map(X_t, a_t)
  A <- A + tcrossprod(phi_t)
  b <- b + r_t * phi_t
}
cumulative_regret <- cumsum(regrets)

cat(sprintf("Final cumulative regret: %.2f\n", cumulative_regret[T]))
# Plot cumulative regret
plot(cumulative_regret, type = "l", main = "Cumulative Regret",
     xlab = "Round", ylab = "Regret")
