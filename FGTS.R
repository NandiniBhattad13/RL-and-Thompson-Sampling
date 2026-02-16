set.seed(123)
library(MASS)

# Parameters
T <- 1000          
K <- 5              
d_context <- 4      
d <- K * d_context  
eta <- 1
sigma0 <- 0.01*eta
sigma <- 1  # variance of noise

# FG / LMC hyperparameters
eta <- 1
K_lmc <- 25      
lambda_fg <- 0.5*eta
b_cap <- 1000
h <- 1e-4

# true parameter vector
theta_star <- rnorm(d, mean = 0, sd = 1)

# Feature map
phi_map <- function(X, a) 
{
  vec <- rep(0, d)
  st <- (a - 1) * d_context + 1
  end_idx <- st + d_context - 1
  vec[st:end_idx] <- X
  return(vec)
}

regrets <- numeric(T)

theta_curr <- rep(0, d)   

# Accumulators
A <- matrix(0, d, d)   # sum φφᵀ
b_vec <- rep(0, d)     # sum φ r
# store past contexts (needed for FG term)
X_history <- list()   

# Gradient function (efficient)
grad_loss_fg <- function(theta) 
{
  grad_prior <- theta / (sigma0^2)
  grad_sqerr <- 2 * eta * (A %*% theta - b_vec)
  grad_fg <- rep(0, d)
  
  if (length(X_history) > 0) 
  {
    for (X in X_history)
    {
      vals <- sapply(1:K, function(a) sum(phi_map(X,a) * theta))
      a_star <- which.max(vals)
      
      if (vals[a_star] < b_cap)
        grad_fg <- grad_fg + phi_map(X, a_star)
    }
  }
  return (grad_prior + grad_sqerr - (lambda_fg * grad_fg))
}

for (t in 1:T) 
{
  # Context
  X_t <- rnorm(d_context, mean = 0, sd = 1)
  
  # LMC updates
  for (k in 1:K_lmc) 
  {
    noise <- rnorm(d)
    theta_curr <- theta_curr - h * grad_loss_fg(theta_curr) + sqrt(2 * h) * noise
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
  
  # Regret
  regrets[t] <- r_star_t - sum(phi_map(X_t, a_t) * theta_star)
  
  # Update accumulators
  A <- A + tcrossprod(phi_t)
  b_vec <- b_vec + phi_t * r_t
  # store context for feel-good term
  X_history[[length(X_history)+1]] <- X_t
}

# Evaluation
cumulative_regret <- cumsum(regrets)
cat(sprintf("Final cumulative regret: %.2f\n", cumulative_regret[T]))
plot(cumulative_regret, type = "l", main = "Cumulative Regret",
     xlab = "Round", ylab = "Regret")
