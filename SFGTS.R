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

# LMC & SFG hyperparameters
h     <- 1e-4
K_lmc <- 25
eta        <- 1
lambda_sfg <- 0.5*eta
b_cap      <- 1000
s_param    <- 5.0

# true parameter vector
theta_star <- rnorm(d, mean = 0, sd = 1)
theta_curr <- rep(0,d)

phi_map <- function(X, a) 
{
  vec <- rep(0, d)
  st <- (a - 1) * d_context + 1
  end_idx <- st + d_context - 1
  vec[st:end_idx] <- X
  return(vec)
}

sigmoid_stable <- function(z) 
{
  ifelse(z >= 0, 1 / (1 + exp(-z)), exp(z) / (1 + exp(z)))
}

# Gradient 
grad_loss_sfg <- function(theta, A, b, ctx)
{
  grad_prior <- theta / (sigma0^2)
  grad_sqerr <- if (sum(A) > 0) 2 * eta * (A %*% theta - b) else rep(0, d)
  # SFG term (only depends on current context)
  vals <- sapply(1:K, function(a) sum(phi_map(ctx, a) * theta))
  a_star <- which.max(vals) 
  phi_max <- phi_map(ctx, a_star)
  
  z <- s_param * (b_cap - vals[a_star])
  grad_sfg <- lambda_sfg * sigmoid_stable(z) * phi_max
  
  return (grad_prior + grad_sqerr - grad_sfg)
}

regrets <- numeric(T)

# Statistics for fast gradient updates
A <- matrix(0, d, d)  # ∑ φ φ^T
b <- rep(0, d)        # ∑ φ r


for (t in 1:T) 
{
  # New context
  X_t <- rnorm(d_context, mean = 0, sd = 1)
  
  # LMC updates
  for (k in 1:K_lmc) 
  {
    noise <- rnorm(d)
    grad <- grad_loss_sfg(theta_curr, A, b, X_t)
    theta_curr <- theta_curr - h * grad + sqrt(2*h) * noise
  }
  
  # Predict rewards and choose arm
  pred_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_curr))
  
  # True reward
  eps_t <- rnorm(1, mean = 0, sd = sigma)
  a_t <- which.max(pred_rewards)
  r_t <- sum(phi_map(X_t, a_t) * theta_star) + eps_t
  # Regret
  true_rewards <- sapply(1:K, function(a) sum(phi_map(X_t, a) * theta_star))
  regrets[t] <- max(true_rewards) - true_rewards[a_t]
  
  # Update statistics
  phi_t <- phi_map(X_t, a_t)
  A <- A + tcrossprod(phi_t)  # φ φ^T
  b <- b + phi_t * r_t
}

cumulative_regret <- cumsum(regrets)
cat(sprintf("Final cumulative regret: %.2f\n", cumulative_regret[T]))

# cumulative regret
plot(cumulative_regret, type = "l", main = "Cumulative Regret",
     xlab = "Round", ylab = "Regret")
