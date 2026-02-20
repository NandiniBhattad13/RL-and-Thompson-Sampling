set.seed(123)
library(MASS)

# ----------------------- Problem setup -----------------------
T <- 1000                 # Horizon
K <- 2                    # Number of arms
d_context <- 4            # Context dimension
d <- K * d_context        # Parameter dimension (block per arm)
eta <- 1
sigma <- 1                # Reward noise std
sigma0 <- 0.01*eta        # Prior std

# ----------------------- Sampler hyperparameters -----------------------
h_lmc <- 1e-4             # Langevin stepsize
h_barker <- 0.1           # Barker proposal scale
K_lmc <- 25               # LMC inner iterations
K_barker <- 25            # Barker inner iterations
lambda_fg <- 0.5*eta      # Feel-good regularization strength
lambda_my <- 5            # Moreau envelope smoothing width
b_cap <- 10               # Optimism reward cap
s_param <- 5.0            # Sigmoid sharpness (smooth FGTS)

# ----------------------- Ground truth model -----------------------
theta_star <- rnorm(d)
theta_star_mat <- matrix(theta_star, d_context, K)

# Shared contexts/noise so all algorithms see identical environment
X_all <- matrix(rnorm(T * d_context), T, d_context)
noise_all <- rnorm(T, 0, sigma)

# Feature mapping: block-vector for chosen arm
get_phi_matrix <- function(X, a, K, d_context) {
  vec <- rep(0, K * d_context)
  st <- (a - 1) * d_context + 1
  vec[st:(st + d_context - 1)] <- X
  return(vec)
}

# Numerically stable sigmoid used in SFGTS
sigmoid_stable <- function(z) {
  ifelse(z >= 0, 1 / (1 + exp(-z)), exp(z) / (1 + exp(z)))
}

# Gradient of Moreau-Yosida envelope of hinge(b − f)
grad_my_envelope <- function(x, a, lam) {
  ifelse(x <= a - lam, -1, ifelse(x >= a, 0, (x - a) / lam))
}

# ================================================================
# 1. Exact Linear Thompson Sampling (Gaussian posterior)
# ================================================================
mu <- rep(0, d) 
Sigma <- diag(sigma0^2, d)
regrets_linear <- numeric(T)

for (t in 1:T) {
  X_t <- X_all[t,]
  
  theta_sample <- MASS::mvrnorm(1, mu, Sigma)      # Exact posterior sample
  a_t <- which.max(as.vector(X_t %*% matrix(theta_sample, d_context, K)))
  
  r_star_t <- max(as.vector(X_t %*% theta_star_mat))
  phi_t <- get_phi_matrix(X_t, a_t, K, d_context)
  regrets_linear[t] <- r_star_t - sum(phi_t * theta_star)
  
  r_obs <- sum(phi_t * theta_star) + noise_all[t]  # Observed reward
  
  # Bayesian linear regression update
  S_inv <- solve(Sigma)
  Sigma <- solve(S_inv + (1/sigma^2)*tcrossprod(phi_t))
  mu <- Sigma %*% (S_inv %*% mu + (1/sigma^2)*phi_t*r_obs)
}

# ================================================================
# 2. LMC Thompson Sampling (approx posterior sampling)
# ================================================================
theta_curr <- rep(0, d); A <- matrix(0, d, d); b_vec <- rep(0, d)
regrets_lmc <- numeric(T)

for (t in 1:T) {
  X_t <- X_all[t,]
  
  for (k in 1:K_lmc) {
    grad <- (theta_curr/sigma0^2) + (2*eta*(A %*% theta_curr - b_vec)) # del(-log posterior)
    theta_curr <- theta_curr - h_lmc * grad + sqrt(2*h_lmc) * rnorm(d) # Langevin step
  }
  
  a_t <- which.max(as.vector(X_t %*% matrix(theta_curr, d_context, K)))
  phi_t <- get_phi_matrix(X_t, a_t, K, d_context)
  regrets_lmc[t] <- max(as.vector(X_t %*% theta_star_mat)) - sum(phi_t * theta_star)
  
  # Update sufficient statistics (A,b)
  A <- A + tcrossprod(phi_t)
  b_vec <- b_vec + phi_t*(sum(phi_t * theta_star) + noise_all[t])
}

# ================================================================
# 3. FGTS — hard optimism bonus
# ================================================================
theta_curr <- rep(0, d); A <- matrix(0, d, d); b_vec <- rep(0, d)
regrets_fg <- numeric(T); X_hist <- matrix(0, T, d_context)

for (t in 1:T) {
  X_t <- X_all[t,]; X_hist[t,] <- X_t
  X_curr_h <- X_hist[1:t, , drop=FALSE]
  
  for (k in 1:K_lmc) {
    g_base <- (theta_curr/sigma0^2) + (2*eta*(A %*% theta_curr - b_vec))
    
    all_rew <- X_curr_h %*% matrix(theta_curr, d_context, K) # predicted rewards
    best_a <- max.col(all_rew)
    g_fg <- rep(0, d)
    
    # add gradient only if predicted reward below optimism cap
    idx_fg <- which(all_rew[cbind(1:t, best_a)] < b_cap)
    if(length(idx_fg)>0){
      for(i in idx_fg) g_fg <- g_fg + get_phi_matrix(X_curr_h[i,], best_a[i], K, d_context)
    }
    
    theta_curr <- theta_curr - h_lmc * (g_base - lambda_fg * g_fg) + sqrt(2*h_lmc) * rnorm(d)
  }
  
  a_t <- which.max(as.vector(X_t %*% matrix(theta_curr, d_context, K)))
  phi_t <- get_phi_matrix(X_t, a_t, K, d_context)
  regrets_fg[t] <- max(as.vector(X_t %*% theta_star_mat)) - sum(phi_t * theta_star)
  
  A <- A + tcrossprod(phi_t)
  b_vec <- b_vec + phi_t*(sum(phi_t * theta_star) + noise_all[t])
}

# ================================================================
# 4. SFGTS — smoothed optimism (sigmoid)
# ================================================================
theta_curr <- rep(0, d); A <- matrix(0, d, d); b_vec <- rep(0, d)
regrets_sfg <- numeric(T); X_hist <- matrix(0, T, d_context)

for (t in 1:T) {
  X_t <- X_all[t,]; X_hist[t,] <- X_t
  X_curr_h <- X_hist[1:t, , drop=FALSE]
  
  for (k in 1:K_lmc) {
    g_base <- (theta_curr/sigma0^2) + (2*eta*(A %*% theta_curr - b_vec))
    
    all_rew <- X_curr_h %*% matrix(theta_curr, d_context, K)
    best_a <- max.col(all_rew)
    
    # smooth approximation of indicator using sigmoid
    weights <- lambda_fg * sigmoid_stable(s_param * (b_cap - all_rew[cbind(1:t, best_a)]))
    
    g_sfg <- rep(0, d)
    for(i in 1:t)
      g_sfg <- g_sfg + weights[i] * get_phi_matrix(X_curr_h[i,], best_a[i], K, d_context)
    
    theta_curr <- theta_curr - h_lmc * (g_base - g_sfg) + sqrt(2*h_lmc) * rnorm(d)
  }
  
  a_t <- which.max(as.vector(X_t %*% matrix(theta_curr, d_context, K)))
  phi_t <- get_phi_matrix(X_t, a_t, K, d_context)
  regrets_sfg[t] <- max(as.vector(X_t %*% theta_star_mat)) - sum(phi_t * theta_star)
  
  A <- A + tcrossprod(phi_t)
  b_vec <- b_vec + phi_t*(sum(phi_t * theta_star) + noise_all[t])
}

# ================================================================
# 5. MY-LMC — Moreau-Yosida smoothed optimism
# ================================================================
theta_curr <- rep(0, d); A <- matrix(0, d, d); b_vec <- rep(0, d)
regrets_mylmc <- numeric(T); X_hist <- matrix(0, T, d_context)

for (t in 1:T) {
  X_t <- X_all[t,]; X_hist[t,] <- X_t
  X_curr_h <- X_hist[1:t, , drop=FALSE]
  
  for (k in 1:K_lmc) {
    g_base <- (theta_curr/sigma0^2) + (2*eta*(A %*% theta_curr - b_vec))
    
    all_rew <- X_curr_h %*% matrix(theta_curr, d_context, K)
    best_a <- max.col(all_rew)
    
    # gradient of Moreau envelope
    g_env <- grad_my_envelope(all_rew[cbind(1:t, best_a)], b_cap, lambda_my)
    g_fg <- rep(0, d)
    idx_my <- which(g_env != 0)
    if(length(idx_my)>0){
      for(i in idx_my)
        g_fg <- g_fg + g_env[i] * get_phi_matrix(X_curr_h[i,], best_a[i], K, d_context)
    }
    
    theta_curr <- theta_curr - h_lmc * (g_base + lambda_fg * g_fg) + sqrt(2*h_lmc) * rnorm(d)
  }
  
  a_t <- which.max(as.vector(X_t %*% matrix(theta_curr, d_context, K)))
  phi_t <- get_phi_matrix(X_t, a_t, K, d_context)
  regrets_mylmc[t] <- max(as.vector(X_t %*% theta_star_mat)) - sum(phi_t * theta_star)
  
  A <- A + tcrossprod(phi_t)
  b_vec <- b_vec + phi_t*(sum(phi_t * theta_star) + noise_all[t])
}

# ================================================================
# 6. MY-Barker — same posterior, Barker MCMC kernel
# ================================================================
theta_curr <- rep(0, d); A <- matrix(0, d, d); b_vec <- rep(0, d)
regrets_myb <- numeric(T); X_hist <- matrix(0, T, d_context)

for (t in 1:T) {
  X_t <- X_all[t,]; X_hist[t,] <- X_t
  X_curr_h <- X_hist[1:t, , drop=FALSE]
  
  for (k in 1:K_barker) {
    g_base <- (theta_curr/sigma0^2) + (2*eta*(A %*% theta_curr - b_vec))
    
    all_rew <- X_curr_h %*% matrix(theta_curr, d_context, K)
    best_a <- max.col(all_rew)
    
    g_env <- grad_my_envelope(all_rew[cbind(1:t, best_a)], b_cap, lambda_my)
    g_fg <- rep(0, d)
    idx_my <- which(g_env != 0)
    if(length(idx_my)>0){
      for(i in idx_my)
        g_fg <- g_fg + g_env[i] * get_phi_matrix(X_curr_h[i,], best_a[i], K, d_context)
    }
    
    g_total <- g_base + (lambda_fg * g_fg)
    
    # Barker proposal 
    w <- rnorm(d, 0, h_barker)
    p <- plogis(-0.5 * g_total * w)
    theta_curr <- theta_curr + ifelse(runif(d) < p, 1, -1) * w
  }
  
  a_t <- which.max(as.vector(X_t %*% matrix(theta_curr, d_context, K)))
  phi_t <- get_phi_matrix(X_t, a_t, K, d_context)
  regrets_myb[t] <- max(as.vector(X_t %*% theta_star_mat)) - sum(phi_t * theta_star)
  
  A <- A + tcrossprod(phi_t)
  b_vec <- b_vec + phi_t*(sum(phi_t * theta_star) + noise_all[t])
}

# ----------------------- Plot regret decay -----------------------
time_vec <- 1:T
palette <- c("black", "blue", "red", "green", "darkorange", "purple")
methods <- list(regrets_linear, regrets_lmc, regrets_fg, regrets_sfg, regrets_mylmc, regrets_myb)
method_names <- c("LinearTS", "LMCTS", "FGTS", "SFGTS", "MYLMC", "MYBarker")

plot(cumsum(methods[[1]])/time_vec, type="l", col=palette[1], lwd=2,
     ylab="Average Regret (Rt/t)", xlab="Time (t)", main="Regret Decay (Shared Contexts)",
     ylim=c(0, max((cumsum(methods[[1]])/time_vec)[20:T])))

for(i in 2:6) lines(cumsum(methods[[i]])/time_vec, col=palette[i], lwd=2)
legend("topright", legend=method_names, col=palette, lty=1, lwd=2, cex=0.8)
