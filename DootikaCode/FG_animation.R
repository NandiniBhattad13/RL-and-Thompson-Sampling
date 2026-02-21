# ==========================================
# Sequential TS simulation
# Posterior plotted AFTER incorporating reward at time t
# Exact Gaussian TS density used for plotting
# ==========================================

set.seed(100)

# -----------------------------
# SETTINGS
# -----------------------------
T_max <- 1000
sigma <- 0.5
sigma0 <- 1
lambda <- 1
b <- 5

theta_true <- c(0.5, 0.6)

eta <- 1/(2*sigma^2)

# Storage
arms <- c()
rewards <- c()

# Grid for visualization
grid_size <- 150
theta1 <- seq(-2, 2, length.out = grid_size)
theta2 <- seq(-2, 2, length.out = grid_size)

times_to_plot <- c(1,2,3,1000)

par(mfrow=c(2,2))

for(t in 1:T_max){
  
  # =====================================================
  # 1️⃣ Posterior from data up to time t-1
  # =====================================================
  if(t == 1){
    theta_hat_prev <- c(0,0)
    Sigma_prev <- diag(sigma0^2,2)
  } else {
    X_prev <- matrix(0, t-1, 2)
    for(i in 1:(t-1)){
      X_prev[i, arms[i]] <- 1
    }
    
    r_prev <- rewards
    
    V0_inv <- diag(1/sigma0^2,2)
    V_prev <- V0_inv + t(X_prev) %*% X_prev / sigma^2
    Sigma_prev <- solve(V_prev)
    theta_hat_prev <- Sigma_prev %*% (t(X_prev) %*% r_prev / sigma^2)
  }
  
  # =====================================================
  # 2️⃣ Thompson Sampling action
  # =====================================================
  theta_sample <- MASS::mvrnorm(1, theta_hat_prev, Sigma_prev)
  chosen_arm <- which.max(theta_sample)
  
  reward <- theta_true[chosen_arm] + rnorm(1,0,sigma)
  
  arms <- c(arms, chosen_arm)
  rewards <- c(rewards, reward)
  
  # =====================================================
  # 3️⃣ Build posterior INCLUDING time t
  # =====================================================
  X_full <- matrix(0, t, 2)
  for(i in 1:t){
    X_full[i, arms[i]] <- 1
  }
  
  r_full <- rewards
  
  V0_inv <- diag(1/sigma0^2,2)
  V_t <- V0_inv + t(X_full) %*% X_full / sigma^2
  Sigma_t <- solve(V_t)
  theta_hat_t <- Sigma_t %*% (t(X_full) %*% r_full / sigma^2)
  
  # =====================================================
  # 4️⃣ Plot at selected times
  # =====================================================
  if(t %in% times_to_plot){
    
    TS <- matrix(0, grid_size, grid_size)
    FG <- matrix(0, grid_size, grid_size)
    
    Sigma_inv_t <- solve(Sigma_t)
    
    for(i in 1:grid_size){
      for(j in 1:grid_size){
        
        th <- c(theta1[i], theta2[j])
        
        # ---- Exact TS Gaussian density
        diff <- th - theta_hat_t
        TS[j,i] <- exp(-0.5 * t(diff) %*% Sigma_inv_t %*% diff)
        
        # ---- FG target density
        ll <- -eta * sum((X_full %*% th - r_full)^2)
        prior <- -0.5 * sum(th^2) / sigma0^2
        bonus <- lambda * sum(pmin(b, X_full %*% th))
        
        FG[j,i] <- exp(ll + prior + bonus)
      }
    }
    
    TS <- TS / sum(TS)
    FG <- FG / sum(FG)
    
    contour(theta1, theta2, TS,
            drawlabels=FALSE,
            col="blue",
            lwd=2,
            xlab=expression(theta[1]),
            ylab=expression(theta[2]),
            main=paste0("T = ", t,
                        ", arm = ", arms[t],
                        ", rew = ", round(rewards[t],2)))
    
    contour(theta1, theta2, FG,
            drawlabels=FALSE,
            col="red",
            lwd=2,
            add=TRUE)
    
    legend("topright",
           legend=c("TS","FGTS"),
           col=c("blue","red"),
           lwd=2,
           cex=0.8)
    
    # Print posterior variances for sanity
    cat("T =", t, " Posterior variances:\n")
    print(diag(Sigma_t))
  }
}