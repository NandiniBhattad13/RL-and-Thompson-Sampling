set.seed(123)
library(MASS)

############################
# Parameters
############################
T <- 150
sigma2 <- 0.5
prior_var <- 5
beta_true <- c(1.5, -1.0)

############################
# Prior
############################
m <- c(0, 0)
V <- diag(prior_var, 2)

############################
# Storage
############################
m_history <- matrix(NA, T, 2)
V_history <- array(NA, dim = c(2,2,T))

############################
# Thompson Sampling Loop
############################
for(t in 1:T){
  
  x_t <- rnorm(1)
  
  beta_sample <- mvrnorm(1, m, V)
  
  mu_hat <- c(x_t * beta_sample[1],
              x_t * beta_sample[2])
  
  a_t <- which.max(mu_hat)
  
  r_t <- rnorm(1,
               mean = x_t * beta_true[a_t],
               sd = sqrt(sigma2))
  
  phi <- if(a_t == 1) c(x_t, 0) else c(0, x_t)
  
  V_inv <- solve(V)
  V_new <- solve(V_inv + (1/sigma2) * tcrossprod(phi))
  m_new <- V_new %*% (V_inv %*% m + (1/sigma2) * phi * r_t)
  
  V <- V_new
  m <- as.vector(m_new)
  
  m_history[t,] <- m
  V_history[,,t] <- V
}

############################
# Prepare grid
############################
x_seq <- seq(-3, 3, length = 120)
y_seq <- seq(-3, 3, length = 120)
grid  <- expand.grid(x_seq, y_seq)

############################
# Times to display
############################
snapshots <- c(1, 10, 100, 150)

############################
# Plot
############################
par(mfrow = c(2,2), mar=c(4,4,3,1))

for(t in snapshots){
  
  z <- matrix(
    dmvnorm(grid,
            mean = m_history[t,],
            sigma = V_history[,,t]),
    nrow = length(x_seq)
  )
  
  contour(x_seq, y_seq, z,
          levels = pretty(z, 6),
          drawlabels = FALSE,
          xlab = expression(beta[1]),
          ylab = expression(beta[2]),
          main = paste("t =", t),
          col = "black",
          lwd = 2)
  
  # True beta
  points(beta_true[1], beta_true[2],
         col="darkgreen", pch=19, cex=1.5)
  
  # Posterior mean
  points(m_history[t,1],
         m_history[t,2],
         col="red", pch=19, cex=1.2)
  
  legend("topright",
         legend=c("True β", "Posterior Mean"),
         col=c("darkgreen","red"),
         pch=19,
         bty="n",
         cex=0.8)
}