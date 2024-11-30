rm(list=objects())

#' ******************************
#' *                            *
#' *   Condition d'expérience   *
#' *                            *
#' ******************************

# graine :
seed <- 10
set.seed(seed)

# taille totale de l'échantillon généré
n <- 1000

# proportions de chaque classe
p1 <- 1./5
p2 <- 7./15
p3 <- 1./3

# esperances
u1 <- -6.
u2 <- -1.
u3 <- 3.

# ecart type (identique sur les trois classes)
std1 <- 1.25
std2 <- 1.
std3 <- .5

# --- visualisation de la distribution des données et densités théoriques

# graphics.off()
default_plot <- function(){
  hist(X, breaks=100, xlab="X", main="Modeles de melange", freq=FALSE, col="whitesmoke", border="lightgrey")                 # données non labelisées

  # densités du mélange
  x <- seq(-5, 5, length=100)
  lines(x, lwd=5, dnorm(x, mean = u1, sd = std1)*p1, col='salmon', type='l')
  lines(x, lwd=5, dnorm(x, mean = u2, sd = std2)*p2, col='limegreen', type='l')
  lines(x, lwd=5, dnorm(x, mean = u3, sd = std3)*p3, col='lightskyblue', type='l')

  # densité du mélange
  lines(x, lwd=3,
        dnorm(x, mean = u1, sd = std1)*p1 +
        dnorm(x, mean = u2, sd = std2)*p2 +
        dnorm(x, mean = u3, sd = std3)*p3
    , col='black', type='l')
}

default_plot()

#' ******************************
#' *                            *
#' *      Exemple Rmixmod       *
#' *                            *
#' ******************************

# --- Génération d'un échantillon
d <- runif(rep(1,n))

X1 <- rnorm(sum(d <= p1), u1, std1)
X2 <- rnorm(sum((p1 < d) * (d <= (p1+p2))), u2, std2)
X3 <- rnorm(sum((p1+p2) < d), u3, std3)

length(X1); length(X2); length(X3)
length(X1) + length(X2) + length(X3)

X <- sample(c(X1, X2, X3)) # mélange de toutes les données

# --- Algo

library(Rmixmod)

strat <- mixmodStrategy(algo =  c("SEM", "EM"), initMethod = "random", nbTry = 10, epsilonInInit = 0.00001)
mod <- mixmodGaussianModel(family =  c("diagonal", "spherical"))
mixmod <- mixmodCluster(X, nbCluster=3,
                       criterion= c("BIC", "ICL", "NEC"),
                       strategy= strat,
                       models=mod)
summary(mixmod)

# --- Estimation

mu.est  <- mixmod@bestResult@parameters@mean[,1]
std.est <- sapply(1:length(mixmod@bestResult@parameters@variance), function(k) sqrt(mixmod@bestResult@parameters@variance[[k]][1,1]))
pi.est  <- mixmod@bestResult@parameters@proportions
classif <- mixmod@bestResult@partition

ord  <- sort(mixmod@bestResult@parameters@mean[,1], index.return=TRUE)$ix
mixmod@bestResult@parameters@mean[,1][ord]
sapply(1:length(mixmod@bestResult@parameters@variance), function(k) sqrt(mixmod@bestResult@parameters@variance[[k]][1,1]))[ord]
mixmod@bestResult@parameters@proportions[ord]

# --- Visualisation

default_plot()

lines(x, dnorm(x, mean=mu.est[1], sd=std.est[1])*pi.est[1], lwd=3, lty=3, col='dimgrey')
lines(x, dnorm(x, mean=mu.est[2], sd=std.est[2])*pi.est[2], lwd=3, lty=3, col='dimgrey')
lines(x, dnorm(x, mean=mu.est[3], sd=std.est[3])*pi.est[3], lwd=3, lty=3, col='dimgrey')
lines(x,
      dnorm(x, mean=mu.est[1], sd=std.est[1])*pi.est[1]+
      dnorm(x, mean=mu.est[2], sd=std.est[2])*pi.est[2]+
      dnorm(x, mean=mu.est[3], sd=std.est[3])*pi.est[3],
      lwd=3, col='black', lty=3)

c(likelihood=mixmod@bestResult@likelihood)
c(BIC=mixmod@bestResult@criterionValue[1],ICL=mixmod@bestResult@criterionValue[2])

#' ******************************
#' *                            *
#' *        test complet        *
#' *          Rmixmod           *
#' *                            *
#' ******************************

N <- 1000

res.likelihood <- rep(0,N)
res.BIC <- rep(0,N)
res.ICL <- rep(0,N)

res.pi <- matrix(rep(0,3*N),N,3)
res.mu <- matrix(rep(0,3*N),N,3)
res.std <- matrix(rep(0,3*N),N,3)

strat <- mixmodStrategy(algo =  c("SEM"), initMethod = "random", nbTry = 10, epsilonInInit = 0.00001)
mod <- mixmodGaussianModel(family =  "spherical")

for(i in 1:N){
  if(i%%10 == 0){print(i)}

  d <- runif(rep(1, n))

  X1 <- rnorm(sum(d <= p1), u1, std1)
  X2 <- rnorm(sum((p1 < d) * (d <= (p1+p2))), u2, std2)
  X3 <- rnorm(sum((p1+p2) < d), u3, std3)

  length(X1); length(X2); length(X3)
  length(X1) + length(X2) + length(X3)

  X <- sample(c(X1, X2, X3))

  mixmod <- mixmodCluster(X, nbCluster=3,
                       criterion= c("BIC", "ICL"),
                       strategy= strat,
                       models=mod)

  ord  <- sort(mixmod@bestResult@parameters@mean[,1], index.return=TRUE)$ix
  res.mu[i,]  <- mixmod@bestResult@parameters@mean[,1][ord]
  res.std[i,] <- sapply(1:length(mixmod@bestResult@parameters@variance), function(k) sqrt(mixmod@bestResult@parameters@variance[[k]][1,1]))[ord]
  res.pi[i,]  <- mixmod@bestResult@parameters@proportions[ord]

  res.likelihood[i] <- mixmod@bestResult@likelihood
  res.BIC[i] <- mixmod@bestResult@criterionValue[1]
  res.ICL[i] <- mixmod@bestResult@criterionValue[2]

}

mean(res.likelihood)
mean(res.BIC)
mean(res.ICL)

apply(res.pi, 2, mean)
apply(res.mu, 2, mean)
apply(res.std, 2, mean)

# RMSE
sum((res.mu - matrix(rep(c(u1,u2,u3),N),N,3, byrow=TRUE))^2)
sum((res.std - matrix(rep(c(std1,std2,std3),N),N,3, byrow=TRUE))^2)
sum((res.pi - matrix(rep(c(p1,p2,p3),N),N,3, byrow=TRUE))^2)



#' ******************************
#' *                            *
#' *         blockmodels        *
#' *                            *
#' ******************************

library(blockmodels)

### EXEMPLE

# npc <- 20 # nodes per class
# Q <- 4 # classes
# n <- npc * Q # nodes
# Z<-diag(Q)%x%matrix(1,npc,1)
# Mu<-20*matrix(runif(Q*Q),Q,Q)
# M<-matrix(rnorm(n*n,sd=10),n,n)+Z%*%Mu%*%t(Z) ## adjacency matrix
#
# ## estimation
# my_model <- BM_gaussian("SBM",M)
# my_model$estimate()
# which.max(my_model$ICL)
# my_model$model_parameters





