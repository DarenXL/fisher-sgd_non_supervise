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
p1 = .6
p2 = .1
p3 = .3

# esperances
u1 = 2
u2 = 0
u3 = -2

# ecart type (identique sur les trois classes)
std1 = .5
std2 = .5
std3 = .5

d = runif(rep(1,n))

X1 = rnorm(sum(d <= p1), u1, std1)
X2 = rnorm(sum((p1 < d) * (d <= (p1+p2))), u2, std2)
X3 = rnorm(sum((p1+p2) < d), u3, std3)

length(X1); length(X2); length(X3)
length(X1) + length(X2) + length(X3)

X = sample(c(X1, X2, X3)) # mélange de toutes les données

# --- visualisation de la distribution des données et densités théoriques

# graphics.off()
default_plot = function(){
  hist(X, breaks=100, xlab="X", main="Modeles de melange", freq=FALSE, col="whitesmoke", border="lightgrey")                 # données non labelisées

  # densités du mélange
  x = seq(-5, 5, length=100)
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
#' *         Fisher-SGD         *
#' *                            *
#' ******************************



#' ******************************
#' *                            *
#' *           Rmixmod          *
#' *                            *
#' ******************************

library(Rmixmod)

strat = mixmodStrategy(algo =  c("SEM", "EM"), initMethod = "random", nbTry = 10, epsilonInInit = 0.00001)
mod = mixmodGaussianModel(family =  c("diagonal", "spherical"))
mixmod = mixmodCluster(X, nbCluster=3,
                       criterion= c("BIC", "ICL", "NEC"),
                       strategy= strat,
                       models=mod)
summary(mixmod)

# --- Estimation

mu.est  = mixmod@bestResult@parameters@mean[,1]
std.est = sapply(1:length(mixmod@bestResult@parameters@variance), function(k) sqrt(mixmod@bestResult@parameters@variance[[k]][1,1]))
pi.est  = mixmod@bestResult@parameters@proportions
classif = mixmod@bestResult@partition

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
#' *         blockmodels        *
#' *                            *
#' ******************************

library(blockmodels)


