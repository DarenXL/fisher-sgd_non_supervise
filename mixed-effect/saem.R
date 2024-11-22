library(saemix)
library(dplyr)
library(tidyr)

growthOrange3eff<-function(psi,id,xidep) {
  x<-xidep[,1]
  psi1<-psi[id,1]
  psi2<-psi[id,2]
  psi3<-psi[id,3]
  f <- psi1/(1+exp(-(x-psi2)/psi3))
  return(f)
}

t <- c(100,173.68422,247.36842,321.05264, 394.73685,
       468.42108,  542.1053 ,  615.7895 ,  689.4737 ,  763.15796,
       836.84216,  910.52637,  984.2106 , 1057.8948 , 1131.579  ,
       1205.2632 , 1278.9474 , 1352.6317 , 1426.3159 , 1500)

saemix.model <- saemixModel(model=growthOrange3eff,description="Orange trees",
                            psi0=matrix(c(300,400,100),ncol=3,byrow=TRUE,
                                        dimnames=list(NULL,c("psi1","psi2","psi3"))),transform.par=c(0,0,0),fixed.estim=c(1,1,1),
                            covariance.model=matrix(c(1,1,0,
                                                      1,1,0,
                                                      0,0,0),ncol=3,byrow=TRUE),
                            omega.init=matrix(c(60,10,0,
                                                10,80,0,
                                                0,0,10),ncol=3,byrow=TRUE),error.model="proportionnal")
saemix.options <- saemixControl(nbiter.saemix=c(500,500),print=FALSE)

theta0 <- c(200,500,150,40,0,100,100)

il <- lower.tri(matrix(1,nr=2,nc=2),diag = TRUE)

lf <- list.files('data/',pattern='y', full.names = TRUE)
theta <- matrix(0,nr=length(lf),nc=7)
fim <- list()
quadform <- rep(0,length(lf))
ci_lower <- matrix(0,nr=length(lf),nc=7)
ci_upper <- matrix(0,nr=length(lf),nc=7)

for (i in 1:length(lf)){
  print(i)
  data <- read.table(lf[i],header=TRUE,sep=",")[,-1]
  n <- nrow(data)
  data_long <- data %>% dplyr::mutate(id=1:n) %>% pivot_longer(cols = !id, names_to = "Time", names_prefix = "x_", 
                                                                  names_transform = list(Time=as.numeric), values_to = "Circonference")
  data_long$Time <- rep(t,n)
  head(data_long)

  saemix.data <- saemixData(name.data=data_long,name.group=c("id"),
                          name.predictors=c("Time"),name.response=c("Circonference"),
                          units=list(x="days",y="cm"))

  orange.saemix <- saemix(saemix.model,saemix.data,saemix.options)
  theta[i,] <- c(orange.saemix@results@betas,orange.saemix@results@omega[1:2,1:2][il],orange.saemix@results@respar[1]^2)
  fim[[i]] <- orange.saemix@results@fim
  fim[[i]][7,] <- fim[[i]][7,]/(2*sqrt(theta[i,7]))
  fim[[i]][,7] <- fim[[i]][,7]/(2*sqrt(theta[i,7]))
  quadform[i] <- t(theta[i,] - theta0) %*% fim[[i]] %*% (theta[i,] - theta0)
  ci_lower[i,] <- theta[i,] - qnorm(0.975) * sqrt(diag(solve(fim[[i]])))
  ci_upper[i,] <- theta[i,] + qnorm(0.975) * sqrt(diag(solve(fim[[i]])))
}

saveRDS(theta,"outputs/theta.rds")
saveRDS(fim,"outputs/fim.rds")
saveRDS(quadform,"outputs/quadform.rds")
saveRDS(ci_lower,"outputs/ci_lower.rds")
saveRDS(ci_upper,"outputs/ci_upper.rds")

# RMSE
theta0mat <- matrix(theta0, nr=1000, nc=7, byrow = T)
mse <- apply((theta - theta0mat),2,mean)^2 + apply(theta,2,var)
rmse <- sqrt(mse)

mse_global <- mean(sapply(1:1000,FUN=function(i){sum((theta[i,]-theta0)^2)}))
rmse_global <- sqrt(mse_global)

# Empirical coverage
ci_global <- mean(quadform < qchisq(0.95,7)) # 0.95
se_ci_global <- qnorm(0.975)*sqrt(ci_global*(1-ci_global)/n)
in_ci <- t(sapply(1:nrow(ci_lower),FUN=function(i){return(ci_lower[i,] < theta0 & theta0 < ci_upper[i,])}))
ci_indiv <- apply(in_ci,2,mean)
se_ci_indiv <- qnorm(0.975)*sqrt(ci_indiv*(1-ci_indiv)/n)


