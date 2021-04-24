# Copied from github of replication code

library(ICEbox)
library(randomForest)
library(gam)
library(gbm)
library(nnet)
library(missForest)
library(MASS) #has Boston Housing data, Pima

additivity_ex_sim = function(n,seednum=NULL){
  if(!is.null(seednum)){
    set.seed(seednum)
  }
  p = 2
  X = as.data.frame(matrix(runif(n * p, -1, 1), ncol = p))	
  colnames(X) = paste("x_", 1 : p, sep = "")
  bbeta = c(1,1)
  
  y = bbeta[1] * X[,1]^2  + bbeta[2] * X[,2]
  y = y + rnorm(n)
  Xy = as.data.frame(cbind(X, y))
  return(list(Xy=Xy,X=X,y=y))
}

# generate data:
additivity_ex_data = additivity_ex_sim(1000, seednum = 50)
Xy = additivity_ex_data$Xy
X  = additivity_ex_data$X
y  = additivity_ex_data$y

# build gam with possible interactions:
gam_mod = gam(y~s(x_1)+s(x_2)+s(x_1*x_2),data=Xy)   

# build ICE and d-ICE:
gam.ice = ice(gam_mod, X, predictor = 1, frac_to_build = 1) 
gam.dice = dice(gam.ice)

# plot the ICE plot with pdp, and d-ICE with dpdp
pdf("../../data/results/01-replication/fig 7a.pdf", width = 7, height = 10) 
plot(gam.ice, x_quantile = F, plot_pdp = T, frac_to_plot = 0.25)  
dev.off()
pdf("../../data/results/01-replication/fig 7b.pdf", width = 7, height = 10) 
plot(gam.dice, x_quantile = F, plot_dpdp = T, frac_to_plot = 0.25) 
dev.off()