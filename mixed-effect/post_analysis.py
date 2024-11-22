import scipy.stats
import model
import glob
import jax.numpy as jnp
import jax

import parametrization_cookbook.jax as pc

with open('theta_all.npy', 'rb') as f:
    theta = jnp.load(f)
    
with open('fim_all.npy', 'rb') as f:
    fim = jnp.load(f)
     

def vect_params(p):
    il1 = jnp.tril_indices(2) # indices for lower triangular of a 2x2 matrix
    return jnp.concatenate((jnp.array([p.asymptotic,p.inflexion,p.tau]), p.cov_latent[il1], jnp.array([p.var_residual])))
    

theta0 = model.parametrization.params_to_reals1d(
    asymptotic=200,
    inflexion=500,
    tau=150,
    cov_latent=jnp.diag(jnp.array([40, 100])),
    var_residual=100,
)
p0 = model.parametrization.reals1d_to_params(theta0)
p0_vec = vect_params(p0)
        
n = 1000
nrep = 1000

ci_lower = jnp.zeros(shape=(nrep,theta0.size))
ci_upper = jnp.zeros(shape=(nrep,theta0.size))
quadform = jnp.zeros(shape=(nrep,))


p_all = jnp.empty_like(theta)
for i in range(nrep):
   fim_inv = jnp.linalg.inv(n*fim[i,:,:])
   
   p = model.parametrization.reals1d_to_params(theta[i,:])
   p_vec = vect_params(p) 
   p_all = p_all.at[i,:].set(p_vec)
     
   jac_theta = jax.jacfwd(lambda theta: model.parametrization.reals1d_to_params(theta))(theta[i,:])
   jac_theta_shaped = jnp.concatenate((jac_theta.asymptotic[None,:], jac_theta.inflexion[None,:], jac_theta.tau[None,:], jac_theta.cov_latent[0,0,:][None,:], jac_theta.cov_latent[0,1,:][None,:], jac_theta.cov_latent[1,1,:][None,:], jac_theta.var_residual[None,:]))
    
   fim_inv_p = jac_theta_shaped @ fim_inv @ jac_theta_shaped.T
   quadform = quadform.at[i].set((p_vec - p0_vec) @ jnp.linalg.inv(fim_inv_p) @ (p_vec - p0_vec))
        
   ci_lower = ci_lower.at[i,:].set(p_vec - scipy.stats.norm.ppf(0.975) * jnp.sqrt(jnp.diag(fim_inv_p)))
   ci_upper = ci_upper.at[i,:].set(p_vec + scipy.stats.norm.ppf(0.975) * jnp.sqrt(jnp.diag(fim_inv_p)))


# RMSE
mse = jnp.var(p_all,axis=0) + jnp.mean(p_all-p0_vec,axis=0)**2
rmse = jnp.sqrt(mse)

mse_global = jnp.mean(jnp.sum((p_all-p0_vec)**2,axis=1))
rmse_global = jnp.sqrt(mse_global)


# Empirical coverage
ci_global = jnp.mean(quadform < scipy.stats.chi2.ppf(0.95,df=theta0.size))
se_ci_global = scipy.stats.norm.ppf(0.975)*jnp.sqrt(ci_global*(1-ci_global)/n)

ci_indiv = jnp.mean((ci_lower <= p0_vec) & (p0_vec <= ci_upper),axis=0)
se_ci_indiv = scipy.stats.norm.ppf(0.975)*jnp.sqrt(ci_indiv*(1-ci_indiv)/n)
