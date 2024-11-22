import matplotlib.pyplot as plt
import parametrization_cookbook.jax as pc
import numpy as np
import pickle
import model
import jax.numpy as jnp

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

with open("example_res.npy", 'rb') as f:
     ex_res = pickle.load(f)

evol_p = np.zeros(shape=(ex_res.evol_theta.shape[0],p0_vec.size))
for i in range(evol_p.shape[0]):
    evol_p[i,:] = vect_params(model.parametrization.reals1d_to_params(ex_res.evol_theta[i,:]))

evol_fim = ex_res.evol_fim[0:5000,:,:]
factor = ex_res.factor[0:5000]
evol_p = evol_p[0:5000,:]

subtitles = [r'$\beta_1$',r'$\beta_2$',r'$\alpha$',r'$\Gamma_{11}$',r'$\Gamma_{12}$',r'$\Gamma_{22}$',r'$\sigma^2$']
subtitles_fim = [r'$\hat{I}(\theta_1)$',r'$\hat{I}(\theta_2)$',r'$\hat{I}(\theta_3)$',r'$\hat{I}(\theta_4)$',r'$\hat{I}(\theta_5)$',r'$\hat{I}(\theta_6)$',r'$\hat{I}(\theta_7)$']

factor1 = jnp.where(ex_res.factor == 1)
k1 = factor1[0][0]
k2 = factor1[0][-1]

fig, ax = plt.subplots(nrows=4, ncols=2, sharex=True, sharey=False, figsize=(6,8))
ax = ax.ravel()
for i in range(evol_p.shape[1]):
    ax[i].axhline(p0_vec[i],color='y')
    ax[i].plot(evol_p[:,i])
    if i != 4:
        ax[i].set_yscale('log')
    ax[i].axvline(k1,color='red')
    ax[i].axvline(k2,color='green')
    ax[i].set_title(subtitles[i])
    ax[7].axis('off')

fig.show()

fig.savefig('plots/evol_log_p_5000.pdf',dpi=300)


fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=False, figsize=(10,8))
ax = ax.ravel()
for i in range(evol_p.shape[1]):
    ax[i].plot(evol_fim[:,i,i])
    ax[i].set_ylabel(subtitles_fim[i])
    ax[7].plot(factor)
    ax[7].set_ylabel(r'$\gamma_k$')
    ax[7].set_yscale('log')
    ax[8].axis("off")
    ax[5].set_xlabel("iterations")
    ax[6].set_xlabel("iterations")
    ax[7].set_xlabel("iterations")

fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.show()

fig.savefig('plots/evol_fim_with_factor_5000.pdf',dpi=300)
