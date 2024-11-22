import algos
import model
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

def vect_params(p):
    il1 = jnp.tril_indices(2) # indices for lower triangular of a 2x2 matrix
    return jnp.concatenate((jnp.array([p.asymptotic,p.inflexion,p.tau]), p.cov_latent[il1], jnp.array([p.var_residual])))
    

# Import and clean file to use only on the white-browed coucals    
y_pd = pd.read_csv('growth_coucal.csv')
df = pd.DataFrame(y_pd)
df = df[["nestling_ID2_cat","species","weight","age"]]
df = df.dropna()
df_wbc = df.loc[df['species'] == "WBC"]
df_wbc = df_wbc.drop(columns='species')
df_wbc = df_wbc.drop_duplicates()
y_wbc = df_wbc.pivot_table(index='nestling_ID2_cat',columns='age',values='weight',aggfunc='mean')


# Plot the data
fig, ax = plt.subplots(figsize=(6,5))
df_wbc.groupby('nestling_ID2_cat').plot(x='age',y='weight',ax=ax,legend=False)
ax.set_xlabel("days (from hatching)")
ax.set_ylabel("body weight (g)")
fig.savefig('real_data_coucal_wbc.pdf',dpi=300)


# Get data dimension
n = y_wbc.shape[0]
Ji = y_wbc.shape[1] - np.isnan(y_wbc).sum(axis=1)
pd.Series.min(Ji)
pd.Series.max(Ji)
t = jnp.tile(df_wbc['age'].unique(),(n,1))

key_estim = jax.random.PRNGKey(0)

# Set initial value for the parameter
theta0 = model.parametrization.params_to_reals1d(
    asymptotic=120,
    inflexion=10,
    tau=4,
    cov_latent=jnp.diag(jnp.array([100, 10])),
    var_residual=10,
)
p0 = model.parametrization.reals1d_to_params(theta0)
p0_vec = vect_params(p0)

res = algos.estim(y_wbc.to_numpy(), t, stop_crit=1e-6, N_smooth=10000, prng_key=key_estim, pre_heating=2000)

# Plots
evol_p = np.zeros(shape=(res.evol_theta.shape[0],p0_vec.size))
for i in range(evol_p.shape[0]):
    evol_p[i,:] = vect_params(model.parametrization.reals1d_to_params(res.evol_theta[i,:]))


# Estimate at the end of the algorithm
evol_p[evol_p.shape[0]-1,:]


subtitles = [r'$\beta_1$',r'$\beta_2$',r'$\alpha$',r'$\Gamma_{11}$',r'$\Gamma_{12}$',r'$\Gamma_{22}$',r'$\sigma^2$']

fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=False, figsize=(10,8))
ax = ax.ravel()
for i in range(evol_p.shape[1]):
    #if i < 3:
    ax[i].plot(evol_p[:,i])
    ax[i].set_title(subtitles[i])
    #else:
        #ax[i+1].plot(evol_p[:,i])
        #ax[i+1].set_title(subtitles[i])
    ax[4].set_xlabel("iterations")
    ax[5].set_xlabel("iterations")
    ax[6].set_xlabel("iterations")       
    ax[7].set_xlabel("iterations")           
    ax[7].axis('off')
    ax[8].axis('off')

fig.tight_layout()
fig.show()
fig.savefig('evol_theta_real_data_coucal_wbc.pdf',dpi=300)


