import algos
import model
import jax.numpy as jnp
import jax
import pickle

import pandas as pd

from tqdm import tqdm
from collections import namedtuple

def sample_and_estim(theta0, n, prng_key):
    key_simu, key_estim = jax.random.split(prng_key)
    z, y, t = model.simu_data(theta0, n, prng_key=key_simu)
    y_pd = pd.DataFrame(y,columns=["x_" + str(i) for i in range(1,21)])
    y_pd.to_csv('data/y'+str(key_simu)+'.txt')
    res = algos.estim(y, t, stop_crit=1e-6, N_smooth=5000, prng_key=key_estim, pre_heating=2000)
    return res


theta0 = model.parametrization.params_to_reals1d(
    asymptotic=200,
    inflexion=500,
    tau=150,
    cov_latent=jnp.diag(jnp.array([40, 100])),
    var_residual=100,
)

Nsimus = 400
n = 1000
keyy = 0
many_res = list(
    tqdm(
        (
            sample_and_estim(theta0, n, key)
            for key in jax.random.split(jax.random.PRNGKey(keyy), Nsimus)
        ),
        total=Nsimus,
        smoothing=0,
    )
)

with open('outputs/example_res.npy', 'wb') as f:
    pickle.dump(many_res[0],f)


theta = jnp.array([t for t in ((x.theta) for x in many_res)])
fim = jnp.array([f for f in (x.fisher_info_mat for x in many_res)])

theta_no_nan = []
for t in ((x.theta) for x in many_res):
    if not(sum(jnp.isnan(t))) :
        theta_no_nan += [t]
theta_no_nan = jnp.array(theta_no_nan)

fim_no_nan = []
for f in ((x.fisher_info_mat) for x in many_res):
    if not(jnp.sum(jnp.isnan(f))) :
        fim_no_nan += [f]
fim_no_nan = jnp.array(fim_no_nan)
    
with open('outputs/theta_all_%s.npy' % keyy, 'wb') as f:
    jnp.save(f, theta)

with open('outputs/theta_no_nan_all_%s.npy' % keyy, 'wb') as f:
    jnp.save(f, theta_no_nan)
    
with open('outputs/fim_all_%s.npy' % keyy, 'wb') as f:
    jnp.save(f, fim)

with open('outputs/fim_no_nan_all_%s.npy' % keyy, 'wb') as f:
    jnp.save(f, fim_no_nan)
