import jax
import jax.numpy as jnp
import algos
import model


def get_theta0():
    Q = 4
    mod = model.SBMModel(Q)
    alpha0 = jnp.ones(4) / 4
    pi0 = (jnp.array([[1, 1, 0, 1], [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]]) + 1) / 3
    theta0 = mod.parametrization.params_to_reals1d(alpha=alpha0, pi=pi0)
    return theta0, (Q, alpha0, pi0, mod)


def gen_one(key, n):
    theta0, (Q, alpha0, pi0, mod) = get_theta0()
    key1, key2 = jax.random.split(key)
    Z = (
        jnp.zeros((n, Q))
        .at[jnp.arange(n), jax.random.randint(key1, minval=0, maxval=Q, shape=(n,))]
        .set(1)
    )
    Y = (jax.random.uniform(key2, shape=(n, n)) < Z @ pi0 @ Z.T) * 1.0
    obs = model.make_obs(Y)
    return Z, obs


def do_one_estim(args):
    k, n, retries = args
    key_sim, key_estim = jax.random.split(jax.random.PRNGKey(k))
    Z, obs = gen_one(key_sim, n)
    return Z, algos.estim(
        Z.shape[1], obs, key_estim, idmsg=f"key={k}, ", retries=retries
    )
