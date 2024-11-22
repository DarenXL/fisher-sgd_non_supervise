import collections
import functools

import parametrization_cookbook.jax as pc
import jax
import jax.numpy as jnp

Obs = collections.namedtuple("Obs", ("Y", "Yzd", "umYzd", "YzdT", "umYzdT"))


def make_obs(Y):
    import numpy as np  # NumPy only here

    umI = 1 - jnp.eye(Y.shape[0])
    Yzd = Y * umI
    umYzd = (1 - Y) * umI
    return Obs(
        Y=Y,
        Yzd=Yzd,
        umYzd=umYzd,
        YzdT=jnp.array(np.array(Yzd.T, order="C")),
        umYzdT=jnp.array(np.array(umYzd.T, order="C")),
    )


class SBMModel:
    def __init__(self, Q: int):
        self._Q = Q
        self._parametrization = pc.NamedTuple(
            alpha=pc.VectorSimplex(dim=Q - 1),
            pi=pc.RealBounded01(shape=(Q, Q)),
        )

        # gradients
        self.jac_loglikelihood_obs_by_couple = jax.jit(
            jax.jacfwd(self.loglikelihood_obs_by_couple)
        )
        self.jac_loglikelihood_latent_by_node = jax.jit(
            jax.jacfwd(self.loglikelihood_latent_by_node)
        )
        self.hessian_loglikelihood = jax.jit(jax.jacfwd(jax.jacrev(self.loglikelihood)))

    @property
    def parametrization(self):
        return self._parametrization

    @functools.partial(jax.jit, static_argnums=0)
    def loglikelihood(self, theta, Z, obs):
        n = obs.Y.shape[0]
        assert Z.shape == (n, self._Q)
        p = self.parametrization.reals1d_to_params(theta)
        return (Z @ jnp.log(p.alpha)).sum() + (
            (Z.T @ obs.Yzd @ Z) * jnp.log(p.pi)
            + (Z.T @ obs.umYzd @ Z) * jnp.log1p(-p.pi)
        ).sum()

    @functools.partial(jax.jit, static_argnums=0)
    def loglikelihood_obs_by_couple(self, theta, Z, obs):
        n = obs.Y.shape[0]
        assert Z.shape == (n, self._Q)
        p = self.parametrization.reals1d_to_params(theta)
        return obs.Yzd * (Z @ jnp.log(p.pi) @ Z.T) + obs.umYzd * (
            Z @ jnp.log1p(-p.pi) @ Z.T
        )

    @functools.partial(jax.jit, static_argnums=0)
    def loglikelihood_latent_by_node(self, theta, Z):
        assert Z.shape[1] == self._Q
        p = self.parametrization.reals1d_to_params(theta)
        return Z @ jnp.log(p.alpha)

    @functools.partial(jax.jit, static_argnums=0)
    def gibbs_step(
        self,
        theta,
        Z,
        obs,
        key,
    ):
        n, Q = Z.shape
        p = self.parametrization.reals1d_to_params(theta)

        def _loop(carry, i):
            key, Z = carry
            key, localkey = jax.random.split(key)
            lp = (
                jnp.log(p.alpha)
                + obs.Yzd[i] @ Z @ jnp.log(p.pi.T)
                + obs.umYzd[i] @ Z @ jnp.log1p(-p.pi.T)
                + obs.YzdT[i] @ Z @ jnp.log(p.pi)
                + obs.umYzdT[i] @ Z @ jnp.log1p(-p.pi)
            )
            unormalized_p = jnp.exp(lp - lp.max())
            prob = unormalized_p / unormalized_p.sum()

            Z = Z.at[i].set(
                jnp.zeros(Z.shape[1])
                .at[jax.random.choice(key=localkey, a=jnp.arange(Z.shape[1]), p=prob)]
                .set(1)
            )
            return (key, Z), None

        key, keyperm, keyloop = jax.random.split(key, 3)
        (_, Z), _ = jax.lax.scan(
            _loop,
            (keyloop, Z),
            jax.random.permutation(key=keyperm, x=jnp.arange(n)),
        )

        return key, Z
