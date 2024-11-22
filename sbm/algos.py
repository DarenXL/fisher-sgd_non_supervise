import sys
import itertools
import functools
import math
import jax
import jax.numpy as jnp

from model import SBMModel

import jax

jax.config.update("jax_platform_name", "cpu")


class O3filter:
    m1: jnp.array
    m2: jnp.array
    m3: jnp.array
    cst: float
    mone: float

    def __init__(self, size, cst):
        self.m1 = jnp.zeros(size)
        self.m2 = jnp.zeros(size)
        self.m3 = jnp.zeros(size)
        self.cst = cst
        self.mone = 0.0

    def update(self, val):
        self.mone += self.cst * (1 - self.mone)
        self.m1 = self.m1 + self.cst * (val - self.m1)
        self.m2 = self.m2 + self.cst * (self.m1 / self.mone - self.m2)
        self.m3 = self.m3 + self.cst * (self.m2 / self.mone - self.m3)

    @property
    def unbiaised_m3(self):
        return self.m3 / self.mone


PREHEATING = 1000
HEATING_CST = 1 / 1000

from parametrization_cookbook.functions.jax import reals_to_simplex


@functools.partial(jax.jit, static_argnums=0)
def one_iter(model, k, lr_step, theta, Z, obs, prng_key, Delta_lat, Delta_obs):
    prng_key, subkey = jax.random.split(prng_key)
    prng_key, Z = model.gibbs_step(theta, Z, obs, subkey)

    jac_latent = model.jac_loglikelihood_latent_by_node(theta, Z)
    jac_obs = model.jac_loglikelihood_obs_by_couple(theta, Z, obs)
    Delta_lat += lr_step * (jac_latent - Delta_lat)
    Delta_obs += lr_step * (jac_obs - Delta_obs)
    precond_matrix = jnp.einsum(
        "iq,il->ql", Delta_lat, Delta_lat, optimize="optimal"
    ) + jnp.einsum("ijq,ijl->ql", Delta_obs, Delta_obs, optimize="optimal")
    precond_matrix = jax.lax.cond(
        k < PREHEATING,
        lambda precond_matrix: (1 - lr_step)
        * jnp.maximum(jnp.diag(precond_matrix).sum(), 1.0)
        * jnp.eye(model.parametrization.size)
        + lr_step * precond_matrix,
        lambda precond_matrix: precond_matrix,
        precond_matrix,
    )

    gradient = jac_latent.sum(axis=0) + jac_obs.sum(axis=(0, 1))
    direction = jnp.linalg.solve(precond_matrix, gradient)
    theta += lr_step * direction
    return (theta, Z, prng_key, Delta_lat, Delta_obs, precond_matrix, gradient)


class EmptyError(Exception):
    pass


class NanError(Exception):
    pass


import collections

IterRes = collections.namedtuple("IterRes", ("Z", "nhessian", "theta", "loglikelihood"))
EstimRes = collections.namedtuple("EstimRes", ("Z", "FIM", "theta", "loglikelihood"))


def estim_with_model(model, Q, obs, prng_key, K1=10000, K2=20000, yield_all=False):
    n = obs.Y.shape[0]
    grad_filter = O3filter(model.parametrization.size, HEATING_CST)

    prng_key, subkey1, subkey2 = jax.random.split(prng_key, 3)
    Z = (
        jnp.zeros((n, Q))
        .at[
            jnp.arange(n),
            jax.random.randint(key=subkey1, minval=0, maxval=Q, shape=(n,)),
        ]
        .set(1)
    )
    theta = model.parametrization.params_to_reals1d(
        alpha=reals_to_simplex(0.1 * jax.random.normal(subkey2, shape=(Q - 1,))),
        pi=obs.Yzd.sum() / (n * (n - 1)) * jnp.ones((Q, Q)),
    )

    Delta_obs = jnp.zeros((n, n, model.parametrization.size))
    Delta_lat = jnp.zeros((n, model.parametrization.size))

    heating = True
    end_heating = 0

    norm2_mean_grad = -jnp.inf
    for k in itertools.count():
        if k < PREHEATING:
            lr_step = math.exp((1 - k / PREHEATING) * math.log(1e-4))
        elif heating:
            lr_step = 1.0
        else:
            lr_step = (k - end_heating) ** (-2 / 3)

        (theta, Z, prng_key, Delta_lat, Delta_obs, precond_matrix, gradient) = one_iter(
            model, k, lr_step, theta, Z, obs, prng_key, Delta_lat, Delta_obs
        )

        if Z.sum(axis=0).min() < 0.5:
            raise EmptyError
        if jnp.isnan(theta).any():
            raise NanError
        if heating:
            grad_filter.update(gradient)
            last_norm2_mean_grad, norm2_mean_grad = (
                norm2_mean_grad,
                (grad_filter.unbiaised_m3**2).sum(),
            )
            if k > PREHEATING and norm2_mean_grad < last_norm2_mean_grad:
                heating = False
                end_heating = k
            if yield_all:
                yield IterRes(
                    Z=Z,
                    nhessian=-model.hessian_loglikelihood(theta, Z, obs),
                    theta=theta,
                    loglikelihood=model.loglikelihood(theta, Z, obs),
                )
        else:
            if k - end_heating > K1 or yield_all:
                yield IterRes(
                    Z=Z,
                    nhessian=-model.hessian_loglikelihood(theta, Z, obs),
                    theta=theta,
                    loglikelihood=model.loglikelihood(theta, Z, obs),
                )
            if k - end_heating == K2:
                return


import operator


def estim(Q, obs, prng_key, K1=10000, K2=20000, idmsg="", retries=3, debug=False):
    model = SBMModel(Q)
    allres = []
    while len(allres) < retries:
        prng_key, local_key = jax.random.split(prng_key)
        try:
            Zsum = jnp.zeros((obs.Y.shape[0], Q))
            thetasum = jnp.zeros(model.parametrization.size)
            FIMsum = jnp.zeros((model.parametrization.size, model.parametrization.size))
            loglikelihood = 0.0
            count = 0
            for res in estim_with_model(model, Q, obs, local_key, K1, K2):
                Zsum += res.Z
                thetasum += res.theta
                FIMsum += res.nhessian
                loglikelihood += res.loglikelihood
                count += 1
        except EmptyError:
            if debug:
                print(
                    f"{idmsg}EmptyError, len(allres)={len(allres)}, prng_key={local_key}",
                    file=sys.stderr,
                )
            continue
        except NanError:
            if debug:
                print(
                    f"{idmsg}NanError, len(allres)={len(allres)}, prng_key={local_key}",
                    file=sys.stderr,
                )
            continue
        allres.append(
            EstimRes(
                Z=Zsum / count,
                FIM=FIMsum / count,
                theta=thetasum / count,
                loglikelihood=loglikelihood / count,
            )
        )
    return max(allres, key=lambda x: x.loglikelihood)


def labelswitch(model, estim_res, Zsim, thetasim):
    Q = Zsim.shape[1]
    perm = max(
        itertools.permutations(range(Q)),
        key=lambda p: jnp.diag(estim_res.Z[:, p].T @ Zsim).sum(),
    )
    revperm = tuple(sorted(range(Q), key=lambda k: perm[k]))
    params = model.parametrization.reals1d_to_params(thetasim)
    return (
        perm,
        Zsim[:, perm],
        model.parametrization.params_to_reals1d(
            alpha=params.alpha[(revperm,)], pi=params.pi[revperm, :][:, revperm]
        ),
    )


@functools.partial(jax.jit, static_argnums=0)
def build_params_est_and_std(mod, estim_res, perm):
    original_params = lambda theta: (
        lambda p: jnp.concatenate((p.alpha[(perm,)], p.pi[perm, :][:, perm].ravel()))
    )(mod.parametrization.reals1d_to_params(theta))
    med = original_params(estim_res.theta)
    covar = jnp.linalg.inv(estim_res.FIM)
    jac = jax.jacfwd(original_params)(estim_res.theta)
    radius = jnp.sqrt(jnp.array([jac[k] @ covar @ jac[k] for k in range(med.size)]))
    return med, radius
