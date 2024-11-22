import itertools
import math

import jax
import jax.numpy as jnp

import model


class O3filter:
    m1: jnp.array
    m2: jnp.array
    m3: jnp.array
    time_cst: float
    last_time: float
    mone: float

    def __init__(self, size, time_cst):
        self.m1 = jnp.zeros(size)
        self.m2 = jnp.zeros(size)
        self.m3 = jnp.zeros(size)
        self.time_cst = time_cst
        self.mone = 0.0
        self.factor = -jnp.expm1(-1.0 / self.time_cst)

    def update(self, val):
        self.mone += self.factor * (1 - self.mone)
        self.m1 = self.m1 + self.factor * (val - self.m1)
        self.m2 = self.m2 + self.factor * (self.m1 / self.mone - self.m2)
        self.m3 = self.m3 + self.factor * (self.m2 / self.mone - self.m3)

    @property
    def unbiaised_m3(self):
        return self.m3 / self.mone


@jax.jit
def mh_step_adaptative(*, sigma_proposal, current_ar, theta, z, y, t, prng_key):
    n, J = y.shape
    assert t.shape == (n, J)
    assert z.shape == (n, 2)
    prng_key, key = jax.random.split(prng_key)
    naccept, z = model.mh_step_gibbs(theta, z, y, t, sigma_proposal, key)
    current_ar += 0.02 * (naccept / n - current_ar)
    for i, _ in enumerate(sigma_proposal):
        sigma_proposal = jax.lax.cond(
            current_ar[i] < 0.4,
            lambda sigma_proposal: sigma_proposal.at[i].multiply(1 / 1.01),
            lambda sigma_proposal: sigma_proposal.at[i].multiply(1.01),
            sigma_proposal,
        )
    return (sigma_proposal, current_ar, z, prng_key)


@jax.jit
def one_iter(
    *,
    it,
    pre_heating,
    end_heating,
    theta,
    sigma_proposal,
    z,
    y,
    t,
    jac,
    current_ar,
    grad_mean,
    step_mean,
    prng_key,
):
    n, J = y.shape
    assert t.shape == (n, J)
    assert z.shape == (n, 2)

    if end_heating is None:
        factor = jax.lax.cond(
            it >= pre_heating,
            lambda it: 1.0,
            lambda it: jnp.exp((1 - it / pre_heating) * jnp.log(1e-4)),
            it,
        )
    else:
        factor = (it - end_heating) ** (-2/3)

    (sigma_proposal, current_ar, z, prng_key) = mh_step_adaptative(
        sigma_proposal=sigma_proposal,
        current_ar=current_ar,
        theta=theta,
        z=z,
        y=y,
        t=t,
        prng_key=prng_key,
    )

    current_jac = model.jac_log_likelihood_rows(theta, z, y, t)
    jac += factor * (current_jac - jac)
    fisher_info_mat = jac.T @ jac / n
    fisher_info_mat = jax.lax.cond(
        it < pre_heating,
        lambda fisher_info_mat: factor * fisher_info_mat + (1 - factor) * jnp.eye(model.parametrization.size),
        lambda fisher_info_mat: fisher_info_mat,
        fisher_info_mat,
    )
    grad = current_jac.mean(axis=0)
    theta_step = jnp.linalg.solve(fisher_info_mat, grad)
    theta += factor * theta_step

    step_mean = factor * (theta_step - step_mean)
    grad_mean = factor * (grad - grad_mean)

    return (
        theta,
        sigma_proposal,
        z,
        t,
        jac,
        current_ar,
        grad_mean,
        step_mean,
        prng_key,
        factor,
        theta_step,
        fisher_info_mat,
    )


def gsto(
    y,
    t,
    prng_key=None,
    pre_heating=1000,
    theta0=None,
):
    if prng_key is None:
        prng_key = 0
    if isinstance(prng_key, int):
        prng_key = jax.random.PRNGKey(prng_key)

    n, J = y.shape
    assert t.shape == (n, J)

    prng_key, key = jax.random.split(prng_key)
    if theta0 is None:
        theta = jax.random.normal(key=key, shape=(model.parametrization.size,))
    else:
        theta = theta0

    sigma_proposal = jnp.ones(2)
    current_ar = jnp.ones(1) * 0.4

    z = jnp.zeros((n, 2))
    for _ in range(pre_heating):
        (sigma_proposal, current_ar, z, prng_key) = mh_step_adaptative(
            sigma_proposal=sigma_proposal,
            current_ar=current_ar,
            theta=theta,
            z=z,
            y=y,
            t=t,
            prng_key=prng_key,
        )

    jac = jnp.zeros((n, model.parametrization.size))
    end_heating = None
    o3_filter = O3filter(model.parametrization.size, 100)
    o3_step_mean = jnp.zeros(model.parametrization.size)

    step_mean = jnp.zeros(model.parametrization.size)
    grad_mean = jnp.zeros(model.parametrization.size)

    for it in itertools.count():
        (
            theta,
            sigma_proposal,
            z,
            t,
            jac,
            current_ar,
            grad_mean,
            step_mean,
            prng_key,
            factor,
            theta_step,
            fisher_info_mat,
        ) = one_iter(
            it=it,
            pre_heating=pre_heating,
            end_heating=end_heating,
            theta=theta,
            sigma_proposal=sigma_proposal,
            z=z,
            y=y,
            t=t,
            jac=jac,
            current_ar=current_ar,
            grad_mean=grad_mean,
            step_mean=step_mean,
            prng_key=prng_key,
        )

        if end_heating is None:
            o3_filter.update(theta_step)
            o3_step_mean, old_o3_step_mean = o3_filter.unbiaised_m3, o3_step_mean
            if (
                it > pre_heating
                and (o3_step_mean**2).sum() > (old_o3_step_mean**2).sum()
            ):
                end_heating = it

        yield Retdata(
            it,
            end_heating,
            z,
            theta,
            factor,
            fisher_info_mat,
            step_mean,
            grad_mean,
        )


def estim(y, t, stop_crit=1e-6, N_smooth=10000, prng_key=0, pre_heating=1000, theta=None):
    n, J = y.shape
    assert t.shape == (n, J)
    usefull_values = list(
        itertools.islice(
                gsto(y, t, prng_key=prng_key, pre_heating=pre_heating, theta0=theta),
            N_smooth,
        )
    )    
    return ResEstim(
        jnp.array([x.theta for x in usefull_values]).mean(axis=0),
        jnp.array([x.theta for x in usefull_values]),
        jnp.array([x.factor for x in usefull_values]),        
        jnp.array([x.fisher_info_mat for x in usefull_values]).mean(axis=0),
        jnp.array([x.fisher_info_mat for x in usefull_values]),
        jnp.array([x.step_mean for x in usefull_values]),
        jnp.array([x.grad_mean for x in usefull_values]),
        #jnp.array([x.end_heating for x in usefull_values]),
    )


import collections

Retdata = collections.namedtuple(
    "Retdata",
    (
        "it",
        "end_heating",
        "z",
        "theta",
        "factor",
        "fisher_info_mat",
        "step_mean",
        "grad_mean",
    ),
)

ResEstim = collections.namedtuple("ResEstim", ("theta", "evol_theta", "factor", "fisher_info_mat", "evol_fim","step_mean","grad_mean"))#,"end_heating"))
