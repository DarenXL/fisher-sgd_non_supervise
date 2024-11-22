import algos
from model import SBMModel
import many_estim

theta0, (Q, alpha0, pi0, model) = many_estim.get_theta0()

import pickle
import gzip
import tqdm

import jax.numpy as jnp
import numpy as np
import scipy.stats


def _res(pklfile):
    L = pickle.load(gzip.open(pklfile, "rb"))
    perms_theta0 = list(
        tqdm.tqdm(
            (((x := algos.labelswitch(model, r, Z, theta0))[0], x[2]) for Z, r in L),
            total=len(L),
        )
    )

    rmse_theta = np.sqrt(
        np.array(
            [
                ((res.theta - t0) ** 2).sum()
                for (_, res), (_, t0) in zip(L, perms_theta0)
            ]
        ).mean()
    )
    print(f"RMSE theta: {rmse_theta}")

    seuil = scipy.stats.chi2(theta0.size).ppf(0.95)
    c_theta = np.array(
        [
            (res.theta - t0) @ res.FIM @ (res.theta - t0) < seuil
            for (_, res), (_, t0) in zip(L, perms_theta0)
        ]
    ).mean()
    c_theta_pm = scipy.stats.norm.ppf(0.975) * np.sqrt(c_theta * (1 - c_theta) / len(L))
    print(f"Global empirical coverage of confidence ellipsoid: {c_theta}±{c_theta_pm}")

    all_estimates = [
        algos.build_params_est_and_std(model, res, perm)
        for (_, res), (perm, _) in zip(L, perms_theta0)
    ]
    all_params0 = jnp.concatenate((alpha0, pi0.ravel()))
    rmse_all_params = np.sqrt(
        (np.array([pest - all_params0 for (pest, _) in all_estimates]) ** 2).mean(
            axis=0
        )
    )
    print(f"RMSE original params:\n{rmse_all_params}")

    seuil = scipy.stats.norm.ppf(0.975)
    c_all_params = np.array(
        [np.abs(pest - all_params0) < seuil * pstd for (pest, pstd) in all_estimates]
    ).mean(axis=0)
    c_all_params_pm = scipy.stats.norm.ppf(0.975) * np.sqrt(
        c_all_params * (1 - c_all_params) / len(L)
    )
    print(f"Empirical coverage of original params: {c_all_params}±{c_all_params_pm}")

    print("\nLaTeX table")
    params_name = [rf"$\alpha_{k}$" for k in range(1, Q + 1)] + [
        r"$p_{" + f"{k1},{k2}" + "}$"
        for k1 in range(1, Q + 1)
        for k2 in range(1, Q + 1)
    ]

    print(r"{} & Simulated value & RMSE & Empirical coverage \\")
    print(
        rf"Global $\theta$ &         & ${rmse_theta:.3f}$ & ${c_theta:.3f}\pm{c_theta_pm:.3f}$ \\"
    )
    for z, pname in enumerate(params_name):
        print(
            rf"{pname:15s} & ${all_params0[z]:.3f}$ & ${rmse_all_params[z]:.3f}$ & ${c_all_params[z]:.3f}\pm{c_all_params_pm[z]:.3f}$ \\"
        )


print("SBM 100")
_res("outputs/sbm_100_small.pkl.gz")

print("\n\n\n")
print("SBM 200")
_res("outputs/sbm_200_small.pkl.gz")
