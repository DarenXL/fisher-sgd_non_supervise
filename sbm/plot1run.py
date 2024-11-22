import many_estim
import jax
import algos
import tqdm
import collections
import matplotlib.pyplot as plt

n = 100
theta0, (Q, alpha0, pi0, mod) = many_estim.get_theta0()

prng_key_sim, prng_key_estim = jax.random.split(jax.random.PRNGKey(0))

Z, obs = many_estim.gen_one(prng_key_sim, n)

while True:
    local_key, prng_key_estim = jax.random.split(prng_key_estim)
    try:
        L = list(
            tqdm.tqdm(algos.estim_with_model(mod, Q, obs, local_key, yield_all=True))
        )
    except (algos.NanError, algos.EmptyError):
        continue
    break

perm, _, t0p = algos.labelswitch(mod, L[-1], Z, theta0)

end_smart = 1000
end_heat = len(L) - 20000

xlim = end_heat + 1000

PlotDesc = collections.namedtuple("PlotDesc", ("name", "fun"))

PLOTS = {
    "sbm_main": [
        [
            PlotDesc(r"\alpha_1", lambda p: p.alpha[perm[0]]),
            PlotDesc(r"\alpha_2", lambda p: p.alpha[perm[1]]),
        ],
        [
            PlotDesc(r"p_{11}", lambda p: p.pi[perm[0], perm[0]]),
            PlotDesc(r"p_{12}", lambda p: p.pi[perm[0], perm[1]]),
        ],
        [
            PlotDesc(r"p_{13}", lambda p: p.pi[perm[0], perm[2]]),
            PlotDesc(r"p_{14}", lambda p: p.pi[perm[0], perm[3]]),
        ],
    ],
    "sbm_all_alpha": [
        [
            PlotDesc(r"\alpha_1", lambda p: p.alpha[perm[0]]),
            PlotDesc(r"\alpha_2", lambda p: p.alpha[perm[1]]),
        ],
        [
            PlotDesc(r"\alpha_3", lambda p: p.alpha[perm[2]]),
            PlotDesc(r"\alpha_4", lambda p: p.alpha[perm[3]]),
        ],
    ],
    "sbm_all_pi": [
        [
            PlotDesc(
                r"p_{" + f"{q+1}{l+1}" + "}",
                (lambda q, l: lambda p: p.pi[perm[q], perm[l]])(q, l),
            )
            for l in range(Q)
        ]
        for q in range(Q)
    ],
}

for name, desc in PLOTS.items():
    fig, axs = plt.subplots(
        len(desc),
        len(desc[0]),
        sharex=True,
        sharey=True,
        figsize=(3 * len(desc[0]), 9 / 4 * len(desc)),
    )
    for a, b in zip(axs, desc):
        for ax, plot_desc in zip(a, b):
            ax.plot((end_smart, end_smart), (0, 1), "r")
            ax.plot((end_heat, end_heat), (0, 1), "g")
            ax.plot(
                (0, xlim - 1),
                (plot_desc.fun(mod.parametrization.reals1d_to_params(t0p)),) * 2,
                "y",
            )
            ax.plot(
                [
                    plot_desc.fun(mod.parametrization.reals1d_to_params(r.theta))
                    for r in L[:xlim]
                ]
            )
            ax.set_ylim(0, 1)
            ax.set_title(f"${plot_desc.name}$")
            if (a == axs[-1]).all():
                ax.set_xlabel("iterations")

    fig.show()
    fig.savefig(f"{name}.pdf")
