import many_estim
import jax
import jax.numpy as jnp

N = 10 # 2000

n = 100
retries = 1 # 20
import algos
import multiprocessing as mpc
from tqdm import tqdm
from interruptible_list import interruptible_list

# with mpc.Pool(48) as p:
#     R = list(
#         tqdm(
#             p.imap_unordered(
#                 many_estim.do_one_estim, ((key, n, retries) for key in range(N))
#             ),
#             total=N,
#             smoothing=0,
#         ),
#         save_whole=True,
#     )

R = list(
        tqdm(
            (
                many_estim.do_one_estim((key, n, retries)) for key in range(N)
            ),
            total=N,
            smoothing=0,
        ),
)

print([R[i][1].theta for i in range(N)])

# import gzip
# import pickle

# pickle.dump(R, gzip.open("sbm_100.pkl.gz", "wb"))

# with open('sbm_100.npy', 'wb') as f:
#     jnp.save(f,R)
