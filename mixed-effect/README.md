# Fisher-SGD in nonlinear mixed-effect models

This is the code associated to the 2023 ICML paper "Efficient preconditioned stochastic gradient descent for estimation in  latent variable models" by C. Baey, M. Delattre, E. Kuhn, J.-B. Leger and S. Lemler.

It performs parameter estimation in a logistic mixed-effect model with two random effects and one fixed effect.

## Dependencies

All modules are available on PyPI:
 - `jax`
 - `numpy`
 - `scipy`
 - `parametrization_cookbook`
 - `matplotlib`
 - `pickle`
 - `functools`
 - `maths`
 - `itertools`
 
 And on the CRAN:
 - `saemix`
 - `dplyr`
 - `tidyr`
 
 ## How to reproduce the results -- Python and Fisher-SGD
 
  
 ### Run algo
 ```
 python3 run_simus.py
 ```
 
 ### Figures 
 ```
 python3 plots.py
 ```
 
 ### Table 1
 ```
 python3 post_analysis.py
 ```

 ## How to reproduce the results -- R and SAEM
You first need to run the simulations in Python. It will create a folder with the simulate data, which can then be called by the R script.

 ```
 R saem.R
 ```

 ## On real data
 ```
 python3 run_realdata.py
 ```
