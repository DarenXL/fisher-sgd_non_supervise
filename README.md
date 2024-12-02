# Estimation de paramètres de modèles à variables latentes par Fisher-SGD

Ce git contient des implémentations pour l'algorithme Fisher-SGD [1] sur différents modèles.

## mixed-effect :
Implémentation faite par les auteurs du Fisher-SGD sur *mixed-effect models*.

#### Données simulées
* `run_simus.py` pour la génération de données simulées et l'exécution de Fisher-SGD sur ces mêmes variables (*attention à la variable dans `config.py`*).
* `plots.py` pour afficher un exemple d'évolution des estimateurs.
* `post_analysis.py` pour l'analyse du Fisher-SGD sur les données simulées.
* `saem.R` pour l'exécution du SAEM sur les mêmes données simulées.

#### Données réelles
* `run_realdata.py` pour l'exécution et l'analyse du Fisher-SGD sur les données réelles : https://datadryad.org/stash/dataset/doi:10.5061/dryad.23gt0

## SBM :
Implémentation faite par les auteurs du Fisher-SGD sur *stochastic block models*.

* `plot1run.py` pour afficher un exemple d'évolution des estimateurs.
* `run_all_100/200.py` pour l'exécution du Fisher-SGD sur des données simulées.
* `build_res.py` pour l'analyse du Fisher-SGD sur les données simulées.

## mixture
Implémentation du Fisher-SGD sur les modèles de mélange.



*[1] Charlotte Baey, Maud Delattre, Estelle Kuhn, Jean-Benoist Leger, and
Sarah Lemler. Efficient preconditioned stochastic gradient descent for es-
timation in latent variable models. June 2023.*
