# Estimation de paramètres de modèles à variables latentes par Fisher-SGD

Ce git contient des implémentations pour l'algorithme Fisher-SGD [1] sur différents modèles.

## mixed-effect :
Implémentation réalisée par les auteurs de l'article [1] pour les modèles à effets mixtes afin de retrouver leurs résultats.
Voir le code original ici : https://github.com/baeyc/fisher-sgd-nlme

#### Données simulées
* `run_simus.py` pour la génération de données simulées et l'exécution de Fisher-SGD sur ces mêmes variables (*attention à la variable dans `config.py`*).
* `plots.py` pour afficher un exemple d'évolution des estimateurs.
* `post_analysis.py` pour l'analyse du Fisher-SGD sur les données simulées.
* `saem.R` pour l'exécution du SAEM sur les mêmes données simulées.

#### Données réelles
* `run_realdata.py` pour l'exécution et l'analyse du Fisher-SGD sur les données réelles : https://datadryad.org/stash/dataset/doi:10.5061/dryad.23gt0

## SBM :
Implémentation réalisée par les auteurs de l'article [1] pour les modèles de blocs stochastiques afin de retrouver leurs résultats.
Voir le code original ici : https://gitlab.com/jbleger/sbm_with_fisher-sgd

* `plot1run.py` pour afficher un exemple d'évolution des estimateurs.
* `run_all_100/200.py` pour l'exécution du Fisher-SGD sur des données simulées.
* `build_res.py` pour l'analyse du Fisher-SGD sur les données simulées.

## mixture
Notre implémentation personelle du Fisher-SGD pour les modèles de mélange.
Comparaison avec les méthodes EM fournies par Rmixmod et les algorithmes de descente classique (SGD)

*[1] Charlotte Baey, Maud Delattre, Estelle Kuhn, Jean-Benoist Leger, and
Sarah Lemler. Efficient preconditioned stochastic gradient descent for es-
timation in latent variable models. June 2023.* https://arxiv.org/pdf/2306.12841
