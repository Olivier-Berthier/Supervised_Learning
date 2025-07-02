# Apprentissage-Supervisé
Projet Executive Master Statistique & Intelligence artificielle (Master 2 - Dauphine PSL)
Classification sur ensemble de test (muet) 

Objectif du projet
Ce projet consiste à construire un modèle prédictif afin d’estimer une caractéristique non divulguée liée au statut socio-économique de 100 087 individus résidant en France métropolitaine. La variable cible à prédire n’est disponible que dans le jeu de données d’apprentissage. Le travail porte particulièrement sur la préparation de données, la mise en œuvre de modèles avancés, la gestion des valeurs manquantes, et l’évaluation des performances via des métriques standards et la validation croisée.

Contraintes :
Aucun traitement manuel sur les fichiers de données (tout doit être fait en code)
Rapport rendu sous un format de qualité académique/reproductible.
Importance accordée à la qualité des prédictions produites sur un ensemble de test
Déroulement
Prétraitement & exploration :

Jointure et sélection raisonnée des sous-ensembles de données les plus pertinents.
Gestion des valeurs manquantes par différentes méthodes d’imputation (par exemple, IterativeImputer de scikit-learn).

Modélisation :

Évaluation de divers modèles, XGBoost a été retenu pour sa performance.
Comparaison avec Random Forests.

Évaluation :

Application de la validation croisée.
Surveillance du surapprentissage via écart de performance entre validation croisée et test final.

Code complet : Facilement exécutable, structuré pour la reproductibilité (Python).
Rapport synthétique : Présentation des choix méthodologiques, des résultats et des interprétations.
Fichier de prédictions prêt à être évalué par un script automatique.
Points notables
Analyse rigoureuse du compromis entre performance en cross-validation et généralisation sur les données de test.
Prise de décision motivée concernant l’inclusion/exclusion de données annexes pour limiter le surapprentissage.
Résumé pour Github

Ce projet met en œuvre toutes les étapes d’un pipeline de machine learning supervisé de manière rigoureuse, dans le respect des exigences de reproductibilité et d’automatisation.

