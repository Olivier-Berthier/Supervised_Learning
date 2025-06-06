# from sklearn.experimental import enable_iterative_imputer  # Activation explicite pour IterativeImputer
# from sklearn.impute import IterativeImputer
# from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.preprocessing import OneHotEncoder  # Assurez-vous que cela est importé

def RF_all_shap(X_train_shap, y_train, X_val_shap, y_val, categorical_columns, boolean_columns, numeric_columns):
    """
    Fonction pour entraîner un pipeline avec imputations, transformations et un RandomForestClassifier.
    Utilise GridSearchCV pour ajuster les hyperparamètres et affiche les résultats.
    Retourne le modèle entraîné de GridSearchCV.

    Args:
        X_train_shap (pd.DataFrame): Données d'entraînement (features).
        y_train (pd.Series): Labels d'entraînement.
        X_val_shap (pd.DataFrame): Données de validation (features).
        y_val (pd.Series): Labels de validation.
        # categorical_columns (list): Colonnes catégoriques à encoder.
        # boolean_columns (list): Colonnes de type booléen.
        # numeric_columns (list): Colonnes numériques nécessitant une imputation.

    Returns:
        grid_search_RF (GridSearchCV): Modèle GridSearchCV après entraînement.
    """
    # Définir le pipeline des transformations
    # preprocessor_pipeline = Pipeline(steps=[
    #     ('column_transformer', ColumnTransformer(
    #         transformers=[
    #             ('cat', 
    #              OneHotEncoder(handle_unknown='ignore'), 
    #              categorical_columns),  # Encoder les catégories
    #             ('bool', 
    #              'passthrough', 
    #              boolean_columns),  # Passer sans modification
    #             ('num', 
    #              IterativeImputer(max_iter=10, random_state=42), 
    #              numeric_columns)  # Imputation des valeurs manquantes pour les colonnes numériques
    #         ]
    #     ))
    # ])

    # Pipeline global avec le modèle
    pipeline_RF = Pipeline(steps=[
        ('classifier', RandomForestClassifier(n_jobs=-1, random_state=11))  # Modèle avec parallélisation (-1)
    ])

    # Dictionnaire des paramètres pour la recherche par grille
    param_grid = {
    'classifier__n_estimators': [100],                # Nombre d'arbres
    'classifier__max_depth': [30],                  # Profondeur maximale des arbres
    'classifier__min_samples_split': [10],         # Nombre minimal d'échantillons pour diviser un nœud
    'classifier__min_samples_leaf': [2],                # Nombre minimal d'échantillons dans les feuilles
    'classifier__max_features': [0.3],        # Fraction des features utilisées pour chaque arbre
    'classifier__class_weight': [None]         # Gérer le déséquilibre des classes
    }

    # GridSearchCV pour trouver les meilleurs hyperparamètres
    grid_search_RF = GridSearchCV(pipeline_RF, param_grid, cv=5, scoring='roc_auc')

    # Entraînement du modèle
    print("Entraînement du modèle avec GridSearchCV...")
    grid_search_RF.fit(X_train_shap, y_train)

    # Afficher les meilleurs paramètres et scores
    print("\nMeilleurs paramètres trouvés :", grid_search_RF.best_params_)
    print("Meilleur score ROC-AUC :", grid_search_RF.best_score_)

    # Evaluation sur les ensembles d'entraînement et de validation
    from pipelines_preprocessing.visualization import plot_confusion_matrices  # Import de l'affichage personnalisé
    print("\nAffichage des matrices de confusion...")
    plot_confusion_matrices(grid_search_RF, X_train_shap, y_train, X_val_shap, y_val, model_name="RF_all_shap")

    RF_all_best = grid_search_RF.best_estimator_

    # Retourner le modèle grid_search_RF
    return RF_all_best
