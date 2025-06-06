# Importer les bibliothèques nécessaires
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_iterative_imputer


def XGB_retired(X_train, y_train, X_val, y_val, categorical_columns, boolean_columns, numeric_columns):
    """
    Fonction pour entraîner un pipeline avec imputations, transformations et un XGBoostClassifier.
    Utilise GridSearchCV pour ajuster les hyperparamètres et affiche les résultats
    sans retourner de valeurs.

    Args:
        X_train (pd.DataFrame): Données d'entraînement (features).
        y_train (pd.Series): Labels d'entraînement.
        X_val (pd.DataFrame): Données de validation (features).
        y_val (pd.Series): Labels de validation.
        categorical_columns (list): Colonnes catégoriques à encoder.
        boolean_columns (list): Colonnes de type booléen.
        numeric_columns (list): Colonnes numériques nécessitant une transformation.
    """

    # Étape 1 : Définir le pipeline de préprocessing
    preprocessor_pipeline = Pipeline(steps=[
        ('column_transformer', ColumnTransformer(
            transformers=[
                ('cat', 
                 OneHotEncoder(handle_unknown='ignore'), 
                 categorical_columns),  # Encoder les catégories
                ('bool', 
                 'passthrough',
                 boolean_columns),  # Passer les booléens sans modification
                ('num', 
                 'passthrough',
                 # IterativeImputer(max_iter=10, random_state=11),
                 numeric_columns)  # Passer les numériques sans imputation
            ]
        ))
    ])

    # Définir le pipeline global avec XGBoost
    pipeline_XGB = Pipeline(steps=[
        ('preprocessor', preprocessor_pipeline),
        ('classifier', XGBClassifier(eval_metric='auc', use_label_encoder=False, n_jobs=-1))  # Modèle avec XGBoost
    ])

    # Définir les hyperparamètres pour la grille de recherche
    param_grid = {
        'classifier__n_estimators': [10000],         # Nombre d'arbres
        'classifier__max_depth': [10],             # Profondeur maximale des arbres
        'classifier__learning_rate': [0.04],        # Taux d'apprentissage   [0.04, 0.05, 0.06]
        'classifier__subsample': [0.9],            # Fraction des lignes utilisées pour chaque arbre   [0.8, 0.9, 1]
        'classifier__colsample_bytree': [0.8],     # Fraction des colonnes utilisées pour chaque arbre
        'classifier__gamma': [0.5],                # Minimum loss reduction pour diviser un nœud
        'classifier__lambda': [1],                 # Régularisation L2
        'classifier__alpha': [0.1]                   # Régularisation L1
    }

    # Exécuter GridSearchCV
    print("Entraînement du modèle XGBoost avec GridSearchCV...")
    grid_search_XGB = GridSearchCV(pipeline_XGB, param_grid, cv=5, scoring='roc_auc')
    grid_search_XGB.fit(X_train, y_train)

    # Obtenir le pipeline résultant du meilleur modèle
    best_pipeline = grid_search_XGB.best_estimator_

    # Extraire l'étape `preprocessor` (le pipeline du prétraitement)
    preprocessor_pipeline = best_pipeline.named_steps['preprocessor']

    # Extraire l'étape `column_transformer` (le `ColumnTransformer`)
    column_transformer = preprocessor_pipeline.named_steps['column_transformer']

    # Appliquer la transformation sur les données d'entraînement
    transformed_array = column_transformer.transform(X_train)

    # Nombre de colonnes après le prétraitement
    n_features_after_preprocessing = transformed_array.shape[1]
    print(f"Nombre de variables après preprocessing : {n_features_after_preprocessing}")

    import joblib

    # Meilleur modèle
    XGB_retired_fitted = grid_search_XGB.best_estimator_  # Récupérer le pipeline fitté

    # Sauvegarder le pipeline fitté
    joblib.dump(XGB_retired_fitted, "models/fitted/XGB_retired_fitted.pkl")

    print("Pipeline fitté sauvegardé sous 'models/fitted/XGB_retired_fitted.pkl'")

    # Afficher les meilleurs paramètres et le score obtenu
    print("\nMeilleurs paramètres trouvés :", grid_search_XGB.best_params_)

    # Visualisation des résultats à l'aide de matrices de confusion
    from pipelines_preprocessing.visualization import plot_confusion_matrices
    print("\nAffichage des matrices de confusion...")
    plot_confusion_matrices(grid_search_XGB, X_train, y_train, X_val, y_val, model_name="XGB_retired")


