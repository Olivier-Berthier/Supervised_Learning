from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
import joblib


def XGB_emploi(X_train, y_train, X_val, y_val, categorical_columns, boolean_columns, numeric_columns):
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
                 #IterativeImputer(max_iter=10, random_state=11),
                 numeric_columns)  # Passer les numériques sans imputation
            ]
        ))
    ])

    # Étape 2 : Définir le pipeline global avec XGBoost
    pipeline_XGB = Pipeline(steps=[
        ('preprocessor', preprocessor_pipeline),
        ('classifier', XGBClassifier(eval_metric='auc', use_label_encoder=False, n_jobs=-1))  # Modèle avec XGBoost
    ])

    # Étape 3 : Définir les hyperparamètres pour la grille de recherche
    param_grid = {
        'classifier__n_estimators': [1000],       # comparer 300 et 1000 Nombre d'arbres   déjà essayé [10000, 15000, 20000],
        'classifier__max_depth': [10,],             # Profondeur maximale des arbres déjà essayé [0, 10, 20] et [5, 10, 15]
        'classifier__learning_rate': [0.05],        # Taux d'apprentissage      déjà essayé [0.02, 0.05, 0.07]
        'classifier__subsample': [0.9],            # Fraction des lignes utilisées pour chaque arbre  déjà essayé [0.7, 0.8, 0.9, 1]
        'classifier__colsample_bytree': [0.7],     # Fraction des colonnes utilisées pour chaque arbre  déjà essayé  [0.3, 0.5, 0.7]
        'classifier__gamma': [0.4],                # Minimum loss reduction pour diviser un nœud       déjà essayé   [0.2, 0.3, 0.4, 0.5, 0.6]
        'classifier__lambda': [0.5],                 # Régularisation L2                              déjà essayé       [0.1, 0.5, 1]
        'classifier__alpha': [0.1]                   # Régularisation L1        [0, 0.1, 0.2]                     déjà essayé    [0, 0.1]
    }

    # Exécuter GridSearchCV
    print("Entraînement du modèle XGBoost avec GridSearchCV...")
    grid_search_XGB = GridSearchCV(pipeline_XGB, param_grid, cv=5, scoring='roc_auc')   # 'roc_auc', 'accuracy'
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
    XGB_emploi_fitted = grid_search_XGB.best_estimator_  # Récupérer le pipeline fitté

    # Sauvegarder le pipeline fitté
    joblib.dump(XGB_emploi_fitted, "models/fitted/XGB_emploi_fitted.pkl")

    print("Pipeline fitté sauvegardé sous 'models/fitted/XGB_emploi_fitted.pkl'")

    # Afficher les meilleurs paramètres et le score obtenu
    #  print("\nMeilleurs paramètres trouvés :", grid_search_XGB.best_params_)

    # Visualisation des résultats à l'aide de matrices de confusion
    from pipelines_preprocessing.visualization import plot_confusion_matrices
    print("\nAffichage des matrices de confusion...")
    plot_confusion_matrices(grid_search_XGB, X_train, y_train, X_val, y_val, model_name="XGB_emploi")

   
