# Importer les bibliothèques nécessaires
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def XGB_all_extract_nini(X_train, y_train, X_val, y_val, categorical_columns, boolean_columns, numeric_columns, columns_nini):
    """
    Fonction pour entraîner un pipeline avec imputations, transformations et un XGBoostClassifier.
    Utilise GridSearchCV pour ajuster les hyperparamètres et filtre les résultats
    pour les lignes spécifiées dans `columns_nini`.

    Args:
        X_train (pd.DataFrame): Données d'entraînement (features).
        y_train (pd.Series): Labels d'entraînement.
        X_val (pd.DataFrame): Données de validation (features).
        y_val (pd.Series): Labels de validation.
        categorical_columns (list): Colonnes catégoriques à encoder.
        boolean_columns (list): Colonnes de type booléen.
        numeric_columns (list): Colonnes numériques nécessitant une transformation.
        columns_nini (pd.DataFrame): DataFrame ayant une unique colonne 'Unique_id' pour filtrer les résultats.
    """

    # Vérification de la colonne attendue dans columns_nini
    if 'Unique_id' not in columns_nini.columns:
        raise ValueError("Le DataFrame `columns_nini` doit contenir une colonne 'Unique_id'.")

    # 1. Gestion temporaire de 'Unique_id'
    # Retirer 'Unique_id' des jeux de données d'entraînement et validation pour le modèle
    X_train_no_id = X_train.drop(columns=['Unique_id'], errors='ignore')  # Supprimer Unique_id de X_train
    X_val_no_id = X_val.drop(columns=['Unique_id'], errors='ignore')  # Supprimer Unique_id de X_val

    # 2. Définir le pipeline de préprocessing
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
                 numeric_columns)  # Passer les numériques sans imputation
            ]
        ))
    ])

    # 3. Définir le pipeline global avec XGBoost
    pipeline_XGB = Pipeline(steps=[
        ('preprocessor', preprocessor_pipeline),
        ('classifier', XGBClassifier(eval_metric='auc', use_label_encoder=False, n_jobs=-1))  # Modèle avec XGBoost
    ])

    # 4. Définir les hyperparamètres pour la grille de recherche
    param_grid = {
        'classifier__n_estimators': [200],         # Nombre d'arbres
        'classifier__max_depth': [10],             # Profondeur maximale des arbres
        'classifier__learning_rate': [0.1],        # Taux d'apprentissage
        'classifier__subsample': [0.8],            # Fraction des lignes utilisées pour chaque arbre
        'classifier__colsample_bytree': [0.8],     # Fraction des colonnes utilisées pour chaque arbre
        'classifier__gamma': [0.5],                # Minimum loss reduction pour diviser un nœud
        'classifier__lambda': [1],                 # Régularisation L2
        'classifier__alpha': [0]                   # Régularisation L1
    }

    # Initialiser un GridSearchCV pour optimiser les hyperparamètres
    print("Entraînement du modèle XGBoost avec GridSearchCV...")
    grid_search_XGB = GridSearchCV(pipeline_XGB, param_grid, cv=5, scoring='roc_auc')
    grid_search_XGB.fit(X_train_no_id, y_train)  # Entraîner le modèle

    # Afficher les meilleurs paramètres et score obtenu
    print("\nMeilleurs paramètres trouvés :", grid_search_XGB.best_params_)
    print("Meilleur score ROC-AUC (cross-validation) :", grid_search_XGB.best_score_)

    # 5. Filtrer les prédictions de validation avec columns_nini
    # Identifiez les lignes de validation correspondant à columns_nini
    filtered_unique_ids = columns_nini['Unique_id']
    X_val_filtered = X_val[X_val['Unique_id'].isin(filtered_unique_ids)].copy()
    y_val_filtered = y_val.loc[X_val_filtered.index]  # Synchroniser y_val avec les lignes filtrées

    # Supprimer 'Unique_id' des données utilisées pour la prédiction
    X_val_filtered_no_id = X_val_filtered.drop(columns=['Unique_id'], errors='ignore')

    # 6. Évaluation et visualisation des résultats sur les lignes filtrées
    print("\nEvaluation du modèle sur les données de validation (filtrées avec columns_nini)...")
    from pipelines_preprocessing.visualization import plot_confusion_matrices
    plot_confusion_matrices(grid_search_XGB, X_train_no_id, y_train, X_val_filtered_no_id, y_val_filtered, model_name="XGB_all_extract_nini")
