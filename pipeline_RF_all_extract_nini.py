from sklearn.experimental import enable_iterative_imputer  # Activation explicite pour IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder


def RF_all_extract_nini(X_train, y_train, X_val, y_val, categorical_columns, boolean_columns, numeric_columns, columns_nini):
    """
    Fonction pour entraîner un modèle Random Forest tout en filtrant les résultats pour les lignes de columns_nini.
    La colonne 'Unique_id' est utilisée seulement pour filtrer les résultats et n'est pas utilisée par le modèle.

    Args:
        X_train (pd.DataFrame): Données d'entraînement (features).
        y_train (pd.Series): Labels d'entraînement.
        X_val (pd.DataFrame): Données de validation (features).
        y_val (pd.Series): Labels de validation.
        categorical_columns (list): Colonnes catégoriques à encoder.
        boolean_columns (list): Colonnes de type booléen.
        numeric_columns (list): Colonnes numériques nécessitant une imputation.
        columns_nini (pd.DataFrame): DataFrame ayant une unique colonne 'Unique_id' pour filtrer les résultats.
    """

    # Vérification de la colonne attendue dans columns_nini
    if 'Unique_id' not in columns_nini.columns:
        raise ValueError("Le DataFrame `columns_nini` doit contenir une colonne 'Unique_id'.")

    # 1. Gestion temporaire de 'Unique_id'
    # Supprimer 'Unique_id' des jeux de données d'entraînement et validation pour le modèle
    X_train_no_id = X_train.drop(columns=['Unique_id'], errors='ignore')  # Supprimer Unique_id de X_train
    X_val_no_id = X_val.drop(columns=['Unique_id'], errors='ignore')  # Supprimer Unique_id de X_val

    # 2. Préparation du pipeline de prétraitement et du modèle
    preprocessor_pipeline = Pipeline(steps=[
        ('column_transformer', ColumnTransformer(
            transformers=[
                ('cat',
                 OneHotEncoder(handle_unknown='ignore'),
                 categorical_columns),  # Encoder les colonnes catégoriques
                ('bool',
                 'passthrough',
                 boolean_columns),  # Passer les booléens sans modification
                ('num',
                 IterativeImputer(max_iter=10, random_state=42),
                 numeric_columns)  # Imputation des valeurs manquantes pour les colonnes numériques
            ]
        ))
    ])

    # Pipeline complet avec Random Forest
    pipeline_RF = Pipeline(steps=[
        ('preprocessor', preprocessor_pipeline),
        ('classifier', RandomForestClassifier(n_jobs=-1, random_state=11))  # Modèle avec parallélisation
    ])

    # 3. Réglage des hyperparamètres via GridSearchCV
    param_grid = {
        'classifier__n_estimators': [100],
        'classifier__max_depth': [30],
        'classifier__min_samples_split': [10],
        'classifier__min_samples_leaf': [2],
        'classifier__max_features': [0.5],
        'classifier__class_weight': [None]
    }

    grid_search_RF = GridSearchCV(pipeline_RF, param_grid, cv=5, scoring='roc_auc')

    # Entraînement du modèle
    print("Entraînement du modèle avec GridSearchCV...")
    grid_search_RF.fit(X_train_no_id, y_train)

    # Afficher les meilleurs paramètres et score ROC-AUC
    print("\nMeilleurs paramètres trouvés :", grid_search_RF.best_params_)
    print("Meilleur score ROC-AUC (cross-validation) :", grid_search_RF.best_score_)

    # Filtrer les prédictions de validation avec columns_nini
    # Identifiez les lignes de validation correspondant à columns_nini
    filtered_unique_ids = columns_nini['Unique_id']
    X_val_filtered = X_val[X_val['Unique_id'].isin(filtered_unique_ids)].copy()
    y_val_filtered = y_val.loc[X_val_filtered.index]  # Synchroniser y_val avec X_val_filtered

    # Supprimer 'Unique_id' des données utilisées pour la prédiction
    X_val_filtered_no_id = X_val_filtered.drop(columns=['Unique_id'], errors='ignore')

    # Évaluation du modèle
    print("\nEvaluation du modèle sur les données de validation (filtrées avec columns_nini)...")
    from pipelines_preprocessing.visualization import plot_confusion_matrices
    plot_confusion_matrices(grid_search_RF, X_train_no_id, y_train, X_val_filtered_no_id, y_val_filtered, model_name="RF_all_extract_nini")
