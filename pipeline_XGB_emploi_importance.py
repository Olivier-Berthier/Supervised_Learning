from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
import joblib


def XGB_emploi_importance(X_train, y_train, X_val, y_val, categorical_columns, boolean_columns, numeric_columns):
    

    # # Étape 1 : Définir le pipeline de préprocessing
    # preprocessor_pipeline = Pipeline(steps=[
    #     ('column_transformer', ColumnTransformer(
    #         transformers=[
    #             ('cat', 
    #              OneHotEncoder(handle_unknown='ignore'), 
    #              categorical_columns),  # Encoder les catégories
    #             ('bool', 
    #              'passthrough',
    #              boolean_columns),  # Passer les booléens sans modification
    #             ('num', 
    #              'passthrough',
    #             #IterativeImputer(max_iter=10, random_state=11),
    #              numeric_columns)  # Passer les numériques sans imputation
    #         ]
    #     ))
    # ])

    # Étape 2 : Définir le pipeline global avec XGBoost
    pipeline_XGB = Pipeline(steps=[
        #('preprocessor', preprocessor_pipeline),
        ('classifier', XGBClassifier(eval_metric='auc', use_label_encoder=False, n_jobs=-1))  # Modèle avec XGBoost
    ])

    # Étape 3 : Définir les hyperparamètres pour la grille de recherche
    param_grid = {
        'classifier__n_estimators': [1000],       # comparer 300 et 1000 Nombre d'arbres   déjà essayé [10000, 15000, 20000],
        'classifier__max_depth': [10],             # Profondeur maximale des arbres déjà essayé [0, 10, 20]
        'classifier__learning_rate': [0.02, 0.05, 0.07, 0.1],        # Taux d'apprentissage
        'classifier__subsample': [0.8],            # Fraction des lignes utilisées pour chaque arbre
        'classifier__colsample_bytree': [0.8],     # Fraction des colonnes utilisées pour chaque arbre
        'classifier__gamma': [0.5],                # Minimum loss reduction pour diviser un nœud
        'classifier__lambda': [1],                 # Régularisation L2
        'classifier__alpha': [0]                   # Régularisation L1
    }

    # Exécuter GridSearchCV
    print("Entraînement du modèle XGBoost avec GridSearchCV...")
    grid_search_XGB = GridSearchCV(pipeline_XGB, param_grid, cv=5, scoring='roc_auc')
    grid_search_XGB.fit(X_train, y_train)

    # # Obtenir le pipeline résultant du meilleur modèle
    # best_pipeline = grid_search_XGB.best_estimator_

    # # Extraire l'étape `preprocessor` (le pipeline du prétraitement)
    # preprocessor_pipeline = best_pipeline.named_steps['preprocessor']

    # # Extraire l'étape `column_transformer` (le `ColumnTransformer`)
    # column_transformer = preprocessor_pipeline.named_steps['column_transformer']

    # # Appliquer la transformation sur les données d'entraînement
    # transformed_array = column_transformer.transform(X_train)

    # # Nombre de colonnes après le prétraitement
    # n_features_after_preprocessing = transformed_array.shape[1]
    # print(f"Nombre de variables après preprocessing : {n_features_after_preprocessing}")

    # import joblib

    # # Meilleur modèle
    # XGB_emploi_fitted = grid_search_XGB.best_estimator_  # Récupérer le pipeline fitté

    # # Sauvegarder le pipeline fitté
    # joblib.dump(XGB_emploi_fitted, "models/fitted/XGB_emploi_fitted.pkl")

    # print("Pipeline fitté sauvegardé sous 'models/fitted/XGB_emploi_fitted.pkl'")

    # Afficher les meilleurs paramètres et le score obtenu
    #  print("\nMeilleurs paramètres trouvés :", grid_search_XGB.best_params_)

    # Visualisation des résultats à l'aide de matrices de confusion
    from pipelines_preprocessing.visualization import plot_confusion_matrices
    print("\nAffichage des matrices de confusion...")
    plot_confusion_matrices(grid_search_XGB, X_train, y_train, X_val, y_val, model_name="XGB_emploi")

   
