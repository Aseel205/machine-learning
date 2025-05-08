def get_best_model_large(X_train, y_train, X_val, y_val):
    if is_classification(y_train):
        candidates = [
            {
                "name": "LogisticRegression",
                "model": Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=300, random_state=42))]),
                "param_distributions": {}
            },
            {
                "name": "RandomForest",
                "model": RandomForestClassifier(n_estimators=30, max_depth=2, random_state=42),
                "param_distributions": {}
            },
            {
                "name": "XGBoost",
                "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_estimators=20, max_depth=2, verbosity=0),
                "param_distributions": {}
            }
        ]
        scoring = 'accuracy'
    else:
        candidates = [
            {
                "name": "LinearRegression",
                "model": LinearRegression(),
                "param_distributions": {}
            },
            {
                "name": "RandomForest",
                "model": RandomForestRegressor(n_estimators=30, max_depth=2, random_state=42),
                "param_distributions": {}
            },
            {
                "name": "XGBoost",
                "model": XGBRegressor(n_estimators=20, max_depth=2, random_state=42, verbosity=0),
                "param_distributions": {}
            }
        ]
        scoring = 'neg_mean_squared_error'

    best_model = None
    best_score = -np.inf if scoring == 'neg_mean_squared_error' else -1
    best_name = ""
    best_params = None

    cv = StratifiedKFold(n_splits=2) if is_classification(y_train) else KFold(n_splits=2)

    for candidate in candidates:
        model = candidate["model"]
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        score = accuracy_score(y_val, val_preds) if is_classification(y_train) else -mean_squared_error(y_val, val_preds)

        print(f"{candidate['name']} validation score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_name = candidate["name"]

    print(f"\nSelected Model: {best_name}")
    print("Validation score:", best_score)

    best_model.fit(np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0))
    return best_model