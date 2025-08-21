if __name__ == '__main__':
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
    import concurrent.futures
    from tqdm import tqdm
    from feature_extractor import URLFeatureExtractor
    import numpy as np

    # -------- Feature extraction helper --------
    def safe_extract_features(index_url):
        i, url = index_url
        try:
            extractor = URLFeatureExtractor(url)
            features = extractor.extract_features()
            if features is None:
                return None
            return i, features
        except Exception:
            return None

    # -------- Load dataset --------
    df = pd.read_csv("url_dataset_balanced.csv")
    print("Label distribution:\n", df['label'].value_counts())

    # -------- Extract features concurrently --------
    print("⚙️ Extracting features...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(safe_extract_features, enumerate(df['url'])), total=len(df)))

    valid_results = [res for res in results if res is not None]
    if not valid_results:
        raise Exception("❌ No valid URLs were processed. Check your dataset.")

    indices, feature_rows = zip(*valid_results)
    features_df = pd.DataFrame(feature_rows)
    labels = df.loc[list(indices), 'label'].reset_index(drop=True)

    # -------- Compute scale_pos_weight --------
    num_pos = (labels == 1).sum()
    num_neg = (labels == 0).sum()
    scale_pos_weight = num_neg / num_pos
    print(f"scale_pos_weight = {scale_pos_weight:.2f} (neg={num_neg}, pos={num_pos})")

    # -------- 80/20 Train-Test Split --------
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # -------- Convert to DMatrix for xgboost.train --------
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # -------- Training parameters --------
    params = {
        "objective": "binary:logistic",
        "max_depth": 5,
        "learning_rate": 0.01,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "gamma": 2,
        "min_child_weight": 2,
        "reg_lambda": 4,
        "eval_metric": "logloss",
        "tree_method": "hist",
        "scale_pos_weight": scale_pos_weight
    }

    print("🚀 Training model...")
    evals = [(dtrain, "train")]

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=evals,
        verbose_eval=True,
        early_stopping_rounds=50
    )

    # -------- Evaluate --------
    y_pred_prob = bst.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # -------- Save model --------
    bst.save_model("url_xgb_model_v2.json")
    print("✅ Model saved as url_xgb_model_v2.json")
