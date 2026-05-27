"""
train_model.py
──────────────
Run this once to train and save the model before launching the Flask app.
Usage:
    python train_model.py --data path/to/job_salary_prediction_dataset.csv
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train(data_path: str) -> None:
    print(f"[INFO] Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[INFO] Shape: {df.shape}")

    # ── Categorical columns ─────────────────────────────────────────────────
    OR_COLS  = ['education_level', 'company_size']
    OHE_COLS = ['job_title', 'industry', 'location', 'remote_work']

    cdf = df.select_dtypes(include='object')
    ndf = df.select_dtypes(exclude='object')

    oe  = OrdinalEncoder()
    ohe = OneHotEncoder(sparse_output=False)

    oe_enc  = oe.fit_transform(cdf[OR_COLS])
    ohe_enc = ohe.fit_transform(cdf[OHE_COLS])

    oe_enc_df  = pd.DataFrame(oe_enc,  columns=OR_COLS)
    ohe_enc_df = pd.DataFrame(ohe_enc, columns=ohe.get_feature_names_out())

    # Print ordinal mappings for reference
    for col_name, categories in zip(OR_COLS, oe.categories_):
        print(f"\nOrdinal mappings for: {col_name}")
        for idx, cls in enumerate(categories):
            print(f"  {idx} → {cls}")

    final_cdf = pd.concat([ohe_enc_df, oe_enc_df], axis=1)
    final_df  = pd.concat([final_cdf, ndf], axis=1)

    X = final_df.drop('salary', axis=1)
    y = final_df['salary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Train best model ────────────────────────────────────────────────────
    print("\n[INFO] Training Random Forest (this may take a minute)…")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    print("\n── Evaluation ──────────────────────")
    print(f"MAE      : {mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"RMSE     : {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    print(f"R² Score : {r2_score(y_test, y_pred):.4f}")

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs('Best_model', exist_ok=True)
    path = 'Best_model/random_forest_model.pkl'
    joblib.dump(rf, path)
    print(f"\n[SUCCESS] Model saved to: {path}")
    print("[INFO] You can now start the Flask app with: python app.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train salary prediction model')
    parser.add_argument('--data', required=True, help='Path to CSV dataset')
    args = parser.parse_args()
    train(args.data)
