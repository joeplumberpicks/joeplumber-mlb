from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


PA_OUTCOME_CLASSES = [
    "out",
    "walk_hbp",
    "single",
    "double",
    "triple",
    "home_run",
]

PA_OUTCOME_TO_ID = {name: i for i, name in enumerate(PA_OUTCOME_CLASSES)}
PA_ID_TO_OUTCOME = {i: name for name, i in PA_OUTCOME_TO_ID.items()}


@dataclass
class PaOutcomeArtifact:
    model: Pipeline
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    class_names: list[str]
    metadata: dict[str, Any]


def build_pa_target(df: pd.DataFrame) -> pd.Series:
    out = pd.Series("out", index=df.index, dtype="string")

    is_bb = df.get("is_bb", False)
    is_hbp = df.get("is_hbp", False)
    is_1b = df.get("is_1b", False)
    is_2b = df.get("is_2b", False)
    is_3b = df.get("is_3b", False)
    is_hr = df.get("is_hr", False)

    out.loc[is_bb | is_hbp] = "walk_hbp"
    out.loc[is_1b] = "single"
    out.loc[is_2b] = "double"
    out.loc[is_3b] = "triple"
    out.loc[is_hr] = "home_run"

    return out


def encode_pa_target(y: pd.Series) -> np.ndarray:
    y = y.astype("string").fillna("out")
    return y.map(PA_OUTCOME_TO_ID).astype(int).to_numpy()


def decode_pa_target(y: np.ndarray | list[int]) -> list[str]:
    return [PA_ID_TO_OUTCOME[int(v)] for v in y]


def _infer_feature_types(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    numeric_features = []
    categorical_features = []

    for col in feature_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    return numeric_features, categorical_features


def make_pa_outcome_pipeline(
    df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int = 42,
) -> tuple[Pipeline, list[str], list[str]]:
    numeric_features, categorical_features = _infer_feature_types(df, feature_columns)

    num_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_features),
            ("cat", cat_tf, categorical_features),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(PA_OUTCOME_CLASSES),
        eval_metric="mlogloss",
        n_estimators=350,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipe, numeric_features, categorical_features


def fit_pa_outcome_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    random_state: int = 42,
) -> PaOutcomeArtifact:
    pipe, numeric_features, categorical_features = make_pa_outcome_pipeline(
        df=train_df,
        feature_columns=feature_columns,
        random_state=random_state,
    )

    y = encode_pa_target(train_df["pa_outcome_target"])
    X = train_df[feature_columns].copy()

    pipe.fit(X, y)

    metadata = {
        "n_train": int(len(train_df)),
        "class_names": PA_OUTCOME_CLASSES,
        "random_state": random_state,
    }

    return PaOutcomeArtifact(
        model=pipe,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        class_names=PA_OUTCOME_CLASSES,
        metadata=metadata,
    )


def predict_pa_outcome_proba(
    artifact: PaOutcomeArtifact,
    df: pd.DataFrame,
) -> pd.DataFrame:
    X = df[artifact.feature_columns].copy()

    proba = artifact.model.predict_proba(X)

    return pd.DataFrame(
        proba,
        columns=[f"p_{c}" for c in artifact.class_names],
        index=df.index,
    )


def predict_pa_outcome_class(
    artifact: PaOutcomeArtifact,
    df: pd.DataFrame,
) -> pd.Series:
    X = df[artifact.feature_columns].copy()

    pred = artifact.model.predict(X)

    return pd.Series(
        decode_pa_target(pred),
        index=df.index,
        name="pred_pa_outcome",
    )


def save_pa_outcome_artifact(
    artifact: PaOutcomeArtifact,
    out_dir: str | Path,
    artifact_name: str = "pa_outcome_xgb",
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{artifact_name}.pkl"
    meta_path = out_dir / f"{artifact_name}.meta.json"

    with model_path.open("wb") as f:
        pickle.dump(artifact, f)

    payload = {
        "model_path": str(model_path),
        "feature_columns": artifact.feature_columns,
        "numeric_features": artifact.numeric_features,
        "categorical_features": artifact.categorical_features,
        "class_names": artifact.class_names,
        "metadata": artifact.metadata,
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "model_path": str(model_path),
        "meta_path": str(meta_path),
    }


def load_pa_outcome_artifact(model_path: str | Path) -> PaOutcomeArtifact:
    model_path = Path(model_path)

    with model_path.open("rb") as f:
        return pickle.load(f)
