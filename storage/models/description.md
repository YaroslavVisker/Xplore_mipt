Purpose: Machine learning models used to infer additional features.

Expected contents:
- Serialized ML models (pickle, joblib, ONNX, Torch, etc.)
- Model metadata (training date, feature schema, version)
- Pipelines for generating derived predictors:
    - risk scores
    - predictions for missing values
    - probability models