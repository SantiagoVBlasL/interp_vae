#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
classifiers10.py - Final Production Version
"""
from __future__ import annotations
import warnings
from typing import Any, Dict, List, Tuple

# Silenciar warnings de librerías
warnings.filterwarnings("ignore")
import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logging.getLogger("catboost").setLevel(logging.ERROR)
logging.getLogger("sklearn").setLevel(logging.ERROR)
logging.getLogger("xgboost").setLevel(logging.ERROR)
from lightgbm import LGBMClassifier, basic as lgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution

# Configuración de GPU y cuML (solo si está disponible)
try:
    import cupy as cp
    has_gpu = cp.cuda.runtime.getDeviceCount() > 0
except (ImportError, cp.cuda.runtime.CUDARuntimeError):
    has_gpu = False

def get_available_classifiers() -> List[str]:
    """Devuelve la lista de clasificadores soportados."""
    return ["rf", "gb", "svm", "logreg", "mlp", "xgb"]

def _parse_hidden_layers(hidden_layers_str: str | None) -> Tuple[int, ...]:
    if not hidden_layers_str:
        return (128, 64)
    return tuple(int(x.strip()) for x in hidden_layers_str.split(',') if x.strip())

logger = logging.getLogger(__name__)

def get_classifier_and_grid(
    classifier_type: str,
    *,
    seed: int = 42,
    latent_dim: int = 512,
    metadata_cols: List[str] = None,
    balance: bool = False,
    use_smote: bool = False,
    tune_sampler_params: bool = False,
    mlp_hidden_layers: str = "128,64",
    calibrate: bool = False
) -> Tuple[ImblearnPipeline, Dict[str, Any], int]:
    """
    Construye un pipeline de imblearn y devuelve el pipeline, el grid de búsqueda y el n_iter.
    """
    if metadata_cols is None:
        metadata_cols = []
    ctype = classifier_type.lower()
    if ctype not in get_available_classifiers():
        raise ValueError(f"Tipo de clasificador no soportado: {classifier_type!r}")

    class_weight = "balanced" if balance else None
    model: Any
    param_distributions: Dict[str, Any] = {}
    n_iter_search = 150

    if ctype == 'svm':
        model = SVC(probability=True, random_state=seed, class_weight=class_weight, cache_size=500)
        param_distributions = {
            'model__C': FloatDistribution(1e-2, 1e4, log=True),
            'model__gamma': FloatDistribution(1e-6, 1e1, log=True),
            'model__kernel': CategoricalDistribution(['rbf']),
        }
        n_iter_search = 200

    elif ctype == 'logreg':
        model = LogisticRegression(random_state=seed, class_weight=class_weight, solver='liblinear', max_iter=2000)
        param_distributions = {'model__C': FloatDistribution(1e-5, 1, log=True)}
        n_iter_search = 200

    elif ctype == "gb":
        model = LGBMClassifier(
            random_state=seed,
            objective="binary",
            class_weight=class_weight,
            n_jobs=1,
            verbose=-1,
        )

        # ---------- 1) ¿La librería fue compilada CON GPU? ----------
        gpu_available = False
        try:
            # -- si la función existe y devuelve 1, la build trae soporte
            from lightgbm.basic import _LIB, _safe_call
            if hasattr(_LIB, "LGBM_HasGPU"):
                gpu_available = bool(_safe_call(_LIB.LGBM_HasGPU()))
        except Exception:
            pass         # cualquier problema ⇒ asumimos que no

        # ---------- 2) ¿Hay hardware CUDA visible? ----------
        hw_gpu = has_gpu   # viene del bloque cupy de más arriba

        # ---------- 3) Elegir dispositivo ----------
        if gpu_available and hw_gpu:
            model.set_params(device_type="gpu", gpu_use_dp=True)
            logger.info("[LightGBM] → GPU habilitada")
        else:
            model.set_params(device_type="cpu")
            logger.info("[LightGBM] → usando CPU")

        # ---------- 4) Espacio de búsqueda ----------
        param_distributions = {
            "model__max_depth":        IntDistribution(3, 12),
            "model__num_leaves":       IntDistribution(8, 2**10),
            "model__bagging_fraction": FloatDistribution(0.5, 1.0),
            "model__feature_fraction": FloatDistribution(0.5, 1.0),
            "model__bagging_freq":     IntDistribution(1, 10),
            "model__learning_rate":    FloatDistribution(5e-4, 0.01, log=True),
            "model__n_estimators":     IntDistribution(300, 1000),
            "model__min_child_samples":IntDistribution(5, 50),
            "model__min_child_weight": FloatDistribution(1e-3, 10, log=True),
            "model__reg_alpha":        FloatDistribution(1e-3, 1.0, log=True),
            "model__reg_lambda":       FloatDistribution(1e-3, 1.0, log=True),
        }
        n_iter_search = 200

    elif ctype == 'rf':
        model = RandomForestClassifier(random_state=seed, class_weight=class_weight, n_jobs=-1)
        param_distributions = {
            'model__n_estimators': IntDistribution(100, 1200),
            'model__max_features': CategoricalDistribution(['sqrt', 'log2', 0.2, 0.4]),
            'model__max_depth': IntDistribution(8, 50),
            'model__min_samples_split': IntDistribution(2, 30),
            'model__min_samples_leaf': IntDistribution(1, 20),
        }
        n_iter_search = 150

    elif ctype == 'mlp':
        hidden = _parse_hidden_layers(mlp_hidden_layers)
        model = MLPClassifier(random_state=seed, hidden_layer_sizes=hidden, max_iter=1000, early_stopping=True, n_iter_no_change=25)
        param_distributions = {
            'model__alpha': FloatDistribution(1e-5, 1e-1, log=True),
            'model__learning_rate_init': FloatDistribution(1e-5, 1e-2, log=True),
        }
        n_iter_search = 200

    elif ctype == "xgb":
        model = XGBClassifier(random_state=seed, eval_metric="auc", n_jobs=1, verbosity=0)
        if has_gpu: model.set_params(tree_method="hist", device="cuda")
        param_distributions = {
            "model__gamma": FloatDistribution(0.0, 5.0),
            "model__n_estimators": IntDistribution(500, 1500),
            "model__learning_rate": FloatDistribution(1e-4, 0.1, log=True),
            "model__max_depth": IntDistribution(4, 12),
            "model__subsample": FloatDistribution(0.3, 1.0),
            "model__colsample_bytree": FloatDistribution(0.5, 1.0),
            "model__min_child_weight": FloatDistribution(0.5, 10., log=True),
            "model__scale_pos_weight": FloatDistribution(0.7, 1.5),
        }
        n_iter_search = 200

    if calibrate and ctype in ["svm", "gb", "rf"]:
        model = CalibratedClassifierCV(model, method="isotonic", cv=3)
        param_distributions = {f"model__base_estimator__{k.split('__', 1)[1]}": v for k, v in param_distributions.items()}

    latent_cols = [f'latent_{i}' for i in range(latent_dim)]
    numeric_meta_cols = [c for c in metadata_cols if c.lower() in ['age', 'years_of_education']]
    categorical_meta_cols = list(set(metadata_cols) - set(numeric_meta_cols))

    numeric_transformer = SklearnPipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    categorical_transformer = SklearnPipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('latent', StandardScaler(), latent_cols),
            ('num', numeric_transformer, numeric_meta_cols),
            ('cat', categorical_transformer, categorical_meta_cols)
        ],
        remainder='drop'
    )
    
    steps_ordered = [('preprocessor', preprocessor)]
    if use_smote:
        steps_ordered.append(('smote', SMOTE(random_state=seed, n_jobs=-1)))
        if tune_sampler_params:
            param_distributions['smote__k_neighbors'] = IntDistribution(3, 25)
            
    steps_ordered.append(('model', model))
    
    full_pipeline = ImblearnPipeline(steps=steps_ordered)

    return full_pipeline, param_distributions, n_iter_search