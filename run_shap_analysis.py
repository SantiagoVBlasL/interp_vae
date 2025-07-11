# run_shap_analysis.py
from typing import Tuple, Union, List
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
import torch
import torch.nn as nn

from models.convolutional_vae2 import ConvolutionalVAE

def _to_sample_feature(sh_vals: Union[np.ndarray, List[np.ndarray]],
                       positive_idx: int,
                       n_samples: int,
                       n_features: int) -> np.ndarray:
    """
    Devuelve siempre un array 2-D (samples, features) con
    los SHAP-values de la clase positiva.
    """
    # a) SHAP viejo: list (outputs) de (samples, features)
    if isinstance(sh_vals, list):
        sh = sh_vals[positive_idx]
    
    # b) SHAP nuevo: ndarray (samples, features, outputs)
    elif sh_vals.ndim == 3:
        sh = sh_vals[:, :, positive_idx]
    
    # c) Caso que te ocurre: ndarray (features, outputs)
    elif sh_vals.ndim == 2 and sh_vals.shape[0] == n_features:
        sh = sh_vals[:, positive_idx].T   # → (features,)  ⇒ trasponer
        sh = np.tile(sh, (n_samples, 1))  # replicamos a todos los sujetos
        logger.warning(
            "Recibido (features, outputs). Se asumió el mismo vector "
            "para los %d sujetos.", n_samples
        )
    else:
        raise ValueError(
            f"Forma inesperada de shap_values: {sh_vals.shape}"
        )
    
    # Asegurarse de que ahora es (samples, features)
    if sh.shape[0] == n_features and sh.shape[1] == n_samples:
        sh = sh.T  # estaba transpuesto
    
    assert sh.shape == (n_samples, n_features), \
        f"Esperaba ({n_samples},{n_features}), obtuve {sh.shape}"
    return sh


# Copia la función `apply_normalization_params` para la consistencia
def apply_normalization_params(data_tensor_subset: np.ndarray, norm_params_per_channel_list: list) -> np.ndarray:
    num_subjects, num_selected_channels, num_rois, _ = data_tensor_subset.shape
    normalized_tensor_subset = data_tensor_subset.copy()
    off_diag_mask = ~np.eye(num_rois, dtype=bool)
    for c_idx, params in enumerate(norm_params_per_channel_list):
        mode = params.get('mode', 'zscore_offdiag')
        if params.get('no_scale', False): continue
        
        current_channel_data = data_tensor_subset[:, c_idx, :, :]
        scaled_channel_data = current_channel_data.copy()
        if off_diag_mask.any():
            if mode == 'zscore_offdiag':
                if params['std'] > 1e-9:
                    scaled_channel_data[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - params['mean']) / params['std']
            elif mode == 'minmax_offdiag':
                range_val = params.get('max', 1.0) - params.get('min', 0.0)
                if range_val > 1e-9:
                    scaled_channel_data[:, off_diag_mask] = (current_channel_data[:, off_diag_mask] - params['min']) / range_val
                else:
                    scaled_channel_data[:, off_diag_mask] = 0.0
        normalized_tensor_subset[:, c_idx, :, :] = scaled_channel_data
    return normalized_tensor_subset

# --- Configuración del Logger ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 🔽🔽🔽 BLOQUE AÑADIDO PARA SILENCIAR SHAP 🔽🔽🔽
# Se establece el nivel de logging de la librería SHAP a WARNING para ocultar los mensajes INFO.
logging.getLogger("shap").setLevel(logging.WARNING)
# 🔼🔼🔼 FIN DEL BLOQUE AÑADIDO 🔼🔼🔼

def run_shap_analysis(args):
    fold_dir = Path(args.run_dir) / f"fold_{args.fold_to_analyze}"
    output_dir = fold_dir / "interpretability_shap"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"--- Iniciando Análisis SHAP para Fold {args.fold_to_analyze}, Clasificador: {args.classifier_type} ---")
    logger.info(f"Directorio de resultados del fold: {fold_dir}")

    try:
        logger.info("Cargando artefactos del entrenamiento...")
        pipeline_path = fold_dir / f"classifier_{args.classifier_type}_pipeline_fold_{args.fold_to_analyze}.joblib"
        pipeline = joblib.load(pipeline_path)
        
        global_tensor_data = np.load(args.global_tensor_path)['global_tensor_data']
        metadata_df_full = pd.read_csv(args.metadata_path)
        metadata_df_full['SubjectID'] = metadata_df_full['SubjectID'].astype(str).str.strip()
        
        test_indices_in_cn_ad_df = np.load(fold_dir / "test_indices.npy")
        vae_norm_params = joblib.load(fold_dir / "vae_norm_params.joblib")
        vae_model_path = fold_dir / f"vae_model_fold_{args.fold_to_analyze}.pt"

        logger.info("Reconstruyendo el conjunto de datos de test...")
        tensor_df = pd.DataFrame({'SubjectID': np.load(args.global_tensor_path)['subject_ids'].astype(str)})
        tensor_df['tensor_idx'] = np.arange(len(tensor_df))
        merged_df = pd.merge(tensor_df, metadata_df_full, on='SubjectID', how='left')
        cn_ad_df = merged_df[merged_df['ResearchGroup_Mapped'].isin(['CN', 'AD'])].copy()
        cn_ad_df.reset_index(drop=True, inplace=True)

        test_subjects_df = cn_ad_df.iloc[test_indices_in_cn_ad_df]
        global_test_indices = test_subjects_df['tensor_idx'].values

        tensor_test_subset = global_tensor_data[global_test_indices][:, args.channels_to_use, :, :]
        tensor_test_normalized = apply_normalization_params(tensor_test_subset, vae_norm_params)
        tensor_test_torch = torch.from_numpy(tensor_test_normalized).float()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 🔽🔽🔽 BLOQUE MODIFICADO 🔽🔽🔽
        # Instanciar el VAE usando TODOS los argumentos relevantes del entrenamiento
        vae_model = ConvolutionalVAE(
            input_channels=len(args.channels_to_use), 
            latent_dim=args.latent_dim,
            image_size=global_tensor_data.shape[-1],
            dropout_rate=args.dropout_rate_vae,
            use_layernorm_fc=args.use_layernorm_vae_fc,
            num_conv_layers_encoder=args.num_conv_layers_encoder,
            decoder_type=args.decoder_type,
            intermediate_fc_dim_config=args.intermediate_fc_dim_vae,
            final_activation=args.vae_final_activation
        ).to(device)
        # 🔼🔼🔼 FIN BLOQUE MODIFICADO 🔼🔼🔼

        # Cargar el estado del modelo guardado
        vae_model.load_state_dict(torch.load(vae_model_path, map_location=device, weights_only=True))
        vae_model.eval()
        logger.info("Modelo VAE cargado y preparado para inferencia.")
        # d) Obtener las características latentes del conjunto de test
        logger.info("Generando características latentes para el conjunto de test...")
        if not hasattr(vae_model, 'encode') or not hasattr(vae_model, 'decode'):
            logger.critical("El modelo VAE cargado no tiene los métodos 'encode' o 'decode'. Verifica la definición del modelo.")
            return

        with torch.no_grad():
            _, mu_test, _, z_test = vae_model(tensor_test_torch.to(device))
        
        latent_features_np = mu_test.cpu().numpy() if args.latent_features_type == 'mu' else z_test.cpu().numpy()
        latent_feature_names = [f'latent_{i}' for i in range(latent_features_np.shape[1])]
        X_latent_test = pd.DataFrame(latent_features_np, columns=latent_feature_names)

        # e) Añadir las características de metadatos
        metadata_test = test_subjects_df[args.metadata_features].copy()
        if 'Sex' in metadata_test.columns:
            metadata_test['Sex'] = metadata_test['Sex'].map({'M': 0, 'F': 1, 'f': 1, 'm': 0})
        
        # Manejar NaNs con la media del dataset completo (simplificación, idealmente se guardaría la media del train)
        for col in metadata_test.columns:
            if metadata_test[col].isnull().any():
                fill_value = metadata_test[col].mean() if pd.api.types.is_numeric_dtype(metadata_test[col]) else metadata_test[col].mode()[0]
                metadata_test[col].fillna(fill_value, inplace=True)

        X_latent_test.reset_index(drop=True, inplace=True)
        metadata_test.reset_index(drop=True, inplace=True)
        X_test_df = pd.concat([X_latent_test, metadata_test], axis=1)
        
        logger.info(f"Datos de test reconstruidos. Forma final: {X_test_df.shape}")

    except Exception as e:
        logger.critical(f"Error CRÍTICO durante la carga o reconstrucción de datos: {e}. Abortando.", exc_info=True)
        return
        
    # --- 2. Preparar el Explicador de SHAP ---
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps.get('scaler', 'passthrough')
    
    if hasattr(preprocessor, 'transform'):
        X_test_processed = preprocessor.transform(X_test_df)
    else:
        X_test_processed = X_test_df.values

    X_test_processed_df = pd.DataFrame(X_test_processed, columns=X_test_df.columns)

    logger.info("Creando el explicador de SHAP...")
    
    # 🔽🔽🔽 BLOQUE MODIFICADO Y CORREGIDO 🔽🔽🔽
    try:
        base_model = model.base_estimator if hasattr(model, "base_estimator") else model
        positive_class_index = list(base_model.classes_).index(1)

    except (AttributeError, IndexError):
        logger.warning("No se pudo determinar el índice de la clase positiva (AD=1). Asumiendo que es 1.")
        positive_class_index = 1
        
    if args.classifier_type in ['xgb', 'gb', 'rf']:
        explainer = shap.TreeExplainer(model)
        shap_values_all_classes = explainer.shap_values(X_test_processed_df)
    else:
        logger.warning("Usando KernelExplainer de SHAP. Puede ser lento.")
        predict_proba_fn = lambda x: model.predict_proba(x)
        background_data = shap.sample(X_test_processed_df, 50, random_state=args.seed)
        explainer = shap.KernelExplainer(predict_proba_fn, background_data)
        shap_values_all_classes = explainer.shap_values(X_test_processed_df, nsamples=100)

    # ─── DEBUG: inspeccionar qué devuelve SHAP ──────────────────────────────
    def _log_shapes(obj, name="obj"):
        if isinstance(obj, list):
            logger.info(f"{name} es list   len={len(obj)}")
            for i, arr in enumerate(obj):
                logger.info(f"  {name}[{i}].shape = {arr.shape}")
        else:
            logger.info(f"{name}.shape = {obj.shape}")

    _log_shapes(shap_values_all_classes, "shap_values_all_classes")
    # ────────────────────────────────────────────────────────────────────────
    n_samples  = X_test_processed_df.shape[0]
    n_features = X_test_processed_df.shape[1]
    shap_values = _to_sample_feature(
        shap_values_all_classes,
        positive_class_index,
        n_samples,
        n_features,
    )
    logger.info("SHAP final -> shape %s", shap_values.shape)
    
    # Seleccionar los SHAP values para la clase positiva (AD)
    #shap_values = shap_values_all_classes[positive_class_index] if isinstance(shap_values_all_classes, list) else shap_values_all_classes

    # Crear el objeto Explanation unificado
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[positive_class_index]
        
    shap_explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=X_test_processed_df.values,
        feature_names=X_test_df.columns
    )
    # 🔼🔼🔼 FIN DEL BLOQUE MODIFICADO 🔼🔼🔼



    # --- 3. Generar y guardar gráficas ---

    # a) Global bar plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test_processed_df,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title(f"Importancia Global (SHAP) - Fold {args.fold_to_analyze} - {args.classifier_type.upper()}")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_global_importance_bar.png", dpi=150)
    plt.close()

    # b) Beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test_processed_df,
        show=False,
        max_display=20,
    )
    plt.title(f"Impacto de Features (SHAP) - Fold {args.fold_to_analyze} - {args.classifier_type.upper()}")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_beeswarm.png", dpi=150)
    plt.close()

    # c) Waterfall del sujeto 0
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap_explanation[0], max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_waterfall_subject_0.png", dpi=150)
    plt.close()

    
    # d) Guardar los valores SHAP para análisis futuros
    shap_artefacts = {
        "shap_values": shap_values,
        "X_test": X_test_df,
        "feature_names": shap_explanation.feature_names,
        "base_value": shap_explanation.base_values,
    }
    joblib.dump(shap_artefacts, output_dir / "shap_values_and_data.joblib")
    logger.info("Artefactos SHAP (valores, datos) guardados.")

    logger.info(f"Análisis SHAP completado. Resultados guardados en: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpretabilidad del clasificador usando SHAP.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # --- 🔽🔽🔽 SECCIÓN MODIFICADA 🔽🔽🔽 ---

    # Grupo para localizar el modelo y los datos
    group_loc = parser.add_argument_group('Localización del Modelo y Datos')
    group_loc.add_argument("--run_dir", type=str, required=True, help="Directorio raíz de los resultados del experimento.")
    group_loc.add_argument("--classifier_type", type=str, required=True, choices=['xgb', 'svm', 'logreg', 'gb', 'rf'], help="Tipo de clasificador a analizar.")
    group_loc.add_argument("--fold_to_analyze", type=int, default=1, help="Número del fold a analizar.")
    group_loc.add_argument("--global_tensor_path", type=str, required=True, help="Ruta al archivo .npz del tensor global.")
    group_loc.add_argument("--metadata_path", type=str, required=True, help="Ruta al archivo CSV de metadatos.")

    # Grupo para definir la arquitectura (debe coincidir con el entrenamiento)
    group_arch = parser.add_argument_group('Parámetros de Arquitectura (deben coincidir con el entrenamiento)')
    group_arch.add_argument("--channels_to_use", type=int, nargs='*', required=True, help="Índices de canales usados en el entrenamiento.")
    group_arch.add_argument("--latent_dim", type=int, required=True, help="Dimensión del espacio latente del VAE.")
    group_arch.add_argument("--metadata_features", nargs="*", default=None, help="Columnas de metadatos añadidas como features.")
    group_arch.add_argument("--latent_features_type", type=str, default="mu", choices=["mu", "z"], help="Tipo de características latentes usadas.")
    
    # --- Argumentos AÑADIDOS para la nueva clase VAE ---
    group_arch.add_argument("--num_conv_layers_encoder", type=int, default=4, help="Capas convolucionales en encoder VAE.")
    group_arch.add_argument("--decoder_type", type=str, default="convtranspose", help="Tipo de decoder para VAE.")
    group_arch.add_argument("--dropout_rate_vae", type=float, default=0.2, help="Tasa de dropout en VAE.")
    group_arch.add_argument("--use_layernorm_vae_fc", action='store_true', help="Usar LayerNorm en capas FC del VAE.")
    group_arch.add_argument("--intermediate_fc_dim_vae", type=str, default="quarter", help="Dimensión FC intermedia en VAE.")
    group_arch.add_argument("--vae_final_activation", type=str, default="tanh", help="Activación final del decoder VAE.")
    # --- Fin de los argumentos añadidos ---

    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad de muestreo SHAP.")

    # --- 🔼🔼🔼 FIN DE LA SECCIÓN MODIFICADA 🔼🔼🔼 ---

    args = parser.parse_args()
    run_shap_analysis(args)