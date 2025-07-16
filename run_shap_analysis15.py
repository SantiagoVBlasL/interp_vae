# run_shap_analysis15.py
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

from models.convolutional_vae3 import ConvolutionalVAE

def _to_sample_feature(sh_vals: Union[np.ndarray, List[np.ndarray]],
                       positive_idx: int,
                       n_samples: int,
                       n_features: int) -> np.ndarray:
    """
    Devuelve siempre un array 2-D (samples, features) con
    los SHAP-values de la clase positiva.
    Maneja la salida de diferentes explicadores de SHAP.
    """
    # CASO 1: TreeExplainer en problema binario (el más común y eficiente)
    # Devuelve un único array (samples, features) para la clase positiva.
    if isinstance(sh_vals, np.ndarray) and sh_vals.ndim == 2 and sh_vals.shape == (n_samples, n_features):
        logger.info("Recibido array 2D (samples, features). Se asume que son los SHAP de la clase positiva.")
        return sh_vals

    # CASO 2: KernelExplainer o TreeExplainer en problema multiclase
    # Devuelve una lista de arrays, uno por clase.
    if isinstance(sh_vals, list):
        return sh_vals[positive_idx]
    
    # CASO 3: Formato más nuevo de SHAP (samples, features, classes)
    if isinstance(sh_vals, np.ndarray) and sh_vals.ndim == 3:
        return sh_vals[:, :, positive_idx]
    
    # Si no coincide con ninguno de los casos esperados, lanzar error.
    raise ValueError(
        f"Forma inesperada de shap_values: {type(sh_vals)} con forma "
        f"{sh_vals.shape if isinstance(sh_vals, np.ndarray) else [a.shape for a in sh_vals]}"
    )


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

        # Al inicio, después de cargar otros artefactos
        import json
        with open(fold_dir / "label_mapping.json", 'r') as f:
            label_info = json.load(f)
        logger.info(
            f"Mapeo de etiquetas cargado. Clase positiva='{label_info['positive_label_name']}' "
            f"(valor={label_info['positive_label_int']}). El índice real se calculará según model.classes_."
        )
        # --- CARGA BACKGROUND SHAP VERSIONADO POR CLASIFICADOR ---------------
        shap_background_path = fold_dir / f"shap_background_data_{args.classifier_type}.joblib"
        if not shap_background_path.exists():
            # retro-compatibilidad con runs viejos
            shap_background_path = fold_dir / "shap_background_data.joblib"
            logger.warning(f"No se encontró background específico '{args.classifier_type}', "
                           f"se usa fallback genérico: {shap_background_path.name}")
        shap_background_data = joblib.load(shap_background_path)
        logger.info(f"Background data para SHAP cargado. Forma: {shap_background_data.shape}")
        # ---------------------------------------------------------------------


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
            final_activation=args.vae_final_activation,
            num_groups=args.gn_num_groups
        ).to(device)
        # 🔼🔼🔼 FIN BLOQUE MODIFICADO 🔼🔼🔼

        # --- LIMPIAR PREFIJO _orig_mod. DEL STATE_DICT -------------------------
        # Cargar el diccionario de estado desde el archivo
        compiled_state_dict = torch.load(vae_model_path, map_location=device)

        # "Limpiar" las llaves quitando el prefijo "_orig_mod." que añade torch.compile
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in compiled_state_dict.items()}

        # Ahora, cargar el diccionario limpio en el modelo. Esto debería funcionar.
        vae_model.load_state_dict(clean_state_dict)

        # Poner el modelo en modo de evaluación
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

# ... (código previo para generar X_latent_test) ...

        # --- RECONSTRUCCIÓN DEL DATAFRAME FINAL (CORREGIDO) ---
        # --- 1. Preparación de datos para SHAP (Flujo Único y Correcto) ---
        logger.info("Construyendo DataFrame crudo para la transformación.")

        # Combina características latentes y metadatos crudos de test
        X_test_raw_df = pd.concat([
            X_latent_test.reset_index(drop=True),
            test_subjects_df[args.metadata_features or []].reset_index(drop=True)
        ], axis=1)

        logger.info(f"DataFrame crudo de test (antes de procesar): {X_test_raw_df.shape}")

        # Extrae el preprocesador del pipeline
        preprocessor = pipeline.named_steps['preprocessor']
        # background_df viene crudo; conviene transformarlo:
        background_processed = preprocessor.transform(background_raw)
        background_df = pd.DataFrame(background_processed,
                                    columns=preprocessor.get_feature_names_out())
        joblib.dump(background_df, fold_output_dir / f"shap_background_data_{current_classifier_type}.joblib")


        # Aplica la transformación APRENDIDA EN ENTRENAMIENTO
        X_test_processed = preprocessor.transform(X_test_raw_df)

        # Obtiene los nombres de las features procesadas (incluye dummies, etc.)
        feature_names_processed = preprocessor.get_feature_names_out()

        # Crea el DataFrame final para SHAP
        X_test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names_processed)

        logger.info(f"Datos de test procesados y listos para SHAP. Forma final: {X_test_processed_df.shape}")

        model_step = pipeline.named_steps['model']
        if hasattr(model_step, "feature_names_in_"):
            assert np.array_equal(X_test_processed_df.columns.values, model_step.feature_names_in_), \
                "Mismatch features..."
        else:
            logger.warning("El estimador del pipeline no expone feature_names_in_; se omite verificación estricta.")


        # # Assert para verificar la consistencia de las features
        # assert list(X_test_processed_df.columns) == list(pipeline.named_steps['model'].feature_names_in_), \
        #     "Las columnas del DataFrame procesado no coinciden con las esperadas por el modelo."
    

        # --- 2. Preparar el Explicador de SHAP ---
        model = pipeline.named_steps['model']
        classes_ = list(model.classes_)
        positive_label_value = label_info['positive_label_int']
        positive_class_index = classes_.index(positive_label_value)
        logger.info(f"Clases del modelo: {classes_}. Índice de la clase positiva={positive_class_index}.")

        model_for_shap = model # Por defecto, es el modelo del pipeline

        # Desenvolver el modelo base si es un CalibratedClassifierCV
        if hasattr(model, "calibrated_classifiers_") and args.classifier_type in ['xgb', 'gb', 'rf', 'lgbm']:
            # Extrae el primer estimador (asume CV interna de CalibratedClassifierCV)
            # Nota: Si usaste cv='prefit', el acceso es a .base_estimator
            if hasattr(model.calibrated_classifiers_[0], "estimator"):
                model_for_shap = model.calibrated_classifiers_[0].estimator
            else: # Para el caso de LGBM puede que no tenga .estimator
                model_for_shap = model.calibrated_classifiers_[0].base_estimator
                
            logger.info("Modelo calibrado detectado. Usando el estimador base para SHAP TreeExplainer.")

        logger.info("Creando el explicador de SHAP...")
        if args.classifier_type in ['xgb', 'gb', 'rf', 'lgbm']:
            background_df = shap_background_data
            explainer = shap.TreeExplainer(model_for_shap, background_df) # Es buena práctica pasar el background también aquí
            shap_values_all_classes = explainer.shap_values(X_test_processed_df)
        else:
            logger.warning("Usando KernelExplainer de SHAP. Puede ser lento.")
            background_df = shap_background_data  # ya está procesado
            explainer = shap.KernelExplainer(model_for_shap.predict_proba, background_df)
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
            feature_names=X_test_processed_df.columns
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
            "X_test": X_test_processed_df,
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
    group_arch.add_argument("--gn_num_groups", type=int, default=16, help="Número de grupos GroupNorm usado en el VAE.")
    # --- Fin de los argumentos añadidos ---

    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad de muestreo SHAP.")

    # --- 🔼🔼🔼 FIN DE LA SECCIÓN MODIFICADA 🔼🔼🔼 ---

    args = parser.parse_args()
    run_shap_analysis(args)