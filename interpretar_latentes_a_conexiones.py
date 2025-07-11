import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging
import torch
from tqdm import tqdm

# Importar la definición del modelo VAE desde tu script de modelos
from models.convolutional_vae2 import ConvolutionalVAE

# --- Configuración del Logger ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def generate_saliency_map(vae_model, shap_explanation, device, top_k_features=50):
    """
    Propaga la importancia de SHAP a través del decoder del VAE para generar un mapa de saliencia.
    """
    logger.info(f"Generando mapa de saliencia usando las top {top_k_features} características latentes.")
    
    # 1. Calcular la importancia media de cada característica latente
    mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)
    
    # Crear un DataFrame para manejar los nombres y valores
    feature_importance_df = pd.DataFrame({
        'feature': shap_explanation.feature_names,
        'importance': mean_abs_shap
    })
    
    # Filtrar solo las características latentes
    latent_features_df = feature_importance_df[feature_importance_df['feature'].str.startswith('latent_')].copy()
    latent_features_df['latent_idx'] = latent_features_df['feature'].str.replace('latent_', '').astype(int)
    latent_features_df = latent_features_df.sort_values(by='importance', ascending=False)
    
    # Normalizar la importancia para que sume 1 (usarla como peso)
    latent_features_df['weight'] = latent_features_df['importance'] / latent_features_df['importance'].sum()
    
    logger.info("Top 10 características latentes más importantes:")
    print(latent_features_df.head(10).to_string())

    # 2. Propagar a través del decoder
    latent_dim = vae_model.latent_dim
    saliency_map = None
    
    with torch.no_grad():
        # Obtener una reconstrucción de referencia desde un vector latente de ceros
        baseline_recon = vae_model.decode(torch.zeros(1, latent_dim).to(device)).squeeze().cpu().numpy()
        
        # Iterar solo sobre las características más importantes para ser eficientes
        for _, row in tqdm(latent_features_df.head(top_k_features).iterrows(), total=top_k_features, desc="Propagando features"):
            k = row['latent_idx']
            weight = row['weight']
            
            # Crear un vector latente con +1 en la dimensión de interés
            z_positive = torch.zeros(1, latent_dim).to(device)
            z_positive[0, k] = 1.0 # Simula una desviación estándar positiva
            
            recon_positive_delta = (vae_model.decode(z_positive).squeeze().cpu().numpy() - baseline_recon)
            
            if saliency_map is None:
                saliency_map = np.zeros_like(recon_positive_delta)
            
            # Acumular el cambio absoluto ponderado por la importancia de la feature
            saliency_map += weight * np.abs(recon_positive_delta)
            
    return saliency_map, latent_features_df

def main(args):
    fold_dir = Path(args.run_dir) / f"fold_{args.fold_to_analyze}"
    shap_dir = fold_dir / "interpretability_shap"
    output_dir = fold_dir / "interpretability_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"--- Iniciando Análisis de Saliencia VAE para Fold {args.fold_to_analyze} ---")

    try:
        # --- 1. Cargar Artefactos ---
        logger.info("Cargando artefactos de SHAP y VAE...")
        #shap_artefacts = joblib.load(shap_dir / "shap_explanation_object.joblib")
        shap_artefacts = joblib.load(shap_dir / "shap_values_and_data.joblib")
        #shap_explanation = shap_artefacts['shap_explanation']
        import shap
        shap_explanation = shap.Explanation(
            values       = shap_artefacts["shap_values"],
            base_values  = shap_artefacts["base_value"],
            data         = shap_artefacts["X_test"].values,
            feature_names= shap_artefacts["feature_names"],
        )
        
        roi_names = np.load(args.roi_order_path, allow_pickle=True).tolist()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        vae_model = ConvolutionalVAE(
            input_channels=len(args.channels_to_use), latent_dim=args.latent_dim,
            image_size=len(roi_names), dropout_rate=args.dropout_rate_vae,
            use_layernorm_fc=args.use_layernorm_vae_fc, num_conv_layers_encoder=args.num_conv_layers_encoder,
            decoder_type=args.decoder_type, intermediate_fc_dim_config=args.intermediate_fc_dim_vae,
            final_activation=args.vae_final_activation
        ).to(device)
        vae_model.load_state_dict(torch.load(fold_dir / f"vae_model_fold_{args.fold_to_analyze}.pt", map_location=device))
        vae_model.eval()

    except FileNotFoundError as e:
        logger.critical(f"Error: No se encontró un archivo necesario: {e}. Asegúrate de haber ejecutado el script SHAP primero.")
        return

    # --- 2. Generar Mapa de Saliencia y Ranking de Nodos ---
    saliency_map, latent_importance_df = generate_saliency_map(vae_model, shap_explanation, device)
    
    # Guardar ranking de importancia de features latentes
    latent_importance_df.to_csv(output_dir / "ranking_features_latentes.csv", index=False)
    logger.info(f"Ranking de features latentes guardado en: {output_dir / 'ranking_features_latentes.csv'}")

    # --- 3. Generar Ranking de Conexiones (Edges) ---
    logger.info("Generando ranking de conexiones más importantes...")
    upper_triangle_indices = np.triu_indices(len(roi_names), k=1)
    
    edge_importance_df = pd.DataFrame({
        'ROI_i_idx': upper_triangle_indices[0],
        'ROI_j_idx': upper_triangle_indices[1],
        'ROI_i_name': [roi_names[i] for i in upper_triangle_indices[0]],
        'ROI_j_name': [roi_names[j] for j in upper_triangle_indices[1]],
        'Saliency_Score': np.mean(saliency_map, axis=0)[upper_triangle_indices] # Promediar saliencia entre canales
    }).sort_values(by='Saliency_Score', ascending=False)
    
    edge_csv_path = output_dir / "ranking_conexiones_importantes.csv"
    edge_importance_df.to_csv(edge_csv_path, index=False)
    logger.info(f"Ranking de conexiones importantes guardado en: {edge_csv_path}")
    logger.info("Top 20 conexiones más importantes según el modelo:")
    print(edge_importance_df.head(20).to_string())

    # --- 4. Visualizar el Mapa de Saliencia Final ---
    plt.figure(figsize=(10, 8))
    final_saliency_map = np.mean(saliency_map, axis=0) # Promediar entre canales para una vista única
    sns.heatmap(final_saliency_map, cmap='hot', cbar_kws={'label': 'Importancia Acumulada Ponderada'})
    plt.title(f'Mapa de Saliencia Agregado (Fold {args.fold_to_analyze}, {args.classifier_type.upper()})')
    plt.xlabel("ROIs")
    plt.ylabel("ROIs")
    plt.tight_layout()
    plt.savefig(output_dir / "mapa_saliencia_agregado.png", dpi=150)
    plt.close()
    
    logger.info(f"Análisis de interpretabilidad completado. Resultados en: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpretabilidad VAE-Clasificador: de SHAP a Conexiones.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Reutilizamos los mismos argumentos que el script de SHAP para consistencia
    group_loc = parser.add_argument_group('Localización del Modelo y Datos')
    group_loc.add_argument("--run_dir", type=str, required=True)
    group_loc.add_argument("--classifier_type", type=str, required=True, choices=['xgb', 'svm', 'logreg', 'gb', 'rf'])
    group_loc.add_argument("--fold_to_analyze", type=int, default=1)
    group_loc.add_argument("--roi_order_path", type=str, required=True, help="Ruta al archivo .npy con los nombres de las ROIs.")
    
    group_arch = parser.add_argument_group('Parámetros de Arquitectura (deben coincidir con el entrenamiento)')
    group_arch.add_argument("--channels_to_use", type=int, nargs='*', required=True)
    group_arch.add_argument("--latent_dim", type=int, required=True)
    group_arch.add_argument("--metadata_features", nargs="*", default=None)
    group_arch.add_argument("--latent_features_type", type=str, default="mu", choices=["mu", "z"])
    group_arch.add_argument("--num_conv_layers_encoder", type=int, default=4)
    group_arch.add_argument("--decoder_type", type=str, default="convtranspose")
    group_arch.add_argument("--dropout_rate_vae", type=float, default=0.2)
    group_arch.add_argument("--use_layernorm_vae_fc", action='store_true')
    group_arch.add_argument("--intermediate_fc_dim_vae", type=str, default="quarter")
    group_arch.add_argument("--vae_final_activation", type=str, default="tanh")
    
    args = parser.parse_args()
    main(args)