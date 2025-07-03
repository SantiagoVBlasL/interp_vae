import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def create_final_analysis(saliency_files: list, roi_map_path: str, roi_labels_path: str, output_dir: Path, top_k: int = 25):
    """
    Fusiona los rankings de saliencia de todos los folds, los une con la
    información anatómica y de red, y genera la tabla y visualización finales.
    """
    # 1. Cargar y consolidar los resultados de saliencia de todos los folds
    try:
        all_fold_dfs = [pd.read_csv(f) for f in saliency_files]
        merged_folds_df = pd.concat(all_fold_dfs)
    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar el fichero {e.filename}. Asegúrate de que todos los CSV de los folds están disponibles.")
        return

    # 2. Calcular estadísticas (media y std) por ROI
    stats_df = (merged_folds_df
                .groupby('ROI_Name')['Saliency_Score']
                .agg(Mean_Saliency='mean', Std_Saliency='std')
                .reset_index())
    stats_df['ROI_Name'] = stats_df['ROI_Name'].str.strip()

    # 3. Cargar los ficheros de mapeo
    yeo_map_df = pd.read_csv(roi_map_path)
    yeo_map_df['AAL3_Name'] = yeo_map_df['AAL3_Name'].str.strip()
    
    roi_labels_df = pd.read_csv(roi_labels_path, sep='\t')
    roi_labels_df['nom_c'] = roi_labels_df['nom_c'].str.strip()
    roi_labels_df['nom_l'] = roi_labels_df['nom_l'].str.strip()

    # --- CORRECCIÓN DEFINITIVA: UNIÓN EN DOS PASOS ---
    # a. Unir el ranking de saliencia (con nombres largos) con la tabla que contiene el mapeo de nombres largos a cortos.
    final_df = pd.merge(
        stats_df,
        roi_labels_df[['nom_l', 'nom_c']],
        left_on='ROI_Name',
        right_on='nom_l',
        how='left'
    )
    
    # b. Unir el resultado con el mapeo de redes, usando ahora el nombre corto como clave.
    final_df = pd.merge(
        final_df,
        yeo_map_df[['AAL3_Name', 'Yeo17_Network']],
        left_on='nom_c',
        right_on='AAL3_Name',
        how='left'
    )

    # 5. Limpiar y ordenar el DataFrame final
    final_df = final_df.drop(columns=['nom_l', 'nom_c', 'AAL3_Name'])
    final_df = final_df.sort_values(by='Mean_Saliency', ascending=False).reset_index(drop=True)

    # Guardar la tabla final y completa
    final_csv_path = output_dir / "global_robust_roi_ranking_final.csv"
    final_df[['ROI_Name', 'Yeo17_Network', 'Mean_Saliency', 'Std_Saliency']].to_csv(final_csv_path, index=False)
    
    print(f"\n--- RANKING GLOBAL Y ROBUSTO DE ROIs (PROMEDIO DE {len(saliency_files)} FOLDS) ---")
    print(f"Tabla de resultados guardada en: {final_csv_path}")
    print(final_df[['ROI_Name', 'Yeo17_Network', 'Mean_Saliency']].head(top_k))

    # 6. Generar la visualización final, ahora con la leyenda de red correcta
    top_rois = final_df.head(top_k)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.barplot(
        x='Mean_Saliency',
        y='ROI_Name',
        data=top_rois,
        hue='Yeo17_Network',
        dodge=False,
        palette='tab20',
        ax=ax
    )
    
    ax.errorbar(
        x=top_rois['Mean_Saliency'],
        y=np.arange(len(top_rois)),
        xerr=top_rois['Std_Saliency'],
        fmt='none',
        ecolor='gray',
        capsize=3,
        elinewidth=1.5,
        alpha=0.7
    )

    ax.set_title(f'Top {top_k} ROIs más Influyentes (Media ± DE sobre {len(saliency_files)} Folds)', fontsize=16)
    ax.set_xlabel('Importancia de Saliencia Global', fontsize=12)
    ax.set_ylabel('Región de Interés (AAL3)', fontsize=12)
    ax.legend(title='Red Funcional (Yeo-17)', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    final_plot_path = output_dir / "final_roi_ranking_with_errorbars.png"
    plt.savefig(final_plot_path, dpi=300)
    print(f"\nGráfico final con barras de error guardado en: {final_plot_path}")
    plt.show()

# --- Ejecución del Script ---
# Lista de los ficheros CSV de saliencia que has subido.
saliency_files = [
    '/home/diego/Escritorio/limpio/resultados_12_inter/fold_1/roi_saliency_ranking.csv',
    '/home/diego/Escritorio/limpio/resultados_12_inter/fold_2/roi_saliency_ranking.csv',
    '/home/diego/Escritorio/limpio/resultados_12_inter/fold_3/roi_saliency_ranking.csv',
    '/home/diego/Escritorio/limpio/resultados_12_inter/fold_4/roi_saliency_ranking.csv',
    '/home/diego/Escritorio/limpio/resultados_12_inter/fold_5/roi_saliency_ranking.csv'
]
roi_map_file = 'aal3_131_to_yeo17_mapping.csv'
roi_labels_file = 'ROI_MNI_V7_vol.txt'
output_directory = Path('./')

create_final_analysis(saliency_files, roi_map_file, roi_labels_file, output_directory)