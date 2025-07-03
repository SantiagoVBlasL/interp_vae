# -*- coding: utf-8 -*-
"""
jacobian_effective_weights_analysis.py
--------------------------------------
Calcula S(x)=J_phi(x)^T w_mu para un subconjunto de sujetos y produce:
  • Heat‑map (ROI×ROI) promedio
  • Ranking de ROIs por magnitud de pesos efectivos

Requisitos previos
-----------------
  1. Haber entrenado el VAE y el clasificador lineal (pipeline sklearn) y
     guardado los artefactos en "resultados_12_inter/fold_1/" con los
     nombres utilizados en este script.
  2. Contar con:
        - global_tensor .npz   (matriz completa)
        - csv de metadatos      (SubjectsData_...)
        - mapping AAL3→Yeo‑17   (aal3_131_to_yeo17_mapping.csv)
        - lista "roi_order_131.joblib" con los 131 nombres (una sola línea)

Uso (ejemplo)
-------------
$ python jacobian_effective_weights_analysis.py \
        --tensor /ruta/GLOBAL_TENSOR.npz \
        --metadata /ruta/SubjectsData_AAL3_procesado.csv \
        --fold-dir resultados_12_inter/fold_1 \
        --out-dir resultados_12_inter/fold_1/interpretabilidad \
        --channels 1 2 5 \
        --n-subjects 20
"""
from __future__ import annotations
import argparse, logging, time
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import joblib

# --------------------------------------------------
#  Utils (normalización VAE fuera de diagonal)
# --------------------------------------------------

def apply_normalization_params(tensor_subset: np.ndarray, params_per_channel: list[dict[str, float]]) -> np.ndarray:
    """Aplica *in‑place* la normalización guardada durante el entrenamiento del VAE.
    Solo escala los elementos fuera de la diagonal.
    """
    n_subj, n_chan, n_roi, _ = tensor_subset.shape
    out = tensor_subset.copy()
    off = ~np.eye(n_roi, dtype=bool)
    if len(params_per_channel) != n_chan:
        raise ValueError("Mismatch canales vs params de normalización")
    for c in range(n_chan):
        p = params_per_channel[c]
        if p.get("no_scale", False):
            continue
        mode = p.get("mode", "zscore_offdiag")
        if mode == "zscore_offdiag":
            mean, std = p["mean"], p["std"]
            out[:, c][:, off] = (out[:, c][:, off] - mean) / std
        elif mode == "minmax_offdiag":
            mn, mx = p["min"], p["max"]
            rng = (mx - mn) if (mx - mn) > 1e-9 else 1.0
            out[:, c][:, off] = (out[:, c][:, off] - mn) / rng
        else:
            raise ValueError(f"Modo de normalización no soportado: {mode}")
    return out

# --------------------------------------------------
#  Arg‑parser
# --------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Jacobiano + w_mu → pesos efectivos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tensor", type=str, required=True, help="Ruta al .npz con global_tensor_data & subject_ids")
    p.add_argument("--metadata", type=str, required=True, help="CSV con metadatos (debe incluir ResearchGroup_Mapped, Age, Sex)")
    p.add_argument("--fold-dir", type=str, default="./resultados_12_inter/fold_1", help="Dir con modelo VAE y pipeline del fold")
    p.add_argument("--out-dir", type=str, default="./interpretabilidad", help="Dir de salida para figuras y csv")
    p.add_argument("--channels", type=int, nargs="*", default=[1, 2, 5], help="Índices de canales a usar del tensor")
    p.add_argument("--n-subjects", type=int, default=20, help="N° de sujetos a usar [<= disponibles]")
    return p.parse_args()

# --------------------------------------------------
#  Main
# --------------------------------------------------

def main() -> None:
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Usando dispositivo: {device}")

    fold_dir = Path(args.fold_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- carga artefactos ----------
    logging.info("Cargando VAE …")
    # Si la clase ConvolutionalVAE está en un módulo, impórtala. Ajusta si tu path es otro.
    from models.convolutional_vae2 import ConvolutionalVAE

    # 1️⃣  Intentamos leer los parámetros guardados; si no existen, usamos fallback coherente
    meta_params_file = fold_dir / "vae_model_meta_params.joblib"
    if meta_params_file.exists():
        vae_params: dict = joblib.load(meta_params_file)
        logging.info("  Parámetros VAE leídos de vae_model_meta_params.joblib")
    else:
        logging.warning("  vae_model_meta_params.joblib no existe. Usando parámetros por defecto coherentes con el entrenamiento …")
        vae_params = dict(
            input_channels=len(args.channels),
            latent_dim=512,               # ← el experimento usó 512 dims
            image_size=131,
            final_activation="tanh",
            intermediate_fc_dim_config="quarter",
            dropout_rate=0.2,
            use_layernorm_fc=False,
            num_conv_layers_encoder=4,
            decoder_type="convtranspose",
            num_groups=8,
        )

    vae = ConvolutionalVAE(**vae_params).to(device)
    vae_ckpt = fold_dir / "vae_model_fold_1.pt"
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device))
    vae.eval()
    logging.info("  Checkpoint cargado ✔️")

    logging.info("Cargando pipeline de clasificador …")
    pipe = joblib.load(fold_dir / "classifier_logreg_pipeline_fold_1.joblib")
    scaler = pipe.named_steps["scaler"]
    logreg = pipe.named_steps["model"]

    latent_dim = logreg.coef_.shape[1] - 2  # restamos Age & Sex => dims latentes reales
    w_mu = torch.tensor(logreg.coef_[0][:latent_dim], device=device)
    mean = torch.tensor(scaler.mean_[:latent_dim], device=device)
    std = torch.tensor(scaler.scale_[:latent_dim], device=device)
    logging.info(f"  Dimensión latente detectada: {latent_dim}")

    logging.info("Cargando parámetros de normalización VAE …")
    norm_params = joblib.load(fold_dir / "vae_norm_params.joblib")

    logging.info("Cargando tensor global …")
    data_npz = np.load(args.tensor)
    tensor = data_npz["global_tensor_data"][:, args.channels]
    subj_ids_tensor = data_npz["subject_ids"].astype(str)

    logging.info("Cargando metadatos …")
    meta_df = pd.read_csv(args.metadata)
    meta_df["SubjectID"] = meta_df["SubjectID"].astype(str).str.strip()
    df = pd.DataFrame({"SubjectID": subj_ids_tensor, "tensor_idx": np.arange(len(subj_ids_tensor))})
    meta = df.merge(meta_df, on="SubjectID", how="left")
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    if cn_ad.empty:
        logging.error("No hay sujetos CN/AD en el tensor + metadatos. Abortando.")
        return

    # ----------- selección de sujetos -----------
    ad = cn_ad[cn_ad["ResearchGroup_Mapped"] == "AD"].head(args.n_subjects // 2)
    cn = cn_ad[cn_ad["ResearchGroup_Mapped"] == "CN"].head(args.n_subjects - len(ad))
    selected = pd.concat([ad, cn])
    logging.info(f"Sujetos seleccionados: {len(selected)} (AD={len(ad)}, CN={len(cn)})")

    tensor_norm = apply_normalization_params(tensor[selected["tensor_idx"].values], norm_params)

    # ---------- función sensibilidad ----------
    def S_one(x_np: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x_np[None]).float().to(device).requires_grad_(True)
        mu, _ = vae.encode(x)
        z = (mu - mean) / std
        score = (z * w_mu).sum()
        (grad,) = torch.autograd.grad(score, x)
        return grad.detach().cpu().numpy()[0]  # (C,131,131)

    logging.info("Computando gradientes …")
    grads = np.zeros_like(tensor_norm[: len(selected)], dtype=np.float32)
    for i in tqdm(range(len(selected)), desc="subjects"):
        grads[i] = np.abs(S_one(tensor_norm[i]))

    G_mean = grads.mean(0)  # (C,131,131)
    G_sum = G_mean.sum(0)   # (131,131) ‑ agregamos canales

    # ---------- heat‑map ----------
    plt.figure(figsize=(6, 5))
    sns.heatmap(G_sum, cmap="coolwarm", center=0, square=True, cbar_kws={"label": "|∂logit/∂x|"})
    plt.title(f"Sensibilidad media ({len(selected)} sujetos)")
    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_Jw_mean.png", dpi=300)
    plt.close()
    logging.info(f"Heat‑map guardado en {out_dir}/heatmap_Jw_mean.png")

    # ---------- ranking ROI ----------
    roi_scores = G_sum.sum(1)  # (131,)
    mapping_df = pd.read_csv("aal3_131_to_yeo17_mapping.csv")
    mapping_df["Score"] = roi_scores
    ranking_df = mapping_df.sort_values("Score", ascending=False)
    ranking_csv = out_dir / "roi_ranking_Jw_fold1.csv"
    ranking_df.to_csv(ranking_csv, index=False)
    logging.info(f"Ranking ROIs guardado en {ranking_csv}")

    print("\nTOP‑15 ROIs por |Jᵀ·w|:")
    print(ranking_df[["AAL3_Name", "Yeo17_Network", "Score"]].head(15).to_string(index=False))

    logging.info("✔️  Análisis finalizado.")

if __name__ == "__main__":
    t0 = time.time()
    main()
    logging.info(f"Tiempo total: {time.time() - t0:.1f}s")
