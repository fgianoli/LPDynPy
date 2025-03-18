import numpy as np
import rasterio
import dask.array as da
import pandas as pd
import os
import subprocess
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from numba import njit

# 🔹 Modalità disponibili
# "global_clustering" → Clustering su tutto il raster ignorando il Land Cover
# "landcover_eft" → Clustering EFT all'interno di ogni classe del Land Cover
CLUSTER_MODE = "landcover_eft"

NUM_CORES = 40  # Usa tutti i core disponibili
NUM_CLUSTERS = 25  # Numero fisso di cluster per global_clustering
OUTPUT_DIR = "/home/gianofe/Documents/output/"  # Directory di output

def save_raster(data_array, profile, output_filename, output_dir=OUTPUT_DIR):
    """Salva un array come raster GeoTIFF."""
    os.makedirs(output_dir, exist_ok=True)  # Crea la cartella se non esiste
    full_path = os.path.join(output_dir, output_filename)

    profile.update(
        dtype=rasterio.uint8,  # Salviamo il file come UINT8 per risparmiare spazio
        count=1,
        nodata=255  # Imposta nodata a 255
    )

    with rasterio.open(full_path, 'w', **profile) as dst:
        dst.write(data_array.astype(np.uint8), 1)

    print(f"💾 Raster salvato in {full_path}")

def load_raster(file_path):
    """Carica un raster con Dask per evitare di occupare troppa RAM."""
    with rasterio.open(file_path) as src:
        data = da.from_array(src.read(1), chunks=(512, 512))
        profile = src.profile
    return data, profile

@njit
def flatten_raster(obj2clust):
    """Trasforma il raster in un array 2D velocemente con numba."""
    return obj2clust.reshape(-1, 1)

def prepare_data(ndvi_raster, landuse_raster):
    """Prepara i dati raster combinando NDVI e Land Cover."""
    ndvi_data, profile = load_raster(ndvi_raster)
    landuse_data, _ = load_raster(landuse_raster)

    ndvi_array = flatten_raster(ndvi_data.compute())
    landuse_array = flatten_raster(landuse_data.compute())

    min_length = min(len(ndvi_array), len(landuse_array))
    ndvi_array = ndvi_array[:min_length]
    landuse_array = landuse_array[:min_length]

    ndvi_array = np.nan_to_num(ndvi_array, nan=np.nanmin(ndvi_array))
    landuse_array = np.nan_to_num(landuse_array, nan=np.nanmin(landuse_array))

    print(f"✅ NDVI e Land Cover allineati con {min_length} pixel validi.")

    df = pd.DataFrame({"NDVI": ndvi_array.flatten(), "LandCover": landuse_array.flatten()})
    df = df[df["NDVI"] != 0]  # Rimuove pixel vuoti

    return df, profile

def perform_global_clustering(df, profile, filename="EFTs_clusters.tif"):
    """Esegue il clustering su tutto il raster."""
    scaler = StandardScaler()
    data = scaler.fit_transform(df["NDVI"].values.reshape(-1, 1))

    print(f"🚀 Clustering globale con {NUM_CLUSTERS} cluster...")
    kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, max_iter=500, batch_size=2048, random_state=42, n_init='auto')
    df["EFT_Cluster"] = kmeans.fit_predict(data)

    cluster_raster = np.full((profile['height'], profile['width']), 255)
    mask = df["NDVI"] != 0
    cluster_raster.flat[np.where(mask)] = df["EFT_Cluster"]

    save_raster(cluster_raster, profile, filename)
    return cluster_raster

def perform_landcover_clustering(df, profile, filename="EFTs_landcover.tif"):
    """Esegue il clustering EFT dentro ogni classe del Land Cover."""
    df["NDVI"] = df["NDVI"].astype(np.float32)
    df["LandCover"] = df["LandCover"].astype(np.int32)
    df["EFT_Cluster"] = np.full(len(df), -1, dtype=np.int32)

    for land_class in df["LandCover"].unique():
        if land_class == 255:
            continue

        print(f"🔹 Clustering per classe Land Cover: {land_class}")
        subset = df[df["LandCover"] == land_class].copy()

        if subset.empty:
            print(f"⚠️ Nessun dato per Land Cover {land_class}, saltato.")
            continue

        subset["NDVI"] = np.nan_to_num(subset["NDVI"], nan=np.nanmin(subset["NDVI"]))
        subset["NDVI"] = np.where(np.isinf(subset["NDVI"]), np.nanmin(subset["NDVI"]), subset["NDVI"])

        scaler = StandardScaler()
        subset["NDVI"] = scaler.fit_transform(subset["NDVI"].values.reshape(-1, 1))

        num_clusters = max(1, min(NUM_CLUSTERS, len(subset) // 5000))
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, max_iter=300, batch_size=2048, random_state=42)
        df.loc[df["LandCover"] == land_class, "EFT_Cluster"] = kmeans.fit_predict(subset[["NDVI"]]).astype(np.int32)

    cluster_raster = np.full((profile['height'], profile['width']), 255, dtype=np.uint8)
    mask = df["NDVI"] != 0
    cluster_raster.flat[np.where(mask)] = df["EFT_Cluster"]

    save_raster(cluster_raster, profile, filename)
    return cluster_raster

if __name__ == "__main__":
    ndvi_raster = "/home/gianofe/Documents/output/no_col.tif"
    landuse_raster = "/home/gianofe/Desktop/Documents/corrected/LC1000.tif"

    df, profile = prepare_data(ndvi_raster, landuse_raster)

    if CLUSTER_MODE == "global_clustering":
        clusters = perform_global_clustering(df, profile, filename=os.path.join(OUTPUT_DIR, "EFTs_clusters.tif"))
    elif CLUSTER_MODE == "landcover_eft":
        clusters = perform_landcover_clustering(df, profile, filename=os.path.join(OUTPUT_DIR, "EFTs_landcover_2023.tif"))

    print("✅ Processo completato!")
