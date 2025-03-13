import numpy as np
import rasterio
import dask.array as da
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from numba import njit

# Imposta il numero massimo di core utilizzabili
NUM_CORES = 40
NUM_CLUSTERS = 35  # Numero fisso di cluster

def load_raster_data(file_path):
    """Carica i dati raster in parallelo con Dask."""
    with rasterio.open(file_path) as src:
        data = da.from_array(src.read(), chunks=(1, 512, 512))  # Ottimizzato per la memoria
        profile = src.profile
    return data, profile

@njit
def flatten_raster(obj2clust):
    """Trasforma il raster in un array 2D velocemente con numba."""
    return obj2clust.reshape(obj2clust.shape[0], -1).T

def prepare_data_for_clustering(obj2clust, standardise_vars=True):
    """Prepara i dati raster per il clustering."""
    obj2clust, profile = load_raster_data(obj2clust)

    data = flatten_raster(obj2clust.compute())  # Converti da Dask a NumPy
    mask = ~np.isnan(data).any(axis=1)  # Filtra i NaN
    data = data[mask]

    if standardise_vars:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    return data, profile, mask

def kmeans_clustering(data, n_clusters, max_iter=500):
    """Esegue K-Means in parallelo con Joblib."""
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, max_iter=max_iter, batch_size=2048, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(data)
    return labels

def perform_clustering(obj2clust, filename=""):
    """Esegue il clustering e salva il risultato come raster."""
    data, profile, mask = prepare_data_for_clustering(obj2clust)

    print(f"Eseguo il clustering con {NUM_CLUSTERS} cluster...")
    clusters = kmeans_clustering(data, NUM_CLUSTERS)

    # Ricostruisci la mappa dei cluster
    cluster_raster = np.full((profile['height'], profile['width']), np.nan)
    cluster_raster.flat[np.where(mask)] = clusters

    if filename:
        print(f"Salvataggio raster clusterizzato in {filename}")
        profile.update(dtype=rasterio.uint8, count=1, nodata=255)
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(cluster_raster.astype(np.uint8), 1)

    return cluster_raster

if __name__ == "__main__":
    input_raster = "/home/gianofe/Documents/output/no_col.tif"
    output_raster = "/home/gianofe/Documents/output/EFTs_clusters2.tif"

    print("Avvio clustering...")
    clusters = perform_clustering(input_raster, filename=output_raster)
    print("Clustering completato!")
