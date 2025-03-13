import numpy as np
import rasterio
import dask.array as da
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from numba import njit

# Imposta il numero massimo di core utilizzabili
NUM_CORES = 40

def load_raster_data(file_path):
    """Carica i dati raster in parallelo con Dask."""
    with rasterio.open(file_path) as src:
        data = da.from_array(src.read(), chunks=(1, 512, 512))  # Chunks per elaborazione parallela
    return data

@njit
def flatten_raster(obj2clust):
    """Trasforma il raster in un array 2D velocemente con numba."""
    return obj2clust.reshape(obj2clust.shape[0], -1).T

def prepare_data_for_clustering(obj2clust, standardise_vars=True):
    """Prepara i dati raster per il clustering."""
    obj2clust = load_raster_data(obj2clust)
    data = flatten_raster(obj2clust.compute())  # Converti da Dask a NumPy
    mask = ~np.isnan(data).any(axis=1)  # Filtra i NaN
    data = data[mask]

    if standardise_vars:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    return data

def kmeans_clustering(data, k, max_iter=100):
    """Esegue K-Means e restituisce la somma dei quadrati delle distanze (WSS)."""
    kmeans = MiniBatchKMeans(n_clusters=k, max_iter=max_iter, batch_size=2048, random_state=42)
    kmeans.fit(data)
    return kmeans.inertia_

def clustering_optimization(obj2clust, num_clstrs=np.arange(5, 55, 5), standardise_vars=True):
    """Trova il numero ottimale di cluster con parallelizzazione."""
    data = prepare_data_for_clustering(obj2clust, standardise_vars)

    print("Performing K-means clustering optimization...")
    wss = Parallel(n_jobs=NUM_CORES, backend="loky")(delayed(kmeans_clustering)(data, k) for k in num_clstrs)

    # Calcolo della differenza relativa per trovare il ginocchio del grafico
    diffs = np.diff(wss)
    elbow_index = np.argmin(np.abs(diffs)) + 1
    optimal_k = num_clstrs[elbow_index]

    print(f"Optimal number of clusters found: {optimal_k}")

    # Calcolo del Compactness Index (CI)
    ci = wss[elbow_index] / wss[0]
    print(f"Compactness Index (CI): {ci:.4f}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(num_clstrs, wss, marker='o', linestyle='-')
    plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k: {optimal_k}')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Total within-cluster sum of squares")
    plt.title("Elbow Method for Optimal Clusters")
    plt.legend()
    plt.grid()
    plt.show()

    return optimal_k, ci

if __name__ == "__main__":
    input_raster = "/home/gianofe/Documents/output/no_col.tif"

    print("Running clustering optimization...")
    optimal_k, ci = clustering_optimization(input_raster, num_clstrs=np.arange(5, 55, 5))
    print("Optimization completed!")
