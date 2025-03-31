import numpy as np
import rasterio
import pandas as pd
import os
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numba import njit
import warnings
from rasterio.warp import reproject, Resampling

# 🔧 CONFIGURAZIONI GENERALI
CLUSTER_MODE = "landcover_eft"  # Opzioni: "global_clustering", "landcover_eft", "combined"
NUM_CORES = 4  # Numero di core da usare (futuro supporto parallelismo)
NUM_CLUSTERS = 15  # Numero di cluster per il clustering globale
OUTPUT_DIR = "/scratch/gianofe/SumNDVI_correction_rev4/outputs/"  # Directory per salvare i raster generati

# Percorsi ai raster NDVI e Land Cover
NDVI_RASTER = "/scratch/gianofe/SumNDVI_correction_rev4/outputs/outputs/no_col_2008_2023.tif"
LANDCOVER_RASTER = "/home/gianofe/Desktop/Documents/corrected/LC1000.tif"


def save_raster(data_array, profile, output_filename, output_dir=OUTPUT_DIR):
    """Salva un array come raster GeoTIFF."""
    os.makedirs(output_dir, exist_ok=True)  # Crea la cartella se non esiste
    full_path = os.path.join(output_dir, output_filename)

    profile.update(
        dtype=rasterio.uint16,  # Usiamo uint16 come nel codice fornito
        count=1,
        nodata=255  # Imposta nodata a 255
    )

    with rasterio.open(full_path, 'w', **profile) as dst:
        dst.write(data_array.astype(np.uint16), 1)

    print(f"💾 Raster salvato in {full_path}")

    # Verifica che il raster sia stato salvato correttamente
    with rasterio.open(full_path) as src:
        stats = {
            "min": np.min(src.read(1)),
            "max": np.max(src.read(1)),
            "mean": np.mean(src.read(1)),
            "std": np.std(src.read(1)),
            "nodata_count": np.sum(src.read(1) == 255)
        }
    print(f"Statistiche del raster salvato: {stats}")


def load_raster(file_path):
    """Carica un raster."""
    with rasterio.open(file_path) as src:
        # Controlla se il raster ha valori validi
        nodata = src.nodata
        data = src.read(1)
        valid_pixels = np.sum(data != nodata if nodata is not None else True)
        print(f"Raster {file_path}: {valid_pixels} pixel validi su {data.size} totali")
        profile = src.profile
    return data, profile


def verify_spatial_consistency(raster1_path, raster2_path):
    """Verifica che due raster abbiano la stessa proiezione, trasformazione e dimensione."""
    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
        consistent = True

        if src1.crs != src2.crs:
            print(f"⚠️ Diversi sistemi di coordinate: {src1.crs} vs {src2.crs}")
            consistent = False

        if src1.transform != src2.transform:
            print(f"⚠️ Diverse trasformazioni affini: {src1.transform} vs {src2.transform}")
            consistent = False

        if src1.shape != src2.shape:
            print(f"⚠️ Diverse dimensioni: {src1.shape} vs {src2.shape}")
            consistent = False

        if not consistent:
            print("⚠️ I raster non sono spazialmente coerenti! I risultati potrebbero essere imprecisi.")

        return consistent


@njit
def flatten_raster(data):
    """Trasforma il raster in un array 1D velocemente con numba."""
    return data.flatten()


def prepare_data(ndvi_raster, landuse_raster):
    """Prepara i dati raster combinando NDVI e Land Cover."""
    print(f"🔍 Verifica della coerenza spaziale tra i raster...")
    spatial_consistent = verify_spatial_consistency(ndvi_raster, landuse_raster)

    ndvi_data, profile = load_raster(ndvi_raster)
    landuse_data, landuse_profile = load_raster(landuse_raster)

    # Se i raster non sono spazialmente coerenti, riproietta il land cover
    if not spatial_consistent:
        print("🔄 Riproiezione del raster land cover per allinearlo con NDVI...")
        with rasterio.open(ndvi_raster) as src_ndvi, rasterio.open(landuse_raster) as src_landuse:
            landuse_reprojected = np.zeros((src_ndvi.height, src_ndvi.width), dtype=np.uint8)
            reproject(
                source=src_landuse.read(1),
                destination=landuse_reprojected,
                src_transform=src_landuse.transform,
                src_crs=src_landuse.crs,
                dst_transform=src_ndvi.transform,
                dst_crs=src_ndvi.crs,
                resampling=Resampling.nearest
            )
            landuse_data = landuse_reprojected

    # Appiattisce gli array
    print("🔄 Preparazione dei dati per il clustering...")
    ndvi_array = flatten_raster(ndvi_data)
    landuse_array = flatten_raster(landuse_data)

    # Verifica dimensioni uguali
    min_length = min(len(ndvi_array), len(landuse_array))
    if len(ndvi_array) != len(landuse_array):
        print(
            f"⚠️ Dimensioni diverse: NDVI {len(ndvi_array)}, Land Cover {len(landuse_array)}. Troncamento a {min_length}.")
        ndvi_array = ndvi_array[:min_length]
        landuse_array = landuse_array[:min_length]

    # Gestione dei valori NaN e no data
    ndvi_min = np.nanmin(ndvi_array)
    landuse_min = np.nanmin(landuse_array)

    # Sostituisci NaN con valori minimi (evita di usare 0 che potrebbe essere un valore valido)
    nan_mask_ndvi = np.isnan(ndvi_array)
    nan_mask_landuse = np.isnan(landuse_array)

    print(f"🔍 Pixel NaN in NDVI: {np.sum(nan_mask_ndvi)}, in Land Cover: {np.sum(nan_mask_landuse)}")

    ndvi_array = np.nan_to_num(ndvi_array, nan=ndvi_min if not np.isnan(ndvi_min) else 0)
    landuse_array = np.nan_to_num(landuse_array, nan=landuse_min if not np.isnan(landuse_min) else 0)

    print(f"✅ NDVI e Land Cover allineati con {min_length} pixel validi.")

    # Crea un DataFrame con i dati
    df = pd.DataFrame({"NDVI": ndvi_array, "LandCover": landuse_array})

    # Rimuove pixel vuoti (dove NDVI è 0 o valore minimo)
    non_zero_mask = df["NDVI"] != (0 if np.isnan(ndvi_min) else ndvi_min)
    df = df[non_zero_mask]

    print(f"📊 DataFrame finale: {len(df)} righe")

    return df, profile


def evaluate_clustering(data, kmeans_model):
    """Valuta la qualità del clustering usando la varianza spiegata."""
    total_ss = np.sum((data - np.mean(data, axis=0)) ** 2)
    within_ss = kmeans_model.inertia_
    between_ss = total_ss - within_ss
    clust_eval = 100 * between_ss / total_ss
    return clust_eval


def perform_global_clustering(df, profile, filename="EFTs_clusters.tif"):
    """Esegue il clustering su tutto il raster."""
    print(f"🚀 Avvio clustering globale con {NUM_CLUSTERS} cluster...")

    # Normalizzazione dei dati
    scaler = StandardScaler()
    data = scaler.fit_transform(df["NDVI"].values.reshape(-1, 1))

    # Verifica che ci siano abbastanza dati per il clustering
    if len(data) < NUM_CLUSTERS:
        warnings.warn(f"Pochi dati ({len(data)}) rispetto al numero di cluster ({NUM_CLUSTERS}). Riduzione cluster.")
        n_clusters = max(2, len(data) // 100)
    else:
        n_clusters = NUM_CLUSTERS

    print(f"🔢 Utilizzando {n_clusters} cluster per {len(data)} pixel")

    # Esecuzione clustering
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        max_iter=500,
        batch_size=min(2048, len(data)),  # Evita batch più grandi del dataset
        random_state=42,
        n_init='auto'
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["EFT_Cluster"] = kmeans.fit_predict(data)

    # Valuta la qualità del clustering
    clust_eval = evaluate_clustering(data, kmeans)
    print(f"📊 CI (varianza spiegata): {clust_eval:.2f}%")

    # Creazione raster di output
    print("🎨 Creazione raster di output...")
    cluster_raster = np.full((profile['height'] * profile['width']), 255, dtype=np.uint16)
    cluster_raster[:len(df)] = df["EFT_Cluster"].values
    cluster_raster = cluster_raster.reshape(profile['height'], profile['width'])

    save_raster(cluster_raster, profile, filename)

    return df


def perform_landcover_clustering(df, profile, filename="EFTs_landcover_2004_2019.tif"):
    """Esegue il clustering EFT dentro ogni classe del Land Cover."""
    print("🌍 Avvio clustering per classi di Land Cover...")

    # Inizializza la colonna EFT_Cluster
    df["EFT_Cluster"] = -1

    # Identifica classi land cover uniche (escludi nodata)
    unique_land_classes = df["LandCover"].unique()
    unique_land_classes = unique_land_classes[unique_land_classes != 255]

    print(f"🔍 Trovate {len(unique_land_classes)} classi di Land Cover: {unique_land_classes}")

    # Per ogni classe land cover, esegui clustering
    total_failed = 0
    for land_class in unique_land_classes:
        print(f"🔹 Clustering per classe Land Cover: {land_class}")
        subset = df[df["LandCover"] == land_class].copy()

        if subset.empty:
            print(f"⚠️ Nessun dato per Land Cover {land_class}, saltato.")
            total_failed += 1
            continue

        # Normalizzazione
        scaler = StandardScaler()
        subset["NDVI_scaled"] = scaler.fit_transform(subset["NDVI"].values.reshape(-1, 1))

        # Determina numero ottimale di cluster
        num_clusters = max(1, min(NUM_CLUSTERS, len(subset) // 5000))
        print(f"   🔢 Utilizzando {num_clusters} cluster per {len(subset)} pixel")

        # Esegui clustering
        try:
            kmeans = MiniBatchKMeans(
                n_clusters=num_clusters,
                max_iter=300,
                batch_size=min(2048, len(subset)),
                random_state=42,
                n_init='auto'
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df.loc[subset.index, "EFT_Cluster"] = kmeans.fit_predict(subset[["NDVI_scaled"]])

            print(f"   ✅ Clustering completato per classe {land_class}")
        except Exception as e:
            print(f"   ❌ Errore nel clustering per classe {land_class}: {e}")
            total_failed += 1

    if total_failed > 0:
        print(f"⚠️ Clustering fallito per {total_failed}/{len(unique_land_classes)} classi di Land Cover")

    # Creazione raster di output
    print("🎨 Creazione raster di output...")
    cluster_raster = np.full((profile['height'] * profile['width']), 255, dtype=np.uint16)
    cluster_raster[:len(df)] = df["EFT_Cluster"].values
    cluster_raster = cluster_raster.reshape(profile['height'], profile['width'])

    save_raster(cluster_raster, profile, filename)

    return df


def combine_landcover_eft(df, profile, filename="Combined_LC_EFT.tif"):
    """Combina le classi Land Cover con i cluster EFT."""
    print("🔄 Combinazione delle classi Land Cover con i cluster EFT...")

    combined = np.full(len(df), 255, dtype=np.uint16)
    valid_mask = (df["EFT_Cluster"] >= 0) & (df["LandCover"] != 255)
    combined[valid_mask] = df.loc[valid_mask, "LandCover"].astype(int) * 100 + df.loc[valid_mask, "EFT_Cluster"].astype(
        int)

    combined_raster = np.full((profile['height'] * profile['width']), 255, dtype=np.uint16)
    combined_raster[:len(combined)] = combined
    combined_raster = combined_raster.reshape(profile['height'], profile['width'])

    save_raster(combined_raster, profile, filename)

    # Salva anche la tabella CSV con frequenze dei codici combinati
    value_counts = pd.Series(combined[valid_mask]).value_counts().reset_index()
    value_counts.columns = ['Combined_Code', 'Pixel_Count']
    value_counts["Percent"] = 100 * value_counts["Pixel_Count"] / value_counts["Pixel_Count"].sum()

    csv_path = os.path.join(OUTPUT_DIR, filename.replace(".tif", "_legend.csv"))
    value_counts.to_csv(csv_path, index=False)
    print(f"📄 Tabella delle classi combinata salvata in: {csv_path}")

    return combined_raster


def scree_plot(df, cluster_range=range(2, 30)):
    """Genera un grafico scree/elbow per determinare il numero ottimale di cluster."""
    scaler = StandardScaler()
    data = scaler.fit_transform(df["NDVI"].values.reshape(-1, 1))
    wss = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(cluster_range, wss, 'bo-')
    plt.xlabel("Numero di cluster")
    plt.ylabel("Within-cluster sum of squares")
    plt.title("Scree Plot / Elbow Method")
    plt.grid(True)

    # Salva il grafico come immagine
    plot_path = os.path.join(OUTPUT_DIR, "scree_plot.png")
    plt.savefig(plot_path)
    print(f"📊 Scree plot salvato in: {plot_path}")

    plt.close()


if __name__ == "__main__":
    print("🚀 Avvio del processo di clustering EFT...")
    print(f"🔧 Modalità: {CLUSTER_MODE}")
    print(f"🔢 Numero massimo di cluster: {NUM_CLUSTERS}")
    print(f"📂 Raster NDVI: {NDVI_RASTER}")
    print(f"📂 Raster Land Cover: {LANDCOVER_RASTER}")
    print(f"📁 Directory output: {OUTPUT_DIR}")

    df, profile = prepare_data(NDVI_RASTER, LANDCOVER_RASTER)

    if CLUSTER_MODE == "global_clustering":
        df = perform_global_clustering(df, profile, filename="EFTs_clusters.tif")

    elif CLUSTER_MODE == "landcover_eft":
        df = perform_landcover_clustering(df, profile, filename="EFTs_landcover_2004_2019.tif")

    elif CLUSTER_MODE == "combined":
        df = perform_landcover_clustering(df, profile, filename="EFTs_landcover.tif")
        combine_landcover_eft(df, profile, filename="Combined_LC_EFT.tif")
    else:
        raise ValueError(f"Modalità non riconosciuta: {CLUSTER_MODE}")

    # Genera lo scree plot per aiutare a determinare il numero ottimale di cluster
    # scree_plot(df)  # Decommenta per generare il grafico

    print("✅ Processo completato!")
