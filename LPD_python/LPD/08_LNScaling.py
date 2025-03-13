import rasterio
import numpy as np
import pandas as pd
import time
from rasterio.mask import mask
from multiprocessing import Pool, cpu_count


def read_raster(filepath):
    """Legge un raster e restituisce dati e metadati"""
    print(f"Caricamento raster: {filepath}")
    start_time = time.time()
    with rasterio.open(filepath) as src:
        data = src.read()
        meta = src.meta
    print(f"Caricamento completato in {time.time() - start_time:.2f} secondi")
    return data, meta


def compute_mean_last_n_years(data, years, start_year=2008, end_year=2023):
    """Calcola la media degli anni selezionati del raster di produttività."""
    print("Calcolo della media degli anni selezionati...")
    start_time = time.time()

    # Definire l'indice delle bande corrispondenti agli anni richiesti
    year_indices = [i for i, y in enumerate(years) if start_year <= y <= end_year]
    if not year_indices:
        raise ValueError("Nessuna banda corrisponde all'intervallo di anni richiesto.")

    selected_data = data[year_indices]

    # Identifica le bande con almeno un valore non NaN
    valid_bands = ~np.isnan(selected_data).all(axis=(1, 2))
    selected_data = selected_data[valid_bands]  # Mantieni solo le bande valide

    if selected_data.size == 0:  # Se tutte le bande erano NaN
        result = np.full(data.shape[1:], np.nan)
    else:
        result = np.nanmean(selected_data, axis=0)

    print(f"Media calcolata in {time.time() - start_time:.2f} secondi")
    return result


def compute_percentile_by_cluster(ProdVar, EFTs, percentile=90):
    """Calcola il percentile 90 della produttività per ciascun cluster EFT."""
    print("Calcolo del percentile per cluster...")
    start_time = time.time()
    df = pd.DataFrame({
        'ProductivityVariable': ProdVar.flatten(),
        'EFT': EFTs.flatten()
    }).dropna()
    percentile_values = df.groupby('EFT')['ProductivityVariable'].quantile(percentile / 100).to_dict()
    potential_prod = np.vectorize(lambda eft: percentile_values.get(eft, np.nan))(EFTs)
    print(f"Percentile calcolato in {time.time() - start_time:.2f} secondi")
    return potential_prod


def compute_local_scaled_productivity(ProdVar, potential_prod):
    """Calcola la produttività scalata localmente (LSP) come percentuale rispetto al valore potenziale."""
    print("Calcolo della Local Scaled Productivity (LSP)...")
    start_time = time.time()
    with np.errstate(divide='ignore', invalid='ignore'):
        LSP = np.where(np.isnan(ProdVar) | np.isnan(potential_prod), np.nan, (ProdVar / potential_prod) * 100)
    print(f"LSP calcolata in {time.time() - start_time:.2f} secondi")
    return LSP


def save_raster(data, meta, filename):
    """Salva un raster con i metadati aggiornati."""
    print(f"Salvataggio raster: {filename}")
    start_time = time.time()
    meta.update(dtype=rasterio.float32, count=1)
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(data.astype(np.float32), 1)
    print(f"Raster salvato in {time.time() - start_time:.2f} secondi")


def LNScaling(EFTs_path, ProdVar_path, years, filename="", cores=1):
    """Esegue il calcolo della Local Net Productivity Scaling (LNS)."""
    print("Avvio del processo LNScaling...")
    total_start_time = time.time()
    EFTs, meta = read_raster(EFTs_path)
    ProdVar, _ = read_raster(ProdVar_path)
    if EFTs.shape[1:] != ProdVar.shape[1:]:
        raise ValueError("EFTs e ProdVar devono avere la stessa estensione e risoluzione")
    ProdVar_avg = compute_mean_last_n_years(ProdVar, years, start_year=2008, end_year=2023)
    potential_prod = compute_percentile_by_cluster(ProdVar_avg, EFTs[0])
    ProdVar_avg = np.where(ProdVar_avg > potential_prod, potential_prod, ProdVar_avg)
    LSP = compute_local_scaled_productivity(ProdVar_avg, potential_prod)
    if filename:
        save_raster(LSP, meta, filename)
    print(f"Processo LNScaling completato in {time.time() - total_start_time:.2f} secondi")
    return LSP


# Esempio di utilizzo
EFTs_path = "/home/gianofe/Desktop/Documents/corrected/output/EFTs_clusters2.tif"
ProdVar_path = "/home/gianofe/Desktop/Documents/corrected/cf_multiband.tif"
years = list(range(1999, 2025))  # Definiamo gli anni corrispondenti alle bande
output_path = "/home/gianofe/Desktop/Documents/corrected/output/lns.tif"
LSP_result = LNScaling(EFTs_path, ProdVar_path, years, filename=output_path, cores=40)
