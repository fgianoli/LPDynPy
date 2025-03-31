import rasterio
import numpy as np
import time


def read_raster(filepath, first_band_only=False):
    """Legge un raster e restituisce dati e metadati."""
    print(f"Caricamento raster: {filepath}")
    start_time = time.time()
    with rasterio.open(filepath) as src:
        data = src.read(1) if first_band_only else src.read(1)  # Assicuriamoci che sia 2D
        meta = src.meta
    print(f"Caricamento completato in {time.time() - start_time:.2f} secondi")
    return data, meta


def classify_productivity(LandProd_change, LandProd_current, local_prod_threshold=50):
    """Classifica la produttività della terra combinando LandProd_change e LandProd_current."""
    print("Classificazione della produttività...")
    start_time = time.time()

    LandProd_change = np.squeeze(LandProd_change)  # Rimuove dimensioni extra
    if LandProd_current is not None:
        LandProd_current = np.squeeze(LandProd_current)  # Rimuove dimensioni extra se presente

    LPD_CombAssess = np.full_like(LandProd_change, np.nan, dtype=np.float32)

    if LandProd_current is not None:
        cond1 = np.isin(LandProd_change, [1, 2, 3, 4, 5, 6, 8, 9]) & (LandProd_current < local_prod_threshold)
        cond2 = np.isin(LandProd_change, [3, 6]) & (LandProd_current >= local_prod_threshold)
        LPD_CombAssess[cond1 | cond2] = 1  # Declining land productivity

        cond3 = np.isin(LandProd_change, [7]) & (LandProd_current < local_prod_threshold)
        cond4 = np.isin(LandProd_change, [1, 2, 4, 5, 8, 9]) & (LandProd_current >= local_prod_threshold)
        LPD_CombAssess[cond3 | cond4] = 2  # Early signs of decline

        cond5 = np.isin(LandProd_change, [7]) & (LandProd_current >= local_prod_threshold)
        cond6 = np.isin(LandProd_change, [10, 11, 12])
        LPD_CombAssess[cond5 | cond6] = 3  # Negative fluctuation (stressed)

        cond7 = np.isin(LandProd_change, [13, 14, 15])
        cond8 = np.isin(LandProd_change, [16, 17, 19]) & (LandProd_current < local_prod_threshold)
        LPD_CombAssess[cond7 | cond8] = 4  # Positive fluctuation (not stressed)

        cond9 = np.isin(LandProd_change, [18, 20, 21, 22]) & (LandProd_current < local_prod_threshold)
        cond10 = np.isin(LandProd_change, [16, 17, 18, 19, 20, 21, 22]) & (LandProd_current >= local_prod_threshold)
        LPD_CombAssess[cond9 | cond10] = 5  # Increasing land productivity
    else:
        print("LandProd_current non fornito, usando solo LandProd_change")
        cond1 = np.isin(LandProd_change, [1, 2, 3, 4, 5, 6, 8, 9])
        cond2 = np.isin(LandProd_change, [7])
        cond3 = np.isin(LandProd_change, [10, 11, 12])
        cond4 = np.isin(LandProd_change, [13, 14, 15])
        cond5 = np.isin(LandProd_change, [16, 17, 18, 19, 20, 21, 22])
        LPD_CombAssess[cond1] = 1
        LPD_CombAssess[cond2] = 2
        LPD_CombAssess[cond3] = 3
        LPD_CombAssess[cond4] = 4
        LPD_CombAssess[cond5] = 5

    print(f"Classificazione completata in {time.time() - start_time:.2f} secondi")
    return LPD_CombAssess


def save_raster(data, meta, filename):
    """Salva un raster con i metadati aggiornati."""
    print(f"Salvataggio raster: {filename}")
    start_time = time.time()
    meta.update(dtype=rasterio.float32, count=1)
    with rasterio.open(filename, 'w', **meta) as dst:
        dst.write(data.astype(np.float32), 1)
    print(f"Raster salvato in {time.time() - start_time:.2f} secondi")


def LPD_CombAssess(LandProd_change_path, LandProd_current_path=None, local_prod_threshold=50, filename=""):
    """Esegue l'analisi combinata della produttività della terra."""
    print("Avvio del processo LPD_CombAssess...")
    total_start_time = time.time()

    LandProd_change, meta = read_raster(LandProd_change_path, first_band_only=True)
    LandProd_current = None
    if LandProd_current_path:
        LandProd_current, _ = read_raster(LandProd_current_path)

    LPD_result = classify_productivity(LandProd_change, LandProd_current, local_prod_threshold)

    if filename:
        save_raster(LPD_result, meta, filename)

    print(f"Processo LPD_CombAssess completato in {time.time() - total_start_time:.2f} secondi")
    return LPD_result


# Esempio di utilizzo
LandProd_change_path = "/scratch/gianofe/SumNDVI_correction_rev3/outputs/long_term_change_2004_2019.tif"
LandProd_current_path = "/scratch/gianofe/SumNDVI_correction_rev3/outputs/lns_2004_2019.tif"
output_path = "/scratch/gianofe/SumNDVI_correction_rev3/outputs/LPD_finalMap_2019.tif"

LPD_finalMap = LPD_CombAssess(LandProd_change_path, LandProd_current_path, filename=output_path)

