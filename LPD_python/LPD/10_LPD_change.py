import os
import numpy as np
import rasterio
import logging
from rasterio.warp import reproject, Resampling

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Percorsi dei file raster
baseline_path = "/home/gianofe/Desktop/Documents/corrected/OldLPD/LC_sum_reporting_LPD_finalMap.tif"
reporting_path = "/home/gianofe/Desktop/Documents/corrected/output/bs_LPD_finalMap.tif"
output_path = "/home/gianofe/Desktop/Documents/corrected/output/change_2019-2023.tif"

# Caricamento dei raster
logging.info("Caricamento dei raster...")
with rasterio.open(baseline_path) as src:
    LPD_baseline = src.read(1)
    baseline_meta = src.meta.copy()
    baseline_transform = src.transform
    baseline_crs = src.crs
    nodata_value = src.nodata

with rasterio.open(reporting_path) as src:
    LPD_reporting = src.read(1)
    reporting_meta = src.meta.copy()
    reporting_transform = src.transform
    reporting_crs = src.crs

# Log delle informazioni sui raster
logging.info(f"Dimensioni baseline: {LPD_baseline.shape}, Transform: {baseline_transform}, CRS: {baseline_crs}")
logging.info(f"Dimensioni reporting: {LPD_reporting.shape}, Transform: {reporting_transform}, CRS: {reporting_crs}")

# Controllo di allineamento tra i due raster
if LPD_baseline.shape != LPD_reporting.shape or baseline_transform != reporting_transform or baseline_crs != reporting_crs:
    logging.warning("I raster non sono allineati. Tentativo di riallineamento...")
    LPD_reporting_aligned = np.empty_like(LPD_baseline)
    reproject(
        source=LPD_reporting,
        destination=LPD_reporting_aligned,
        src_transform=reporting_transform,
        src_crs=reporting_crs,
        dst_transform=baseline_transform,
        dst_crs=baseline_crs,
        resampling=Resampling.nearest
    )
    LPD_reporting = LPD_reporting_aligned
    logging.info("Raster di reporting riallineato con successo!")

# Creazione della matrice per i cambiamenti
logging.info("Inizializzazione della matrice di cambiamento...")
LPD_change_vals = np.full(LPD_baseline.shape, np.nan)

# Lookup table per la riclassificazione
define_change = {
    (1, 1): 2, (1, 2): 3, (1, 3): 3, (1, 4): 3, (1, 5): 3,
    (2, 1): 1, (2, 2): 2, (2, 3): 3, (2, 4): 3, (2, 5): 3,
    (3, 1): 1, (3, 2): 1, (3, 3): 2, (3, 4): 3, (3, 5): 3,
    (4, 1): 1, (4, 2): 1, (4, 3): 1, (4, 4): 2, (4, 5): 3,
    (5, 1): 1, (5, 2): 1, (5, 3): 1, (5, 4): 1, (5, 5): 2
}

logging.info("Calcolo delle classi di cambiamento...")
for (base, report), change in define_change.items():
    mask = (LPD_baseline == base) & (LPD_reporting == report)
    LPD_change_vals[mask] = change

# Gestione dei NoData nel baseline ma con valore nel reporting
decision = "assign"  # Opzioni: "nodata" per mantenere NoData, "assign" per assegnare la classe riclassificata
if decision == "assign":
    for report_value in np.unique(LPD_reporting[~np.isnan(LPD_reporting)]):
        if report_value in [1, 2, 3, 4, 5]:  # Solo classi valide
            mask = (np.isnan(LPD_baseline)) & (LPD_reporting == report_value)
            if mask.any():
                LPD_change_vals[mask] = define_change.get((report_value, report_value), np.nan)
elif decision == "nodata":
    LPD_change_vals[np.isnan(LPD_baseline)] = nodata_value

# Scrittura del raster risultante
baseline_meta.update(dtype=rasterio.float32)
logging.info("Scrittura del raster di output...")
with rasterio.open(output_path, 'w', **baseline_meta) as dst:
    dst.write(LPD_change_vals.astype(rasterio.float32), 1)

logging.info("Processo completato con successo!")
