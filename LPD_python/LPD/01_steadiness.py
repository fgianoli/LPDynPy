import os
import numpy as np
import xarray as xr
import rasterio
from scipy.stats import linregress

def slp_lm(x, yrs):
    print("Eseguendo slp_lm...")
    valid_mask = ~np.isnan(x)
    valid_yrs = yrs[valid_mask]  # Filtra solo gli anni corrispondenti ai dati validi
    valid_x = x[valid_mask]

    if len(valid_x) == 0:
        print("  Nessun dato valido, restituisco NaN")
        return np.nan
    elif len(valid_x) == 1:
        print("  Solo un valore valido, restituisco 0")
        return 0
    else:
        slope, _, _, _, _ = linregress(valid_yrs, valid_x)
        print(f"  Slope calcolato: {slope}")
        return slope

def mtid_function(x):
    print("Eseguendo mtid_function...")
    valid_mask = ~np.isnan(x)
    if np.all(~valid_mask):
        print("  Nessun dato valido, restituisco NaN")
        return np.nan
    elif np.sum(valid_mask) == 1:
        print("  Solo un valore valido, restituisco 0")
        return 0
    else:
        years1 = np.max(np.where(valid_mask))
        mtid_value = np.nansum(x[years1] - x[valid_mask])
        print(f"  MTID calcolato: {mtid_value}")
        return mtid_value

def steadiness(obj2process, filename=""):
    print("Eseguendo steadiness...")
    if isinstance(obj2process, str):
        print(f"  Aprendo raster: {obj2process}")
        with rasterio.open(obj2process) as src:
            obj2process = xr.DataArray(src.read(), dims=("band", "y", "x"))

    start_year = 1999
    selected_years = np.arange(2008, 2024 + 1)  # Anni dal 2008 al 2023
    band_indices = selected_years - start_year  # Converti anni in indici (base 0)

    print("  Selezionando gli anni di interesse...")
    obj2process = obj2process[band_indices, :, :]
    yrs = selected_years

    print("  Calcolando slope...")
    slope_rstr = np.apply_along_axis(slp_lm, 0, obj2process, yrs)
    print("  Calcolando MTID...")
    mtid_rstr = np.apply_along_axis(mtid_function, 0, obj2process)

    print("  Calcolando Steadiness Index...")
    SteadInd_rstr = np.full_like(slope_rstr, np.nan)

    SteadInd_rstr[(slope_rstr < 0) & (mtid_rstr > 0)] = 1
    SteadInd_rstr[(slope_rstr < 0) & (mtid_rstr < 0)] = 2
    SteadInd_rstr[(slope_rstr > 0) & (mtid_rstr < 0)] = 3
    SteadInd_rstr[(slope_rstr > 0) & (mtid_rstr > 0)] = 4
    SteadInd_rstr[(slope_rstr == 0) | (mtid_rstr == 0)] = 0

    if filename:
        print(f"  Salvando il raster in: {filename}")
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with rasterio.open(filename, 'w', **src.profile) as dst:
            dst.write(SteadInd_rstr, 1)

    print("  Steadiness Index calcolato con successo!")
    return SteadInd_rstr

if __name__ == "__main__":
    input_raster = "/home/gianofe/Documents/corrected/sumndvi_multiband_xxl.tif"
    output_raster = "/home/gianofe/Documents/corrected/SteadIndxxl2.tif"

    print(f"Caricando il raster di input: {input_raster}")
    with rasterio.open(input_raster) as src:
        data = src.read()
        profile = src.profile

    obj2process = xr.DataArray(data, dims=("band", "y", "x"))

    print("Avvio del calcolo della Steadiness Index...")
    SteadInd = steadiness(obj2process)

    print(f"Salvando il risultato in: {output_raster}")
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(SteadInd, 1)

    print("Calcolo completato! Risultato salvato in:", output_raster)
