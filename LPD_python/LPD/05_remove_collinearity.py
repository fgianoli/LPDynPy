import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


def mean_years_function(x):
    """Calcola la media dei valori non-NaN, sostituendo NaN con 0."""
    if np.isnan(x).all():
        print("Warning: All values are NaN. Replacing with zeros.")
        return np.zeros_like(x[0])
    return np.nanmean(np.where(np.isnan(x), 0, x), axis=0)


def remove_multicollinearity(file_paths, yrs2use=None, multicol_cutoff=0.7, filename=""):
    """
    Rimuove la multicollinearità tra variabili raster.

    Args:
        file_paths (list): Lista dei percorsi ai file raster.
        yrs2use (list, optional): Indici delle bande da utilizzare.
        multicol_cutoff (float, optional): Soglia per identificare la multicollinearità.
        filename (str, optional): Nome del file di output.

    Returns:
        numpy.ndarray: Stack di raster filtrati.
    """
    print("Starting multicollinearity removal process...")

    if not isinstance(file_paths, list) or not all(os.path.isfile(fp) for fp in file_paths):
        raise ValueError("Please provide a list of valid file paths.")

    tif_files = file_paths
    if not tif_files:
        raise ValueError("No .tif files found in the specified directory.")

    raster_stack = []
    var_names = []
    profile = None
    ref_shape = None
    ref_transform = None
    ref_crs = None

    for tif in tif_files:
        print(f"Processing file: {tif}")
        with rasterio.open(tif) as src:
            data = src.read()
            print(f"Raster {tif} shape: {data.shape}")

            # Imposta il primo raster come riferimento
            if ref_shape is None:
                ref_shape = data.shape
                ref_transform = src.transform
                ref_crs = src.crs
                profile = src.profile
            elif data.shape != ref_shape:
                print(f"Resampling {tif} to match reference shape {ref_shape}.")
                output = np.zeros(ref_shape, dtype=data.dtype)
                for i, band in enumerate(data):
                    # Usa reproject di rasterio per un corretto resampling
                    reproject(
                        source=band,
                        destination=output[i],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear
                    )
                data = output

        # Seleziona solo le bande specificate
        if yrs2use is not None:
            # Verifica che gli indici richiesti esistano nel raster
            valid_indices = [idx for idx in yrs2use if idx < data.shape[0]]
            if len(valid_indices) != len(yrs2use):
                missing = set(yrs2use) - set(valid_indices)
                print(f"Warning: Indices {missing} exceed raster dimensions. Using only valid indices.")

            # Se ci sono indici validi, seleziona quelle bande
            if valid_indices:
                data = data[valid_indices]
            else:
                print(f"Warning: No valid bands for {tif} with specified indices.")
                continue

        # Calcola la media delle bande selezionate
        mean_data = mean_years_function(data)
        raster_stack.append(mean_data)
        var_names.append(os.path.basename(tif).replace(".tif", ""))

    # Verifica che ci siano dati da elaborare
    if not raster_stack:
        raise ValueError("No valid data to process after filtering.")

    raster_stack = np.stack(raster_stack, axis=0)  # Shape: (n_variables, height, width)
    print("Stacked raster shape:", raster_stack.shape)

    # Analisi di correlazione solo se ci sono almeno 2 variabili
    if len(var_names) > 1:
        # Prepara i dati per l'analisi di correlazione
        # Reshaping per avere le variabili come colonne e i pixel come righe
        height, width = raster_stack.shape[1], raster_stack.shape[2]
        reshaped_stack = raster_stack.reshape(len(var_names), -1).T

        # Filtra i pixel dove tutte le variabili sono NaN
        valid_pixels = ~np.isnan(reshaped_stack).all(axis=1)
        if valid_pixels.sum() == 0:
            print("Warning: No valid pixels for correlation analysis. Returning all variables.")
            filtered_stack = raster_stack
        else:
            valid_data = reshaped_stack[valid_pixels]

            print("Calculating Pearson correlation...")
            # Calcola la matrice di correlazione
            corr_matrix = np.corrcoef(valid_data, rowvar=False)

            print("Correlation matrix:")
            for i, var1 in enumerate(var_names):
                row = [f"{corr:.2f}" for corr in corr_matrix[i]]
                print(f"{var1}: {', '.join(row)}")

            # Identifica le variabili da mantenere (con bassa correlazione)
            to_keep = []
            for i in range(len(var_names)):
                if all(abs(corr_matrix[i, j]) < multicol_cutoff or i == j for j in range(len(var_names))):
                    to_keep.append(i)

            if not to_keep:
                print("Warning: All variables are highly correlated. Keeping the first one.")
                to_keep = [0]

            print(f"Variables retained after filtering: {len(to_keep)}")
            print(f"Retained variables: {[var_names[i] for i in to_keep]}")

            filtered_stack = raster_stack[to_keep, :, :]
    else:
        print("Only one variable found, skipping correlation analysis.")
        filtered_stack = raster_stack

    # Salva il risultato se richiesto
    if filename:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Aggiorna il profilo con il numero corretto di bande
        profile.update(count=filtered_stack.shape[0])

        print(f"Saving output raster to {filename}")
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(filtered_stack)

    print("Multicollinearity removal complete!")
    return filtered_stack


if __name__ == "__main__":
    # Esempio di utilizzo
    input_files = [
        "/scratch/gianofe/SumNDVI_correction_rev4/NDVI_SUM_multiband_rev4.tif",
        #"/home/gianofe/Desktop/corrected/cf_multiband.tif",
        "/scratch/gianofe/cf_rev2/cf/cf_new.tif",
        "/scratch/gianofe/seasonal_variables/final_outputs/EOS_multiband_1999_2023.tif",
        "/scratch/gianofe/seasonal_variables/final_outputs/SOS_multiband_1999_2023.tif",
        "/scratch/gianofe/seasonal_variables/final_outputs/GSL_multiband_1999_2023.tif"
    ]
    output_raster = "/scratch/gianofe/SumNDVI_correction_rev4/outputs/outputs/no_col_2008_2023.tif"

    print("Running remove_multicollinearity function...")
    filtered_rasters = remove_multicollinearity(
        file_paths=input_files,
        yrs2use=list(range(5, 20)),  # Esempio: Usa anni 9-24
        multicol_cutoff=0.7,
        filename=output_raster
    )
    print("Process completed successfully!")

