import os
import numpy as np
import rasterio


def mean_years_function(x):
    if np.isnan(x).all():
        print("Warning: All values are NaN. Replacing with zeros.")
        return np.zeros_like(x[0])
    return np.nanmean(np.where(np.isnan(x), 0, x), axis=0)


def remove_multicollinearity(file_paths, yrs2use=None, multicol_cutoff=0.7, filename=""):
    print("Starting multicollinearity removal process...")

    if not isinstance(file_paths, list) or not all(os.path.isfile(fp) for fp in file_paths):
        raise ValueError("Please provide a list of valid file paths.")

    tif_files = file_paths
    if not tif_files:
        raise ValueError("No .tif files found in the specified directory.")

    raster_stack = []
    var_names = []
    profile = None
    ref_shape = None  # Riferimento per la dimensione comune

    for tif in tif_files:
        print(f"Processing file: {tif}")
        with rasterio.open(tif) as src:
            data = src.read()
            print(f"Raster {tif} shape: {data.shape}")  # Stampa la dimensione del raster
            if ref_shape is None:
                ref_shape = data.shape  # Imposta il primo raster come riferimento
            elif data.shape != ref_shape:
                print(f"Resampling {tif} to match reference shape {ref_shape}.")
                data = np.array([np.resize(band, ref_shape[1:]) for band in data])
            if profile is None:
                profile = src.profile

        if yrs2use:
            data = data[yrs2use]
            mean_data = mean_years_function(data)
            raster_stack.append(mean_data)
            var_names.append(os.path.basename(tif).replace(".tif", ""))

    raster_stack = np.stack(raster_stack, axis=0)  # Shape: (n_variables, height, width)
    print("Stacked raster shape:", raster_stack.shape)

    if len(var_names) > 1:
        height, width = raster_stack.shape[1], raster_stack.shape[2]
        reshaped_stack = raster_stack.reshape(len(var_names), -1).T  # Reshape for correlation analysis

        print("Calculating Pearson correlation...")
        corr_matrix = np.corrcoef(reshaped_stack, rowvar=False)

        to_keep = []
        for i in range(len(var_names)):
            if all(abs(corr_matrix[i, j]) < multicol_cutoff or i == j for j in range(len(var_names))):
                to_keep.append(i)

        print(f"Variables retained after filtering: {len(to_keep)}")
        filtered_stack = raster_stack[to_keep, :, :]
    else:
        print("Only one variable found, skipping correlation analysis.")
        filtered_stack = raster_stack

    if filename:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Aggiornare il profilo con il numero corretto di bande
        profile.update(count=filtered_stack.shape[0])

        print(f"Saving output raster to {filename}")
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(filtered_stack)

    print("Multicollinearity removal complete!")
    return filtered_stack


if __name__ == "__main__":
    input_files = [
        "/home/gianofe/Desktop/corrected/sumndvi_multiband_xxl.tif",
        "/home/gianofe/Desktop/corrected/cf_multiband.tif",
        "/scratch/gianofe/seasonal_variables/final_outputs/EOS_multiband_1999_2023.tif",
        "/scratch/gianofe/seasonal_variables/final_outputs/SOS_multiband_1999_2023.tif",
        "/scratch/gianofe/seasonal_variables/final_outputs/GSL_multiband_1999_2023.tif"
    ]
    output_raster = "/home/gianofe/Documents/output/no_col.tif"

    print("Running remove_multicollinearity function...")
    filtered_rasters = remove_multicollinearity(
        file_paths=input_files,
        yrs2use=list(range(9, 25)),  # Example: Use years 9-24
        multicol_cutoff=0.7,
        filename=output_raster
    )
    print("Process completed successfully!")
