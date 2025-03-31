import rasterio
import numpy as np
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os

# Percorsi dei file di input
# Modifica questi percorsi con quelli corretti
lpd_path = "/scratch/gianofe/SumNDVI_correction_rev3/outputs/final_results//LPD_finalMap_2015.tif"  # Il tuo file LPD
landcover_path = "/home/gianofe/Desktop/Documents/LC_1KM_SB.tif"  # Il tuo file Land Cover

# Percorsi dei file di output
output_dir = "/scratch/gianofe/SumNDVI_correction_rev3/outputs"
landcover_aligned_path = os.path.join(output_dir, "LC_aligned.tif")
output_path = os.path.join(output_dir, "LPD_2019_rev3_masked.tif")


def align_rasters(reference_raster, raster_to_align, output_path):
    """
    Riprogetta il raster_to_align per farlo corrispondere alla geometria del reference_raster
    """
    # Apri i raster
    with rasterio.open(reference_raster) as src:
        reference_profile = src.profile.copy()
        reference_crs = src.crs
        reference_transform = src.transform
        reference_bounds = src.bounds
        reference_width = src.width
        reference_height = src.height

        with rasterio.open(raster_to_align) as src_to_align:
            # Calcola la trasformazione necessaria
            transform, width, height = calculate_default_transform(
                src_to_align.crs, reference_crs,
                reference_width, reference_height,
                *reference_bounds
            )

            # Aggiorna il profilo per il file di output
            output_profile = src_to_align.profile.copy()
            output_profile.update({
                'crs': reference_crs,
                'transform': transform,
                'width': reference_width,
                'height': reference_height
            })

            # Riprogetta e scrivi il raster allineato
            with rasterio.open(output_path, 'w', **output_profile) as dst:
                for i in range(1, src_to_align.count + 1):
                    reproject(
                        source=rasterio.band(src_to_align, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src_to_align.transform,
                        src_crs=src_to_align.crs,
                        dst_transform=transform,
                        dst_crs=reference_crs,
                        resampling=Resampling.nearest
                    )

    return output_path


# Allinea il raster del Land Cover con il raster LPD
print("Allineamento dei raster in corso...")
landcover_aligned_path = align_rasters(lpd_path, landcover_path, landcover_aligned_path)
print(f"Land Cover allineato salvato in: {landcover_aligned_path}")

# Apri i raster
with rasterio.open(lpd_path) as lpd_src:
    lpd_data = lpd_src.read(1)
    lpd_profile = lpd_src.profile.copy()

    with rasterio.open(landcover_aligned_path) as lc_src:
        # Ora dovrebbero avere la stessa geometria, ma controlliamo comunque
        if lpd_src.width != lc_src.width or lpd_src.height != lc_src.height:
            print("ERRORE: Anche dopo l'allineamento, i raster hanno dimensioni diverse!")
            print(f"LPD: {lpd_src.width}x{lpd_src.height}, Land Cover allineato: {lc_src.width}x{lc_src.height}")
            exit(1)

        # Leggi il raster del Land Cover
        lc_data = lc_src.read(1)

        # Crea una maschera booleana dove il valore è True se la classe di Land Cover è una di quelle specificate
        mask_classes = [0, 60, 70, 80, 200]
        mask_bool = np.isin(lc_data, mask_classes)

        # Applica la maschera al raster LPD
        # Impostiamo a nodata i pixel dove mask_bool è True (cioè dove ci sono le classi specificate)
        masked_lpd = np.copy(lpd_data)
        masked_lpd[mask_bool] = lpd_profile['nodata'] if lpd_profile.get('nodata') is not None else 0

        # Se nel profilo non è specificato un valore nodata, lo aggiungiamo
        if lpd_profile.get('nodata') is None:
            lpd_profile['nodata'] = 0

        # Scrivi il nuovo raster mascherato
        with rasterio.open(output_path, 'w', **lpd_profile) as dst:
            dst.write(masked_lpd, 1)

print(f"Il raster LPD mascherato è stato salvato in: {output_path}")

# Opzionale: codice per visualizzare i risultati
# Decommentare se si desidera visualizzare
"""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Visualizza il raster LPD originale
axes[0].imshow(lpd_data)
axes[0].set_title('LPD Originale')

# Visualizza il raster Land Cover
axes[1].imshow(lc_data)
axes[1].set_title('Land Cover')

# Visualizza il raster LPD mascherato
axes[2].imshow(masked_lpd)
axes[2].set_title('LPD Mascherato')

plt.tight_layout()
plt.show()
"""
