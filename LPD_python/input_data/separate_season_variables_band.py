import os
import rasterio
import numpy as np

# 📌 Cartella contenente i file GeoTIFF annuali
input_dir = "/scratch/gianofe/seasonal_variables/"
output_dir = "/scratch/gianofe/seasonal_variables/final_outputs/"
os.makedirs(output_dir, exist_ok=True)

# 📌 Definizione dei file di output
sos_output_path = os.path.join(output_dir, "SOS_multiband_1999_2023.tif")
eos_output_path = os.path.join(output_dir, "EOS_multiband_1999_2023.tif")
gsl_output_path = os.path.join(output_dir, "GSL_multiband_1999_2023.tif")

# 📌 Lista degli anni da processare
years = list(range(1999, 2024))

# 📌 Liste per contenere i dati delle bande
sos_bands = []
eos_bands = []
gsl_bands = []

# 📌 Variabili per i metadati
meta = None

# 📌 Legge ogni file annuale e separa le bande
for year in years:
    file_path = os.path.join(input_dir, f"seasonal_variables_{year}.tif")

    if not os.path.exists(file_path):
        print(f"⚠️ File mancante: {file_path}, salto l'anno {year}.")
        continue

    with rasterio.open(file_path) as src:
        if meta is None:
            meta = src.meta.copy()  # 📌 Copia i metadati del primo file valido
            meta.update(count=len(years))  # 📌 Modifica il numero di bande finali
            meta.update(compress="DEFLATE", predictor=2, tiled=True)  # 📌 Aggiunta compressione

        sos_bands.append(src.read(1))  # 📌 SOS (Banda 1)
        eos_bands.append(src.read(2))  # 📌 EOS (Banda 2)
        gsl_bands.append(src.read(3))  # 📌 GSL (Banda 3)

print(f"✅ Caricati {len(sos_bands)} anni validi.")


# 📌 Funzione per salvare un file multibanda con compressione
def save_multiband(output_path, bands, meta, description):
    if len(bands) == 0:
        print(f"❌ Nessun dato valido per {description}, il file non verrà creato.")
        return

    with rasterio.open(output_path, 'w', **meta) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band, i)

    print(f"✅ File salvato con compressione: {output_path} ({description})")


# 📌 Salvataggio dei tre file finali con compressione
save_multiband(sos_output_path, sos_bands, meta, "SOS")
save_multiband(eos_output_path, eos_bands, meta, "EOS")
save_multiband(gsl_output_path, gsl_bands, meta, "GSL")

print("🎯 Elaborazione completata con compressione DEFLATE!")
