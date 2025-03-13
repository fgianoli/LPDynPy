import os
import shutil
import numpy as np
import rasterio
from rasterio.transform import from_origin
from netCDF4 import Dataset
import pandas as pd

# Percorso al file CSV e directory di output
csv_path = '/home/gianofe/Documents/ndvi1km_wgt_avg_FG2.csv'
output_dir= '/eos/jeodpp/data/projects/C-GLS-LTS/data/bias_corrected_FG/cf'
#output_dir = '/home/gianofe/Desktop/Documents/corrected/cf/'

# Costanti di correzione
VGT_ADDITION = 0.027548447
PROBAV_ADDITION = 0.014737188

# Carica i dati dal file CSV
data = pd.read_csv(csv_path)

# Assicurati che la directory di output esista ed è scrivibile
cf_output_dir = os.path.join(output_dir, "cf")
os.makedirs(cf_output_dir, exist_ok=True)

if not os.access(cf_output_dir, os.W_OK):
    print(f"Errore: La cartella {cf_output_dir} non è scrivibile!")
    exit()

# Input per selezionare l'anno di inizio dell'analisi
try:
    start_year = int(input("Inserisci l'anno di inizio analisi (premi Invio per analizzare tutti gli anni): ").strip() or min(data['year']))
except ValueError:
    print("Errore: Devi inserire un numero intero valido per l'anno di inizio.")
    exit()

# Seleziona solo gli anni >= start_year
years = sorted(data['year'].unique())
years = [y for y in years if y >= start_year]

# Funzione per controllare spazio su disco
def check_disk_space(path, min_space=10):  # min_space in GB
    stat = shutil.disk_usage(path)
    available_gb = stat.free / (1024**3)
    return available_gb > min_space

for year in years:
    year_data = data[data['year'] == year]
    ndvi_stack = []  # Stack per calcolare media e deviazione standard
    transform = None  # Trasformazione affine

    print(f"\nElaborazione per l'anno {year}...")

    # Processa tutti i file di un anno
    for index, row in year_data.iterrows():
        file_path = row['path']
        satellite = row['satellite']

        # Controllo spazio su disco
        if not check_disk_space(cf_output_dir):
            print("Errore: Spazio su disco insufficiente, interrompendo elaborazione.")
            break

        # Controlla se il file esiste
        if not os.path.exists(file_path):
            print(f"File non trovato: {file_path}")
            continue

        # Determina il valore da aggiungere (BIAS correction)
        if satellite == 'VGT':
            addition_value = VGT_ADDITION
        elif satellite == 'PROBAV':
            addition_value = PROBAV_ADDITION
        elif satellite == 'OLCI':
            print(f"Saltato sensore OLCI per il file: {file_path}")
            continue
        else:
            print(f"Satellite non supportato: {satellite}")
            continue

        # Leggi il layer NDVI dal file NetCDF
        try:
            with Dataset(file_path, 'r') as nc_file:
                if 'NDVI' not in nc_file.variables:
                    print(f"Layer NDVI non trovato nel file: {file_path}")
                    continue

                # Leggi i dati NDVI e applica la correzione
                ndvi_data = nc_file.variables['NDVI'][:].astype(np.float32) + addition_value
                ndvi_data = np.squeeze(ndvi_data)  # Rimuovi dimensioni inutili

                # Ottieni le informazioni di georeferenziazione dal file NetCDF
                lat = nc_file.variables['lat'][:]
                lon = nc_file.variables['lon'][:]
                resolution_x = lon[1] - lon[0]
                resolution_y = lat[0] - lat[1]  # Y è negativa per scendere dall'alto verso il basso
                origin_x = lon[0]
                origin_y = lat[0]

                # Crea la trasformazione affine
                nc_transform = from_origin(origin_x, origin_y, resolution_x, resolution_y)

                # Inizializza la trasformazione affine
                if transform is None:
                    transform = nc_transform

                # Aggiungi i dati NDVI alla pila (stack)
                ndvi_stack.append(ndvi_data)

        except Exception as e:
            print(f"Errore nella lettura del file NetCDF: {file_path}, {e}")
            continue

        print(f"File processato: {file_path}")

    # Calcolo della CF se ci sono dati disponibili
    if ndvi_stack:
        ndvi_stack = np.array(ndvi_stack)  # Converti in array

        # Calcolo della media annuale e deviazione standard
        ndvi_mean = np.mean(ndvi_stack, axis=0)
        ndvi_std = np.std(ndvi_stack, axis=0)

        # Calcolo della frazione ciclica (CF)
        with np.errstate(divide='ignore', invalid='ignore'):
            cyclic_fraction = np.where(ndvi_mean != 0, ndvi_std / ndvi_mean, 0)

        # Percorso file output
        output_path = os.path.join(cf_output_dir, f'CF_{year}.tif')

        # Controllo spazio su disco prima della scrittura
        if not check_disk_space(cf_output_dir):
            print("Errore: Spazio su disco insufficiente per salvare il file!")
            continue

        # Rimuovi file se esiste già
        if os.path.exists(output_path):
            os.remove(output_path)

        # Profilo di trasformazione con compressione
        profile = {
            'driver': 'GTiff',
            'height': cyclic_fraction.shape[0],
            'width': cyclic_fraction.shape[1],
            'count': 1,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': transform,
            'compress': 'LZW'  # Aggiunta compressione per ridurre errore di memoria
        }

        print(f"Scrivendo il file CF per l'anno {year} in {output_path}...")

        # Scrittura del file GeoTIFF
        try:
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(cyclic_fraction, 1)
            print(f"File CF annuale salvato: {output_path}")

        except rasterio.errors.RasterioIOError as e:
            print(f"Errore I/O durante la scrittura di {output_path}: {e}")
            continue
        except Exception as e:
            print(f"Errore generico nella scrittura di {output_path}: {e}")
            continue

print("\nElaborazione completata!")
