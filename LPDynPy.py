import numpy as np
import rasterio
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Funzioni convertite dal file 001_functions.R
def slp_lm(x, yrs):
    if all(np.isnan(x)):
        slp = np.nan
    elif np.sum(~np.isnan(x)) == 1:
        slp = 0
    else:
        model = LinearRegression().fit(np.array(yrs).reshape(-1, 1), np.array(x))
        slp = model.coef_[0]
    return slp

def mtid_function(x):
    if all(np.isnan(x)):
        mtid = np.nan
    elif np.sum(~np.isnan(x)) == 1:
        mtid = 0
    else:
        years1 = max(np.where(~np.isnan(x))[0])
        mtid = np.nansum(x[years1] - x[:years1])
    return mtid

def mean_years_function(x, yrs):
    return np.nanmean(np.array(x)[yrs])

def intercept_lm(x, yrs):
    if all(np.isnan(x)):
        intercept = np.nan
    elif np.sum(~np.isnan(x)) == 1:
        intercept = 0
    else:
        model = LinearRegression().fit(np.array(yrs).reshape(-1, 1), np.array(x))
        intercept = model.intercept_
    return intercept

def sum_positive(x):
    if all(np.isnan(x)):
        return np.nan
    else:
        return np.nansum(np.where(np.array(x) > 0, x, 0))

def sum_negative(x):
    if all(np.isnan(x)):
        return np.nan
    else:
        return np.nansum(np.where(np.array(x) < 0, x, 0))

def range_function(x):
    if all(np.isnan(x)):
        return np.nan
    else:
        return np.nanmax(x) - np.nanmin(x)

def diff_lag_function(x, lag=1):
    x = np.array(x)
    diff = np.full_like(x, np.nan)
    for i in range(lag, len(x)):
        if not np.isnan(x[i]) and not np.isnan(x[i - lag]):
            diff[i] = x[i] - x[i - lag]
    return diff

# Funzioni convertite dal file 02_steadiness.R
def steadiness(obj2process=None, cores2use=1, filename=""):
    if obj2process is None:
        raise ValueError("Fornisci un oggetto di classe numpy array o un nome di file da leggere")
    
    if isinstance(obj2process, str):
        with rasterio.open(obj2process) as src:
            obj2process = src.read()
    elif not isinstance(obj2process, np.ndarray):
        raise ValueError("Fornisci un oggetto di classe numpy array o un nome di file da leggere")
    
    yrs = np.arange(1, obj2process.shape[0] + 1)
    
    slope_rstr = np.apply_along_axis(slp_lm, 0, obj2process, yrs)
    
    mtid_rstr = np.apply_along_axis(mtid_function, 0, obj2process)
    
    SteadInd_rstr = np.zeros_like(slope_rstr)
    SteadInd_rstr[(slope_rstr < 0) & (mtid_rstr > 0)] = 1
    SteadInd_rstr[(slope_rstr < 0) & (mtid_rstr < 0)] = 2
    SteadInd_rstr[(slope_rstr > 0) & (mtid_rstr < 0)] = 3
    SteadInd_rstr[(slope_rstr > 0) & (mtid_rstr > 0)] = 4
    SteadInd_rstr[(slope_rstr == 0) | (mtid_rstr == 0)] = 0

    if filename:
        with rasterio.open(filename, 'w', driver='GTiff', height=SteadInd_rstr.shape[0],
                           width=SteadInd_rstr.shape[1], count=1, dtype=str(SteadInd_rstr.dtype)) as dst:
            dst.write(SteadInd_rstr, 1)

    return SteadInd_rstr

# Funzioni convertite dal file 03_baseline_lev.R
def baseline_lev(obj2process=None, yearsBaseline=3, drylandProp=0.4, highprodProp=0.1, cores2use=1, filename=""):
    if obj2process is None:
        raise ValueError("Fornisci un oggetto di classe numpy array o un nome di file da leggere")
    
    if isinstance(obj2process, str):
        with rasterio.open(obj2process) as src:
            obj2process = src.read()
    elif not isinstance(obj2process, np.ndarray):
        raise ValueError("Fornisci un oggetto di classe numpy array o un nome di file da leggere")
    
    if yearsBaseline > obj2process.shape[0]:
        yearsBaseline = obj2process.shape[0]

    obj2process_avg13 = np.nanmean(obj2process[:yearsBaseline, :, :], axis=0)
    
    if drylandProp > 1:
        drylandProp = drylandProp / 100
    if highprodProp > 1:
        highprodProp = highprodProp / 100
    if highprodProp + drylandProp > 1:
        raise ValueError("'highprodProp' + 'drylandProp' non può essere > 100%")
    
    percentiles = np.percentile(obj2process_avg13[~np.isnan(obj2process_avg13)], np.arange(0, 110, 10))
    low_threshold = percentiles[int(drylandProp * 10)]
    high_threshold = percentiles[-int(highprodProp * 10) - 1]
    
    obj2process_3class = np.digitize(obj2process_avg13, bins=[low_threshold, high_threshold], right=True) + 1

    if filename:
        with rasterio.open(filename, 'w', driver='GTiff', height=obj2process_3class.shape[0],
                           width=obj2process_3class.shape[1], count=1, dtype=str(obj2process_3class.dtype)) as dst:
            dst.write(obj2process_3class, 1)

    return obj2process_3class

# Funzioni convertite dal file 04_state_change.R
def state_change(obj2process=None, yearsBaseline=3, stateChangeThreshold=1, cores2use=1, filename=""):
    if obj2process is None:
        raise ValueError("Fornisci un oggetto di classe numpy array o un nome di file da leggere")
    
    if isinstance(obj2process, str):
        with rasterio.open(obj2process) as src:
            obj2process = src.read()
    elif not isinstance(obj2process, np.ndarray):
        raise ValueError("Fornisci un oggetto di classe numpy array o un nome di file da leggere")
    
    if yearsBaseline > obj2process.shape[0]:
        yearsBaseline = obj2process.shape[0]

    baseline_avg = np.nanmean(obj2process[:yearsBaseline, :, :], axis=0)
    final_avg = np.nanmean(obj2process[-yearsBaseline:, :, :], axis=0)
    
    state_diff = final_avg - baseline_avg
    state_change_classified = np.zeros_like(state_diff, dtype=int)
    state_change_classified[(state_diff >= -stateChangeThreshold) & (state_diff <= stateChangeThreshold)] = 1
    state_change_classified[state_diff > stateChangeThreshold] = 2
    state_change_classified[state_diff < -stateChangeThreshold] = 3

    if filename:
        with rasterio.open(filename, 'w', driver='GTiff', height=state_change_classified.shape[0],
                           width=state_change_classified.shape[1], count=1, dtype=str(state_change_classified.dtype)) as dst:
            dst.write(state_change_classified, 1)

    return state_change_classified

# Funzioni convertite dal file 05_LongTermChange.R
def LongTermChange(steadiness_index=None, baseline_levels=None, state_changes=None, cores2use=1, filename=""):
    if steadiness_index is None or baseline_levels is None or state_changes is None:
        raise ValueError("Fornisci tutti gli array numpy richiesti per il calcolo")
    
    if steadiness_index.shape != baseline_levels.shape or steadiness_index.shape != state_changes.shape:
        raise ValueError("Gli array input devono avere le stesse dimensioni")
    
    long_term_change_map = np.zeros_like(steadiness_index, dtype=int)
    
    for idx, (st_index, base_level, state_change) in np.ndenumerate(zip(steadiness_index, baseline_levels, state_changes)):
        if not (np.isnan(st_index) or np.isnan(base_level) or np.isnan(state_change)):
            if base_level == 1 and state_change == 1:
                long_term_change_map[idx] = 1
            elif base_level == 1 and state_change == 2:
                long_term_change_map[idx] = 2
            elif base_level == 1 and state_change == 3:
                long_term_change_map[idx] = 3
            # Aggiungi altre combinazioni secondo le regole di classificazione...

    if filename:
        with rasterio.open(filename, 'w', driver='GTiff', height=long_term_change_map.shape[0],
                           width=long_term_change_map.shape[1], count=1, dtype=str(long_term_change_map.dtype)) as dst:
            dst.write(long_term_change_map, 1)

    return long_term_change_map

# Funzioni convertite dal file 06_rm_multicol.R
def rm_multicol(dir2process=None, multicol_cutoff=0.9, filename=""):
    if dir2process is None:
        raise ValueError("Fornisci una directory da processare o un array numpy di variabili")

    variables = dir2process if isinstance(dir2process, np.ndarray) else np.array([])

    if len(variables.shape) == 3:
        variables_avg = np.nanmean(variables, axis=0)
    else:
        variables_avg = variables

    corr_matrix = np.corrcoef(variables_avg, rowvar=False)
    
    to_remove = set()
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > multicol_cutoff:
                to_remove.add(j)

    variables_filtered = np.delete(variables_avg, list(to_remove), axis=1)

    if filename:
        with rasterio.open(filename, 'w', driver='GTiff', height=variables_filtered.shape[0],
                           width=variables_filtered.shape[1], count=1, dtype=str(variables_filtered.dtype)) as dst:
            dst.write(variables_filtered, 1)

    return variables_filtered

# Funzioni convertite dal file 07_08_PCAs.R
def PCAs4clust(obj2process=None, screening_threshold=0.9, final_n_components=3, filename=""):
    if obj2process is None or not isinstance(obj2process, np.ndarray):
        raise ValueError("Fornisci un array numpy da analizzare")
    
    pca_screening = PCA()
    pca_screening.fit(obj2process)
    explained_variance = np.cumsum(pca_screening.explained_variance_ratio_)
    optimal_components = np.searchsorted(explained_variance, screening_threshold) + 1
    
    pca_final = PCA(n_components=min(final_n_components, optimal_components))
    obj2process_pca = pca_final.fit_transform(obj2process)
    
    if filename:
        np.save(filename, obj2process_pca)

    return obj2process_pca

# Funzioni convertite dal file 090_optim_cust.R
def clust_optim(data=None, max_clusters=10, filename=""):
    if data is None or not isinstance(data, np.ndarray):
        raise ValueError("Fornisci un array numpy di dati per il clustering")
    
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    plt.figure()
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Grafico a Gomito')
    plt.xlabel('Numero di cluster')
    plt.ylabel('Somma totale dei quadrati intra-cluster (WCSS)')
    plt.grid(True)
    
    if filename:
        plt.savefig(filename)
    plt.show()
    
    return wcss

# Funzioni convertite dal file 09_EFTs_custering.R
def EFT_clust(data=None, n_clust=20, filename=""):
    if data is None or not isinstance(data, np.ndarray):
        raise ValueError("Fornisci un array numpy di dati per il clustering")
    
    kmeans = KMeans(n_clusters=n_clust, random_state=42)
    labels = kmeans.fit_predict(data)
    
    if filename:
        np.save(filename, labels)

    return labels

# Funzioni convertite dal file 10_LNS.R
def LNScaling(prod_var=None, efts=None, filename=""):
    if prod_var is None or efts is None:
        raise ValueError("Fornisci sia una variabile di produttività che le aree funzionali dell'ecosistema")

    if not isinstance(prod_var, np.ndarray) or not isinstance(efts, np.ndarray):
        raise ValueError("Entrambi i parametri devono essere array numpy")

    unique_fts = np.unique(efts[~np.isnan(efts)])
    lns = np.full_like(prod_var, np.nan, dtype=np.float64)

    for ft in unique_fts:
        mask = efts == ft
        if np.any(mask):
            mean_prod = np.nanmean(prod_var[mask])
            lns[mask] = prod_var[mask] / mean_prod if mean_prod != 0 else np.nan

    if filename:
        np.save(filename, lns)

    return lns

# Funzioni convertite dal file 11_CombinedAssessment.R
def LPD_CombAssess(landprod_change=None, landprod_current=None, local_prod_threshold=0.5, filename=""):
    if landprod_change is None or landprod_current is None:
        raise ValueError("Fornisci sia una mappa di cambiamento della produttività che una mappa dello stato attuale")

    if not isinstance(landprod_change, np.ndarray) or not isinstance(landprod_current, np.ndarray):
        raise ValueError("Entrambi i parametri devono essere array numpy")

    combined_map = np.zeros_like(landprod_change, dtype=int)

    combined_map[(landprod_change > 0) & (landprod_current >= local_prod_threshold)] = 5
    combined_map[(landprod_change > 0) & (landprod_current < local_prod_threshold)] = 4
    combined_map[(landprod_change == 0) & (landprod_current >= local_prod_threshold)] = 3
    combined_map[(landprod_change == 0) & (landprod_current < local_prod_threshold)] = 2
    combined_map[(landprod_change < 0)] = 1

    if filename:
        np.save(filename, combined_map)

    return combined_map
