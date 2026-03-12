import pandas as pd

def obtener_info_valores_unicos(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    soporte_map = {}
    
    for col in cat_cols:
        iloc_col = df.columns.get_loc(col)
        valores_unicos = frozenset(df[col].dropna().unique())
        
        if valores_unicos not in soporte_map:
            soporte_map[valores_unicos] = []
            
        soporte_map[valores_unicos].append(iloc_col)
            
    return pd.DataFrame([
        {"Indices_iloc": indices, "Soporte": list(valores_unicos)}
        for valores_unicos, indices in soporte_map.items()
    ])

def extraccion_columnas(dataset, tabla_soportes, soporte=[[]]):
    columnas = tabla_soportes[tabla_soportes["Soporte"].isin(soporte)]["Indices_iloc"]
    columnas_indices = list({item for sublist in columnas for item in sublist})
    columnas_nombres = dataset.columns[columnas_indices]
    return columnas_nombres
