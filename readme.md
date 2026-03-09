# Estructura de las carpetas

- MemoriaTFM: Será la carpeta que contenga la memoria del TFM.
- Variables_codif: Contiene las codificaciones de las variables. La identificación de cada sistema de codificación se realizo mediante el script `identify_codes.py`.   

    - SNOMED: Contiene los script, excel y json necesarios para la codificación de las variables SNOMED. 
        - Se realizó la separación por diagnósticos, procedimientos... con la norma `Norma_mantener.md` mediante el excel `clasificacion_semantica_snomed_con_porcentajes.xlsx`
        - `analyze_ancestors_api.py`: Script para analizar los ancestros de las variables SNOMED originales mediante consultas en servidor Docker local
        - Se usan scripts `group_disorders_findings`, `group_procedures_therapies` y `group_resto` para agrupar los conceptos SNOMED según su morfología y topografía médica. Estos script generan los JSON que se usaran para las agrupaciones de variables.
        - Por último se analizan las variables originales con `analyze_snomed_data_types.py` para asegurarnos de que las agrupaciones se hacen de manera correcta.

    - RxNorm: Contiene los script, excel y json necesarios para la codificación de las variables RxNorm y su mapeo a ATC. 
        - Se mapean los códigos RxNorm a ATC 5, luego se agrupan en ATC 2. Esta agrupación se encuentra en el JSON. Para crear el csv se usa el script `analyze_rxnorm_data_types.py`.

    - LOINC: Contiene los script, excel y json necesarios para la codificación de las variables LOINC.
    - Reveladoras_Cancer: Contiene json con las variables reveladoras de cáncer que no suelen ser habituales en cribados generales.