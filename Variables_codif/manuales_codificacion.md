# Guía de Estándares de Codificación Clínica y Manuales Oficiales

Este documento proporciona una descripción general de los sistemas de codificación identificados en el dataset, junto con enlaces a sus manuales y navegadores oficiales para su verificación y estudio en profundidad.

## 1. SNOMED CT (Systematized Nomenclature of Medicine -- Clinical Terms)

**Uso principal:** Diagnósticos, Hallazgos Clínicos, Procedimientos, Anatomía, Organismos.
**Identificación en dataset:** Códigos numéricos largos (6-18 dígitos). Descripciones con "Semantic Tags" al final, e.g., `(disorder)`, `(procedure)`.

### Estándar y Estructura
SNOMED CT es una terminología clínica internacional multilingüe. Se basa en tres componentes principales:
- **Conceptos:** Unidades de pensamiento clínico (cada uno con un *SCTID* único).
- **Descripciones:** Términos legibles por humanos (Fully Specified Name, Synonym).
- **Relaciones:** Vínculos lógicos entre conceptos (e.g., *Is a*, *Causative agent*).

### Recursos Oficiales
- **Navegador Oficial (SNOMED International Browser):**  
  [https://browser.ihtsdotools.org/](https://browser.ihtsdotools.org/)  
  *Herramienta esencial para buscar cualquier código y ver sus padres, hijos y relaciones.*

- **Guía Editorial (Editorial Guide):**  
  [https://confluence.ihtsdotools.org/display/DOCEDITORIAL](https://confluence.ihtsdotools.org/display/DOCEDITORIAL)  
  *Explica las reglas de creación de contenido y los "Semantic Tags".*

- **Guía de Inicio (Starter Guide):**  
  [https://confluence.ihtsdotools.org/display/DOCSTART](https://confluence.ihtsdotools.org/display/DOCSTART)

---

## 2. LOINC (Logical Observation Identifiers Names and Codes)

**Uso principal:** Resultados de Laboratorio, Signos Vitales, Medidas Clínicas, Documentos Clínicos.
**Identificación en dataset:** Formato `nnnnn-n` (dígitos con guión y dígito verificador, e.g., `72514-3`).

### Estándar y Estructura
LOINC estandariza los identificadores universales para pruebas médicas. Cada código LOINC se define por 6 ejes (Partes):
1. **Componente:** Qué se mide (e.g., Glucosa).
2. **Propiedad:** Característica medida (e.g., concentración de masa).
3. **Tiempo:** Momento de la medida (e.g., punto en el tiempo, 24 horas).
4. **Sistema:** Muestra (e.g., sangre arterial, orina).
5. **Escala:** Tipo de resultado (e.g., cuantitativo, ordinal).
6. **Método:** (Opcional) Cómo se midió.

### Recursos Oficiales
- **Buscador Oficial (Search LOINC):**  
  [https://loinc.org/search/](https://loinc.org/search/)  
  *Permite buscar códigos y ver todos los detalles técnicos de los 6 ejes.*

- **Manual de Usuario (LOINC Users' Guide):**  
  [https://loinc.org/kb/users-guide/](https://loinc.org/kb/users-guide/)  
  *Documento completo sobre la estructura y uso del estándar.*

---

## 3. RxNorm

**Uso principal:** Medicamentos (Fármacos clínicos y de marca) en EE.UU. (usado extensivamente en Synthea).
**Identificación en dataset:** Códigos numéricos. Descripciones con dosis y forma farmacéutica (e.g., `Lisinopril 10 MG Oral Tablet`).

### Estándar y Estructura
RxNorm normaliza nombres de medicamentos y relacia nombres comerciales con sus ingredientes genéricos.
- **SCD (Semantic Clinical Drug):** Ingrediente + Dosis + Forma (e.g., *Atenolol 50 MG Oral Tablet*).
- **IN (Ingredient):** Solo el principio activo.

### Recursos Oficiales (NLM)
- **RxNav (Navegador):**  
  [https://mor.nlm.nih.gov/RxNav/](https://mor.nlm.nih.gov/RxNav/)  
  *Visualizador interactivo para navegar entre ingredientes, marcas y presentaciones.*

- **Documentación Técnica:**  
  [https://www.nlm.nih.gov/research/umls/rxnorm/index.html](https://www.nlm.nih.gov/research/umls/rxnorm/index.html)

---

## 4. CVX (Vaccines Administered)

**Uso principal:** Vacunas (Inmunizaciones).
**Identificación en dataset:** Códigos numéricos cortos o medios. Descripciones contienen "Vaccine".

### Estándar y Estructura
Conjunto de códigos mantenido por los CDC (Centers for Disease Control and Prevention) para identificar vacunas administradas.

### Recursos Oficiales (CDC)
- **Tabla Maestra de CVX a MVX:**  
  [https://www2a.cdc.gov/vaccines/iis/iisstandards/vaccines.asp?rpt=cvx](https://www2a.cdc.gov/vaccines/iis/iisstandards/vaccines.asp?rpt=cvx)  
  *Listado oficial completo de códigos activos e inactivos.*

- **Guía de Implementación HL7 para Inmunizaciones:**  
  [https://www.cdc.gov/vaccines/programs/iis/technical-guidance/hl7.html](https://www.cdc.gov/vaccines/programs/iis/technical-guidance/hl7.html)

---

## Resumen de Verificación para el Excel
Para verificar la columna `SISTEMA_CODIFICACION` generada en el Excel, puede tomar un código de muestra y buscarlo en el navegador correspondiente arriba enlazado.

- Si tiene formato `12345-6` -> Busque en **LOINC**.
- Si es numérico y parece un diagnóstico/procedimiento -> Busque en **SNOMED CT**.
- Si es numérico y parece un medicamento -> Busque en **RxNorm**.
- Si es vacuna -> Busque en **CDC CVX**.
