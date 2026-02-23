# Guía de Despliegue Local: Snowstorm con Docker

Esta guía te permitirá ejecutar tu propio servidor de terminología SNOMED CT. Esto eliminará los límites de velocidad de la API pública y permitirá análisis masivos.

## Requisitos Previos
1.  **Docker Desktop** instalado y ejecutándose.
2.    - **Caso España (Ministerio de Sanidad)**:
      1.  Accede al [Área de Descargas de SNOMED CT del Ministerio de Sanidad](https://www.sanidad.gob.es/areas/saludDigital/sistemasInformacionSNS/semantica/snomed-ct/descarga/home.htm).
      2.  Identifícate (probablemente con Cl@ve o certificado digital dada tu licencia).
      3.  Busca la sección de descargas. Es muy probable que el archivo se llame algo como **`SNOMEDCT_International_edition(DeliveryEE_SNS)`** o similar.
      4.  Descárgalo.
      5.  **Verificación:** Abre el ZIP (sin descomprimirlo todo si no quieres) y asegúrate de que dentro hay una carpeta llamada `Snapshot`. Si es así, es el archivo correcto.
      6.  *Nota:* Este archivo suele ser la Edición Internacional redistribuida por el Ministerio. Úsalo en el proceso de importación.

## Paso 1: Levantar el Servidor

1.  Abre una terminal en la carpeta donde guardaste `docker-compose.yml`.
2.  Ejecuta el siguiente comando:
    ```bash
    docker-compose up -d
    ```
3.  Espera unos minutos. Verifica que el servidor está listo accediendo en tu navegador a:
    - [http://localhost:8080](http://localhost:8080)
    - Deberías ver la interfaz de Swagger UI o un mensaje de bienvenida.
    - *Nota:* El error "NOT_FOUND" en `/snowstorm/snomed-ct` es **buena señal** (significa que el servidor responde, solo que la URL local es más simple).

## Paso 2: Cargar los Datos (Método INFALIBLE: Navegador - 2 Pasos)

Tu versión del servidor utiliza un proceso de **dos pasos** (crear tarea -> subir archivo).

### Parte A: Crear el Trabajo de Importación
1.  Abre [http://localhost:8080/swagger-ui/index.html](http://localhost:8080/swagger-ui/index.html).
2.  Busca **`Import Controller`** -> **`POST /imports`**.
3.  Dale a **Try it out**.
4.  Copia y pega este JSON en el cuerpo:
    ```json
    {
      "type": "SNAPSHOT",
      "branchPath": "MAIN",
      "createCodeSystemVersion": true
    }
    ```
5.  Dale a **Execute**.
6.  Busca en la respuesta (abajo) la cabecera **`location`**.
    - Ejemplo: `http://localhost:8080/imports/a5e5c86a-...`
    - **Copia el código largo del final** (el ID).

### Parte B: Subir el Archivo
1.  Busca ahora el endpoint **`POST /imports/{importId}/archive`**.
2.  Dale a **Try it out**.
3.  Pega el ID que copiaste en el campo `importId`.
4.  Elige tu archivo ZIP en el campo `file`.
5.  Dale a **Execute**.
6.  **¡Espera!** Aquí es donde tarda los 15-20 minutos. No cierres la ventana hasta que veas un código 200.

## Paso 3: Cargar los Datos con Postman (Método Recomendado)

Dado que la terminal puede dar problemas con el formato del archivo, usaremos **Postman**, que es más visual y maneja mejor los archivos grandes.

### 1. Preparación
1.  Asegúrate de que tu contenedor Docker sigue encendido (`docker ps` debe mostrar `snowstorm-server` y `snowstorm-es`).
2.  Ten localizado tu archivo: `SnomedCT_International_Edition_(Dependencia_EE_SNS).zip`.

### 2. Configurar la Petición en Postman
1.  Abre Postman y crea una nueva petición (botón `+`).
2.  Cambia el método de `GET` a **`POST`**.
3.  En la URL escribe: `http://localhost:8080/imports`
4.  Ve a la pestaña **Body**.
5.  Selecciona la opción **form-data** (¡muy importante!).

### 3. Rellenar los Campos (Key - Value)
Tienes que añadir exactamente estas 4 filas. **Ojo al tipo de campo**:

| Key (Clave) | Value (Valor) | Tipo (Desplegable a la derecha) |
| :--- | :--- | :--- |
| `type` | `SNAPSHOT` | Text |
| `branchPath` | `MAIN` | Text |
| `createCodeSystemVersion` | `true` | Text |
| `file` | *(Selecciona tu archivo zip)* | **File** (Debes cambiar "Text" a "File" al pasar el ratón por la clave) |

**Detalle Crítico:**
- En la fila de `file`, el desplegable de tipo suele estar oculto hasta que pasas el ratón por encima de la celda "Key". Cámbialo a **File**.
- Luego dale al botón "Select Files" y busca tu `SnomedCT...zip`.

### 4. Ajuste TRUCO para evitar el error "Content-Type"
El error que te da (`charset=UTF-8 is not supported`) es porque Postman añade "extra" información que el servidor no entiende. Haz esto:

1.  Ve a la pestaña **Headers** de la petición.
2.  Busca la fila `Content-Type` (a veces está oculta, dale al botón "Hidden" o el ojo si no la ves).
3.  **DESMARCA** la casilla de `Content-Type`. Queremos que Postman lo genere solo, pero sin el "charset".
    - *Alternativa si sigue fallando:* Escribe manualmente en Headers:
        - Key: `Content-Type`
        - Value: `multipart/form-data` (sin nada más).
        - *Nota:* Esto a veces rompe el envío de archivos, prueba primero a desmarcarlo.

### 5. Enviar y Esperar

### 5. Monitorizar (Opcional pero recomendado)
Mientras Postman "piensa", puedes ver qué hace el servidor. Abre una terminal y escribe:
```bash
docker logs -f --tail 100 snowstorm-server
```
Verás mensajes como `Importing...`, `Loading...`, `Building index...`.

### Posibles Problemas y Soluciones
- **Error 400 Bad Request / Content-Type**:
  - Asegúrate de que en Postman NO has tocado nada en la pestaña `Headers` manualmente. Postman pone el `boundary` automáticamente.
  - Verifica que el campo `file` es de tipo **File** y no Text.
- **Error "Connection Refused"**:
  - Tu Docker se ha apagado. Ejecuta `docker-compose up -d` de nuevo.
- **Error de Memoria (Java Heap Space) en los logs**:
  - Significa que el archivo es muy grande. Reinicia Docker dándole más RAM si puedes, pero con los 4GB configurados debería bastar para la edición internacional.

## Paso 4: Verificar Importación

Una vez termine la importación, verifica que hay datos:
1.  Ve a `http://localhost:8080/browser/MAIN`
2.  Deberías ver conceptos como "SNOMED CT Concept".

## Paso 4: Ejecutar el Análisis

Una vez cargados los datos:
1.  Edita el script `analyze_ancestors_api.py`.
2.  Cambia la variable `USE_LOCAL` a `True`.
3.  Ejecuta: `python analyze_ancestors_api.py`.
