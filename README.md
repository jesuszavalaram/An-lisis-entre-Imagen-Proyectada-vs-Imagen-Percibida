# Imagen proyectada vs. imagen percibida de la CDMX

Autor: Jesús Ignacio Zavala Ramírez

Programa de línea de comandos en Python. Compara la imagen turística que proyecta la DMO (SECTUR) con la que perciben los usuarios en TripAdvisor, Instagram y Reddit. Funciona sin internet y sin APIs una vez instaladas las dependencias. Al terminar genera un reporte HTML interactivo.

## Estructura

| Archivo | Contenido |
|---|---|
| `analisis_dmo.py` | Programa principal: llama a las funciones en orden |
| `configuracion.py` | Imports, constantes y parámetros de la línea de comandos |
| `datos.py` | Carga del CSV, validaciones y limpieza de textos |
| `atipicos.py` | TF-IDF + LSA, outliers con Isolation Forest y n-gramas |
| `sentimiento.py` | Sentimiento con pysentimiento y validación contra estrellas |
| `topicos.py` | Tópicos por sentimiento (NMF) y comparación DMO vs. percibido |
| `precio.py` | Comentarios sobre precio, valor y costo por similitud semántica |
| `graficas.py` | Gráficas de matplotlib |
| `interactivas.py` | Gráficas interactivas de plotly |
| `reporte.py` | Resumen en consola y reporte HTML |

Todos los archivos deben estar en la misma carpeta. Solo se ejecuta `analisis_dmo.py`.

## Instalación

Se usa Python 3.13.13.

```
pip install -r requirements.txt
python -m nltk.downloader stopwords
```

Para usar la GPU (opcional), primero instala PyTorch con CUDA desde https://pytorch.org y después el `requirements.txt`. Sin GPU el programa funciona igual, solo tarda más en el sentimiento.

### Preparar la ejecución sin internet

La primera vez que se corre, pysentimiento descarga su modelo y lo guarda en caché. Por eso hay que ejecutar el programa una vez con internet. Después ya funciona sin conexión.

## Uso

```
python analisis_dmo.py
```

Sin argumentos usa `cdmx_unificado.csv` con los valores por defecto. Para otra base, los argumentos van en este orden (los que no se pongan toman su valor por defecto):

```
python analisis_dmo.py [csv] [columna_texto] [columna_fuente] [etiqueta_dmo] [idioma] [titulo] [paleta]
```

Ejemplo:

```
python analisis_dmo.py base_profesor.csv comentario plataforma sectur es
```

| Parámetro | Por defecto | Descripción |
|---|---|---|
| `csv` | `cdmx_unificado.csv` | Archivo de datos |
| `columna_texto` | `texto` | Columna con el comentario |
| `columna_fuente` | `fuente` | Columna con la fuente |
| `etiqueta_dmo` | `dmo` | Valor de la fuente que corresponde a la DMO |
| `idioma` | `es` | `es` o `en` |
| `titulo` | `CDMX: imagen proyectada vs percibida` | Título del reporte |
| `paleta` | `cividis` | `viridis`, `cividis`, `plasma`, `inferno`, `magma` o `turbo` |
| `--contaminacion` | `0.05` | Proporción esperada de outliers |
| `--umbral-polaridad` | `0` | Polaridad mínima para considerar un comentario positivo |
| `--columna-validacion` | `estrellas` | Columna de estrellas para validar el sentimiento (si no existe, se omite) |
| `--semilla` | `42` | Semilla aleatoria |
| `--minimo-topicos` | `50` | Mínimo de textos para usar NMF; con menos se usan frecuencias |
| `--umbral-precio` | `0.3` | Similitud mínima con el concepto precio/valor/costo |

En Spyder los argumentos se ponen en Ejecutar → Configuración por archivo → Opciones de línea de comandos.

## Datos

La base `cdmx_unificado.csv` no se incluye en el repositorio porque contiene comentarios obtenidos de TripAdvisor, Instagram y Reddit. Para ejecutar el programa, coloca el archivo en la misma carpeta que el código o usa cualquier CSV con el formato descrito abajo.

## Formato del CSV

- Separador `,` o `;` (se detecta solo), codificación UTF-8 o latin-1.
- Debe tener una columna de texto y una de fuente.
- La columna de fuente debe incluir la etiqueta de la DMO y al menos otra fuente.
- Si el archivo no cumple esto, el programa se detiene con un mensaje que dice qué falta.

## Resultados

- Resumen en consola: datos, outliers, sentimiento, tópicos, alineación y precio.
- `resultados/reporte.html`: reporte interactivo con las secciones Datos iniciales, Comentarios analizados, Índices, Resultados, Precio, valor y costo y Metodología. Se abre solo al terminar.
- `resultados/graficas/`: cada gráfica interactiva en su propio HTML.
- Ventanas de matplotlib con las gráficas de los avances.

## Métodos

| Etapa | Método |
|---|---|
| Limpieza | Minúsculas, sin enlaces, menciones ni acentos; stopwords de NLTK; stemming Snowball |
| Representación | TF-IDF reducido con LSA (TruncatedSVD) |
| Outliers | Isolation Forest por fuente |
| Sentimiento | pysentimiento (RoBERTuito para español, BERTweet para inglés); polaridad = P(positivo) − P(negativo) |
| Tópicos | NMF por sentimiento |
| Comparación | Proporción de tópicos por fuente, similitud coseno e índice de alineación |
| Precio | Similitud coseno con un vector del concepto precio/valor/costo |
