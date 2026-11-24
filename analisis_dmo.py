from configuracion import *
from configuracion import leer_parametros
from datos import cargar_datos, limpiar_textos
from atipicos import detectar_atipicos
from sentimiento import clasificar_sentimiento, validar_sentimiento
from graficas import graficas_avances, grafica_topicos, graficas_comparacion, matriz_confusion
from topicos import topicos_por_sentimiento, comparar_fuentes
from precio import analizar_precio
from interactivas import graficas_interactivas
from reporte import imprimir_resumen, crear_reporte
parametros = leer_parametros()
dmo = parametros.etiqueta_dmo.strip().lower()
registros = cargar_datos(parametros, dmo)
stemmer, registros, resumen_datos = limpiar_textos(parametros, registros, dmo)
tfidf, k, svd, vectores, atipicos, ngramas, atipicos_por_fuente, normales = detectar_atipicos(registros, parametros)
sent, por_rol, positivos, negativos = clasificar_sentimiento(parametros, normales)
validacion, real, mascara = validar_sentimiento(parametros, normales)
tit, pal = parametros.titulo, parametros.paleta
colores = plt.get_cmap(pal)
fuentes = [dmo] + sorted(f for f in registros["fuente"].unique() if f != dmo)
puntos = graficas_avances(ngramas, colores, tit, sent, fuentes, resumen_datos, registros, parametros, vectores, normales)
topicos, reporte_topicos = topicos_por_sentimiento(normales, parametros)
grafica_topicos(topicos, colores)
tabla_fuente, comparacion_fuente, comparacion_sent, similitud, indice_alineacion, tabla_por_fuente = comparar_fuentes(normales, parametros, dmo, sent, registros, vectores, fuentes)
graficas_comparacion(tabla_fuente, fuentes, pal, comparacion_sent, dmo, similitud, colores)
relacionados, precio_por_fuente, contexto_precio, top_precio = analizar_precio(parametros, svd, tfidf, vectores, registros, normales, fuentes, stemmer)
carpeta_html, tres, g, frases_html, contexto_html = graficas_interactivas(puntos, registros, normales, topicos, parametros, fuentes, resumen_datos, ngramas, sent, similitud, comparacion_sent, dmo, validacion, real, mascara, pal, tabla_fuente, precio_por_fuente, comparacion_fuente, contexto_precio, colores)
matriz_confusion(validacion, real, normales, mascara, pal)
imprimir_resumen(parametros, registros, normales, atipicos_por_fuente, ngramas, atipicos, sent, por_rol, positivos, negativos, validacion, reporte_topicos, tabla_fuente, dmo, indice_alineacion, precio_por_fuente, contexto_precio, top_precio)
crear_reporte(fuentes, colores, tres, reporte_topicos, topicos, tabla_fuente, top_precio, normales, dmo, relacionados, resumen_datos, tit, registros, parametros, g, sent, por_rol, atipicos_por_fuente, indice_alineacion, similitud, validacion, tabla_por_fuente, frases_html, precio_por_fuente, contexto_html, k, pal, carpeta_html)
datos_limpios = registros
outliers = atipicos[["fuente", "puntaje_outlier", "texto_original"]]
outliers_por_fuente = atipicos_por_fuente
unigramas, bigramas, trigramas = ngramas.values()
sentimiento_por_fuente = sent
topico_sentimiento = topicos
topico_fuente = tabla_fuente
similitud_fuente = similitud
comparacion_sentimiento = comparacion_sent
 
plt.show()
