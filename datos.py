from configuracion import *
def cargar_datos(parametros, dmo):
    print("limpiando datos")
    ruta = Path(parametros.csv)
    if not ruta.exists():
        ruta = CARPETA / ruta
    if not ruta.exists():
        raise SystemExit(f"No se encontro el archivo {parametros.csv}")
    with open(ruta, encoding="utf-8", errors="ignore") as archivo:
        primera = archivo.readline()
    separador = ";" if primera.count(";") > primera.count(",") else ","
    try:
        registros = pd.read_csv(ruta, sep=separador, low_memory=False)
    except UnicodeDecodeError:
        registros = pd.read_csv(ruta, sep=separador, low_memory=False, encoding="latin-1")
    faltan = [c for c in (parametros.columna_texto, parametros.columna_fuente) if c not in registros.columns]
    if faltan:
        raise SystemExit(f"No existen las columnas {faltan}. Columnas del archivo: {list(registros.columns)}")
    registros = registros.dropna(subset=[parametros.columna_texto, parametros.columna_fuente]).copy()
    registros["fuente"] = registros[parametros.columna_fuente].astype(str).str.strip().str.lower()
    registros["rol"] = np.where(registros["fuente"] == dmo, "proyectada (DMO)", "percibida")
    if dmo not in set(registros["fuente"]):
        raise SystemExit(f"No hay registros con la etiqueta '{dmo}'. Fuentes del archivo: {sorted(registros['fuente'].unique())}")
    if registros["fuente"].nunique() < 2:
        raise SystemExit("Se necesita al menos una fuente ademas de la DMO para comparar")
    registros = registros.reset_index(drop=True)
     
    return registros
def limpiar_textos(parametros, registros, dmo):
    vacias = VACIAS[parametros.idioma]
    stemmer = SnowballStemmer(IDIOMAS[parametros.idioma])
    registros["texto_original"] = registros[parametros.columna_texto].astype(str)
    idiomas_detectados, tokens, raices = [], [], []
    for texto in registros["texto_original"]:
        texto = ENLACE_RE.sub(" ", texto.lower())
        texto = MENCION_RE.sub(" ", texto)
        texto = ETIQUETA_RE.sub(r" \1 ", texto)
        texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
        texto = ESPACIOS_RE.sub(" ", NO_LETRAS_RE.sub(" ", texto)).strip()
        todas = texto.split()
        conteos = {i: sum(w in VACIAS[i] for w in todas) for i in ("es", "en", "fr")}
        orden = sorted(conteos, key=conteos.get, reverse=True)
        idioma = orden[0] if conteos[orden[0]] >= 2 and conteos[orden[0]] > conteos[orden[1]] else "indef"
        palabras = [w for w in todas if len(w) > 2 and w not in vacias]
        idiomas_detectados.append(idioma)
        tokens.append(" ".join(palabras))
        raices.append(" ".join(stemmer.stem(w) for w in palabras))
    registros["idioma_detectado"] = idiomas_detectados
    registros["tokens"] = tokens
    registros["texto_proc"] = raices
     
    vacio = registros["texto_proc"].str.split().str.len().fillna(0) < 3
    otro_idioma = ~registros["idioma_detectado"].isin([parametros.idioma, "indef"])
    resumen_datos = pd.DataFrame({
        "registros": registros.groupby("fuente").size(),
        "excluidos_otro_idioma": registros[otro_idioma].groupby("fuente").size(),
        "excluidos_vacios": registros[vacio & ~otro_idioma].groupby("fuente").size(),
    }).fillna(0).astype(int)
    resumen_datos["analizados"] = resumen_datos["registros"] - resumen_datos["excluidos_otro_idioma"] - resumen_datos["excluidos_vacios"]
    registros = registros[~vacio & ~otro_idioma].reset_index(drop=True)
    if dmo not in set(registros["fuente"]) or registros["fuente"].nunique() < 2:
        raise SystemExit("Despues de filtrar por idioma no quedan textos de la DMO o de otras fuentes, revisa el parametro idioma")
    print(resumen_datos)
    return stemmer, registros, resumen_datos
