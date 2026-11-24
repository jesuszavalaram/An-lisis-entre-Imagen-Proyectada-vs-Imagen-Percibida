from configuracion import *
def analizar_precio(parametros, svd, tfidf, vectores, registros, normales, fuentes, stemmer):
    print("Analizando precio, valor y costo")
    concepto = " ".join(stemmer.stem(w) for w in CONCEPTO_PRECIO[parametros.idioma].split())
    vector_concepto = normalize(svd.transform(tfidf.transform([concepto])))[0]
    normales["similitud_precio"] = vectores[~registros["outlier"].values] @ vector_concepto
    relacionados = normales[normales["similitud_precio"] >= parametros.umbral_precio]
    precio_por_fuente = pd.DataFrame({
        "comentarios": normales.groupby("fuente").size(),
        "sobre_precio": relacionados.groupby("fuente").size(),
        "pct_positivo_precio": relacionados.groupby("fuente")["sentimiento"].apply(lambda s: (s == "positivo").mean() * 100),
    }).reindex(fuentes).fillna(0)
    precio_por_fuente["pct_sobre_precio"] = precio_por_fuente["sobre_precio"] / precio_por_fuente["comentarios"] * 100
    precio_por_fuente = precio_por_fuente.round(1)
    contexto_precio = {}
    for rol, grupo in relacionados.groupby("rol"):
        conteo = pd.Series(" ".join(grupo["tokens"]).split()).value_counts()
        contexto_precio[rol] = ", ".join(conteo[~conteo.index.isin(CONCEPTO_PRECIO[parametros.idioma].split())].head(10).index)
    top_precio = normales.nlargest(5, "similitud_precio")[["fuente", "similitud_precio", "sentimiento", "texto_original"]]
    return relacionados, precio_por_fuente, contexto_precio, top_precio
