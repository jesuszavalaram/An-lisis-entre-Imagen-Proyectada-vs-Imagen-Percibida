from configuracion import *
def detectar_atipicos(registros, parametros):
    print("Buscando outliers")
    tfidf = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True)
    X = tfidf.fit_transform(registros["texto_proc"])
    k = max(2, min(100, X.shape[1] - 1, X.shape[0] - 1))
    svd = TruncatedSVD(n_components=k, random_state=parametros.semilla)
    vectores = normalize(svd.fit_transform(X))
    registros["outlier"] = False
    registros["puntaje_outlier"] = 0.0
    for fuente, indices in registros.groupby("fuente").indices.items():
        if len(indices) < 20:
            continue
        modelo = IsolationForest(contamination=parametros.contaminacion, random_state=parametros.semilla, n_estimators=200)
        registros.loc[indices, "outlier"] = modelo.fit_predict(vectores[indices]) == -1
        registros.loc[indices, "puntaje_outlier"] = -modelo.score_samples(vectores[indices])
    atipicos = registros[registros["outlier"]].sort_values("puntaje_outlier", ascending=False)
    ngramas = {}
    for n, nombre in ((1, "Unigramas"), (2, "Bigramas"), (3, "Trigramas")):
        textos = atipicos["tokens"][atipicos["tokens"].str.split().str.len() >= n]
        if textos.empty:
            ngramas[nombre] = pd.DataFrame(columns=["ngrama", "frecuencia"])
            continue
        vectorizador = CountVectorizer(ngram_range=(n, n), token_pattern=r"\b[a-z]{3,}\b")
        X = vectorizador.fit_transform(textos)
        frecuencias = np.asarray(X.sum(axis=0)).ravel()
        orden = np.argsort(-frecuencias)[:15]
        ngramas[nombre] = pd.DataFrame({"ngrama": vectorizador.get_feature_names_out()[orden], "frecuencia": frecuencias[orden]})
    atipicos_por_fuente = registros.groupby("fuente")["outlier"].agg(outliers="sum", proporcion="mean").round(3)
    normales = registros[~registros["outlier"]].reset_index(drop=True)
    return tfidf, k, svd, vectores, atipicos, ngramas, atipicos_por_fuente, normales
