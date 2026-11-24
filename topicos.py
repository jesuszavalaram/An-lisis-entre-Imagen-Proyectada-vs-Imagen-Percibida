from configuracion import *
def topicos_por_sentimiento(normales, parametros):
    topicos = {}
    normales["topico"] = ""
    filas_reporte = []
    for nombre in ["positivo", "negativo"]:
        mascara_s = normales["sentimiento"] == nombre
        textos = normales.loc[mascara_s, "tokens"]
        letra = nombre[0].upper()
        if len(textos) < parametros.minimo_topicos:
            frecuentes = pd.Series(" ".join(textos).split()).value_counts().head(10)
            topicos[nombre] = pd.DataFrame({"palabras": [", ".join(frecuentes.index)], "porcentaje": [100.0]})
            normales.loc[mascara_s, "topico"] = letra + "-frecuencias"
            print(f"Solo hay {len(textos)} comentarios {nombre}s, se usan las palabras mas frecuentes en lugar de NMF")
            continue
        tfidf_t = TfidfVectorizer(min_df=5, max_df=0.5)
        X_t = tfidf_t.fit_transform(textos)
        nmf = NMF(n_components=5, random_state=parametros.semilla)
        pesos = nmf.fit_transform(X_t)
        vocab_t = tfidf_t.get_feature_names_out()
        palabras = [", ".join(vocab_t[c.argsort()[::-1][:10]]) for c in nmf.components_]
        asignado = pesos.argmax(axis=1)
        porcentaje = pd.Series(asignado).value_counts(normalize=True).reindex(range(5), fill_value=0) * 100
        topicos[nombre] = pd.DataFrame({"palabras": palabras, "porcentaje": porcentaje.round(1)})
        normales.loc[mascara_s, "topico"] = [f"{letra}{i + 1}" for i in asignado]
        grupo = normales[mascara_s].reset_index(drop=True)
        for i in range(5):
            origen = grupo.loc[asignado == i, "fuente"].value_counts(normalize=True).head(3) * 100
            mejor = grupo.iloc[pesos[:, i].argmax()]
            filas_reporte.append({"sentimiento": nombre, "topico": f"{letra}{i + 1}", "palabras": palabras[i],
                                  "porcentaje": round(porcentaje[i], 1),
                                  "fuentes": ", ".join(f"{f} {v:.0f}%" for f, v in origen.items()),
                                  "fuente_ejemplo": mejor["fuente"], "comentario_representativo": mejor["texto_original"][:250]})
    reporte_topicos = pd.DataFrame(filas_reporte)
    for nombre, tabla in topicos.items():
        print(f"\nTópicos en comentarios {nombre}s:")
        print(tabla)
    return topicos, reporte_topicos
def comparar_fuentes(normales, parametros, dmo, sent, registros, vectores, fuentes):
    print("Comparando DMO vs percibido")
    tfidf_f = TfidfVectorizer(min_df=5, max_df=0.5)
    X_f = tfidf_f.fit_transform(normales["tokens"])
    nmf = NMF(n_components=5, random_state=parametros.semilla)
    pesos = nmf.fit_transform(X_f)
    vocab_f = tfidf_f.get_feature_names_out()
    tabla_fuente = pd.crosstab(pesos.argmax(axis=1), normales["fuente"].values, normalize="columns") * 100
    tabla_fuente.index.name, tabla_fuente.columns.name = "topico", None
    tabla_fuente = tabla_fuente.round(1)
    tabla_fuente["percibida"] = (pd.Series(pesos.argmax(axis=1)[(normales["fuente"] != dmo).values])
                                 .value_counts(normalize=True).reindex(tabla_fuente.index, fill_value=0) * 100).round(1)
    tabla_fuente.insert(0, "palabras", [", ".join(vocab_f[c.argsort()[::-1][:10]]) for c in nmf.components_])
    filas = []
    for f in tabla_fuente.columns[1:]:
        if f == dmo:
            continue
        for k in tabla_fuente.index:
            diferencia = tabla_fuente.loc[k, f] - tabla_fuente.loc[k, dmo]
            filas.append({"topico": k, "fuente": f, "dmo": tabla_fuente.loc[k, dmo], "percibido": tabla_fuente.loc[k, f],
                          "diferencia": round(diferencia, 1),
                          "resultado": "coincidencia" if abs(diferencia) <= 5 else "brecha"})
    comparacion_fuente = pd.DataFrame(filas)
    comparacion_sent = sent[["pct_positivo"]].copy()
    comparacion_sent["diferencia_vs_dmo"] = (sent["pct_positivo"] - sent.loc[dmo, "pct_positivo"]).round(2)
    centro = {f: vectores[i].mean(axis=0) for f, i in registros.groupby("fuente").indices.items()}
    similitud = pd.Series({f: np.dot(centro[dmo], c) / (np.linalg.norm(centro[dmo]) * np.linalg.norm(c))
                           for f, c in centro.items() if f != dmo}).round(3)
    es_dmo = (registros["fuente"] == dmo).values
    centro_dmo, centro_percibida = vectores[es_dmo].mean(axis=0), vectores[~es_dmo].mean(axis=0)
    indice_alineacion = round(float(np.dot(centro_dmo, centro_percibida) / (np.linalg.norm(centro_dmo) * np.linalg.norm(centro_percibida))), 3)
    tabla_por_fuente = pd.DataFrame({
        "topico_dominante": tabla_fuente[fuentes].idxmax().map(lambda k: ", ".join(tabla_fuente.loc[k, "palabras"].split(", ")[:4])),
        "pct_topico": tabla_fuente[fuentes].max(),
        "tono": np.where(sent.loc[fuentes, "pct_positivo"] >= 50, "positivo", "negativo"),
        "pct_positivo": sent.loc[fuentes, "pct_positivo"],
        "similitud_dmo": similitud.reindex(fuentes).fillna(1.0)})
    print("\nTópicos por fuente (% de textos):")
    print(tabla_fuente)
    print("\nCoincidencias y brechas por tópico:")
    print(comparacion_fuente)
    print("\nSentimiento comparado con la DMO:")
    print(comparacion_sent)
    print("\nSimilitud de cada fuente con la DMO:")
    print(similitud)
    print("\nIndice de alineacion DMO vs fuentes percibidas:", indice_alineacion)
    print("\nCada fuente frente a la DMO:")
    print(tabla_por_fuente)
    return tabla_fuente, comparacion_fuente, comparacion_sent, similitud, indice_alineacion, tabla_por_fuente
