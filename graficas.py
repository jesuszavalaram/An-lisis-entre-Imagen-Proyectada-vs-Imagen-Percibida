from configuracion import *
def graficas_avances(ngramas, colores, tit, sent, fuentes, resumen_datos, registros, parametros, vectores, normales):
    print("Haciendo gráficas")
     
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for eje, (nombre, tabla) in zip(ax, ngramas.items()):
        tabla = tabla.sort_values("frecuencia")
        eje.barh(tabla["ngrama"], tabla["frecuencia"], color=colores(0.2))
        eje.set_title(nombre)
    fig.suptitle(tit + " - palabras en comentarios atípicos")
    plt.tight_layout()
     
    t = sent.reindex(fuentes)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(fuentes, t["pct_positivo"], label="positivo", color=colores(0.85))
    ax.barh(fuentes, t["pct_negativo"], left=t["pct_positivo"], label="negativo", color=colores(0.15))
    ax.set_xlim(0, 100)
    ax.set_xlabel("% de comentarios")
    ax.legend()
    ax.set_title(tit + " - sentimiento por fuente")
    plt.tight_layout()
     
    t = resumen_datos.reindex(fuentes)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(fuentes, t["analizados"], label="analizados", color=colores(0.15))
    ax.barh(fuentes, t["excluidos_otro_idioma"], left=t["analizados"], label="otro idioma", color=colores(0.5))
    ax.barh(fuentes, t["excluidos_vacios"], left=t["analizados"] + t["excluidos_otro_idioma"], label="vacíos", color=colores(0.85))
    ax.legend()
    ax.set_title("Registros por fuente")
    plt.tight_layout()
     
    fig, ax = plt.subplots(figsize=(10, 5))
    datos = [registros.loc[registros["fuente"] == f, "tokens"].str.split().str.len() for f in fuentes]
    ax.boxplot(datos, vert=False, boxprops=dict(color=colores(0.2)), medianprops=dict(color=colores(0.5)))
    ax.set_yticks(range(1, len(fuentes) + 1), fuentes)
    ax.set_title("Longitud de comentarios")
    ax.set_xlabel("Palabras")
    plt.tight_layout()
     
    fig, ax = plt.subplots(1, len(fuentes), figsize=(4 * len(fuentes), 5))
    for eje, f in zip(np.atleast_1d(ax), fuentes):
        palabras = pd.Series(" ".join(registros.loc[registros["fuente"] == f, "tokens"]).split()).value_counts().head(15)
        eje.barh(palabras.index[::-1], palabras.values[::-1], color=colores(0.2))
        eje.set_title(f)
    fig.suptitle("Palabras más frecuentes")
    plt.tight_layout()
     
    puntos = PCA(n_components=2, random_state=parametros.semilla).fit_transform(vectores)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(puntos[~registros["outlier"], 0], puntos[~registros["outlier"], 1], s=10, alpha=.3, label="normal", color=colores(0.3))
    ax.scatter(puntos[registros["outlier"], 0], puntos[registros["outlier"], 1], s=35, marker="x", label="atípico", color=colores(0.95))
    ax.legend()
    ax.set_title("Mapa de comentarios")
    plt.tight_layout()
     
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, f in enumerate(fuentes):
        p = normales.loc[normales["fuente"] == f, "polaridad"]
        ax.hist(p, bins=30, alpha=.5, label=f, color=colores(i / max(1, len(fuentes) - 1)))
    ax.axvline(parametros.umbral_polaridad, color="black", linestyle="--")
    ax.set_title("Distribución de polaridad")
    ax.legend()
    plt.tight_layout()
     
    palabras = CountVectorizer(token_pattern=r"\b[a-z]{3,}\b", min_df=5)
    X = palabras.fit_transform(normales["tokens"])
    vocab = palabras.get_feature_names_out()
    pos = np.asarray(X[(normales["sentimiento"] == "positivo").values].sum(axis=0)).ravel() + 1
    neg = np.asarray(X[(normales["sentimiento"] == "negativo").values].sum(axis=0)).ravel() + 1
    log_odds = np.log(pos / pos.sum()) - np.log(neg / neg.sum())
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for eje, ids, nombre in ((ax[0], np.argsort(log_odds)[:15], "negativas"),
                              (ax[1], np.argsort(log_odds)[-15:], "positivas")):
        eje.barh(vocab[ids], np.abs(log_odds[ids]), color=colores(0.15) if nombre == "negativas" else colores(0.85))
        eje.set_title("Palabras " + nombre)
    plt.tight_layout()
    return puntos
def grafica_topicos(topicos, colores):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for eje, (nombre, tabla) in zip(ax, topicos.items()):
        etiquetas = [", ".join(p.split(", ")[:4]) for p in tabla["palabras"]]
        eje.barh(etiquetas[::-1], tabla["porcentaje"][::-1], color=colores(0.85) if nombre == "positivo" else colores(0.15))
        eje.set_title(f"Tópicos en comentarios {nombre}s")
        eje.set_xlabel("% de comentarios")
    plt.tight_layout()
def graficas_comparacion(tabla_fuente, fuentes, pal, comparacion_sent, dmo, similitud, colores):
    fig, ax = plt.subplots(figsize=(10, 5))
    valores = tabla_fuente[fuentes].to_numpy()
    imagen = ax.imshow(valores, cmap=pal, aspect="auto")
    for (fila, col), v in np.ndenumerate(valores):
        ax.text(col, fila, f"{v:.1f}", ha="center", va="center",
                color="white" if v < valores.max() / 2 else "black")
    ax.set_xticks(range(len(fuentes)), fuentes)
    ax.set_yticks(range(len(tabla_fuente)), [", ".join(p.split(", ")[:4]) for p in tabla_fuente["palabras"]])
    fig.colorbar(imagen, ax=ax, label="% de textos")
    ax.set_title("Tópicos por fuente (NMF)")
    plt.tight_layout()
     
    t = comparacion_sent.drop(dmo)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].barh(t.index, t["diferencia_vs_dmo"], color=[colores(0.9) if d > 0 else colores(0.1) for d in t["diferencia_vs_dmo"]],
               hatch="//", edgecolor="black")
    ax[0].axvline(0, color="black")
    ax[0].set_xlabel("Puntos porcentuales (fuente − DMO)")
    ax[0].set_title("Diferencia de % positivo contra la DMO")
    ax[1].barh(similitud.index, similitud, color=colores(0.5), hatch="xx", edgecolor="black")
    ax[1].bar_label(ax[1].containers[0], fmt="%.3f")
    ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("Similitud coseno con la DMO")
    ax[1].set_title("Similitud de cada fuente con la DMO")
    plt.tight_layout()
def matriz_confusion(validacion, real, normales, mascara, pal):
    if validacion is not None:
        m = confusion_matrix(real, normales.loc[mascara, "sentimiento"], labels=["negativo", "positivo"])
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(m, cmap=pal)
        for (fila, col), v in np.ndenumerate(m):
            ax.text(col, fila, f"{v:,}", ha="center", va="center", color="white" if v < m.max() / 2 else "black")
        ax.set_xlabel("predicho")
        ax.set_ylabel("real")
        ax.set_xticks([0, 1], ["negativo", "positivo"])
        ax.set_yticks([0, 1], ["negativo", "positivo"])
        ax.set_title("Matriz de confusión")
        plt.tight_layout()
