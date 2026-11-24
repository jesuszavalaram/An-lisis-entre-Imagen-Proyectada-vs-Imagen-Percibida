from configuracion import *
def graficas_interactivas(puntos, registros, normales, topicos, parametros, fuentes, resumen_datos, ngramas, sent, similitud, comparacion_sent, dmo, validacion, real, mascara, pal, tabla_fuente, precio_por_fuente, comparacion_fuente, contexto_precio, colores):
    print("Haciendo graficas interactivas")
    carpeta_html = CARPETA / "resultados"
    carpeta_html.mkdir(exist_ok=True)
    puntos_n = puntos[~registros["outlier"].values]
    interactivo = pd.DataFrame({
        "x": puntos_n[:, 0], "y": puntos_n[:, 1], "fuente": normales["fuente"], "sentimiento": normales["sentimiento"],
        "topico": normales["topico"], "polaridad": normales["polaridad"].round(2),
        "similitud_precio": normales["similitud_precio"].round(3),
        "comentario": normales["texto_original"].str.slice(0, 200).str.wrap(70).str.replace("\n", "<br>")})
    etiquetas = {f"{n[0].upper()}{i + 1}": f"{n[0].upper()}{i + 1}: " + ", ".join(t.loc[i, "palabras"].split(", ")[:3])
                 for n, t in topicos.items() for i in t.index}
    interactivo["topico"] = interactivo["topico"].map(lambda c: etiquetas.get(c, c))
    muestra = interactivo.sample(frac=1, random_state=parametros.semilla).groupby("fuente").head(3000)
    orden_topicos = sorted(interactivo["topico"].unique())
    colores_topico = [to_hex(colores(i / max(1, len(orden_topicos) - 1))) for i in range(len(orden_topicos))]
    colores_fuente = [to_hex(colores(i / max(1, len(fuentes) - 1))) for i in range(len(fuentes))]
    color_fuente = dict(zip(fuentes, colores_fuente))
    tres = [to_hex(colores(v)) for v in (0.15, 0.5, 0.85)]
    formas = ["circle", "x", "diamond", "square", "triangle-up", "cross", "star", "triangle-down"]
    hover = {"fuente": True, "sentimiento": True, "polaridad": True, "comentario": True, "x": False, "y": False}
    estilo = dict(template="plotly_white", font=dict(family="Segoe UI, Arial, sans-serif", size=13), title_font_size=16,
                  margin=dict(l=10, r=10, t=55, b=10), legend=dict(orientation="h", y=-0.18, title=None))
    graf = lambda f: f.update_layout(**estilo).to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})
    g = {}
    t = resumen_datos.reindex(fuentes)[["analizados", "excluidos_otro_idioma", "excluidos_vacios"]].rename_axis("fuente").reset_index()
    t = t.melt(id_vars="fuente", var_name="estado", value_name="registros")
    g["registros"] = graf(px.bar(t, x="registros", y="fuente", color="estado", pattern_shape="estado", orientation="h",
                                 color_discrete_sequence=tres, category_orders={"fuente": fuentes}, text_auto=True,
                                 title="Registros por fuente: analizados y excluidos"))
    t = registros[["fuente"]].assign(palabras=registros["tokens"].str.split().str.len())
    g["longitud"] = graf(px.box(t, x="palabras", y="fuente", color="fuente", log_x=True, points=False,
                                category_orders={"fuente": fuentes}, color_discrete_map=color_fuente,
                                title="Palabras por comentario (escala logarítmica)").update_layout(showlegend=False))
    g["ngramas"] = [graf(px.bar(t.sort_values("frecuencia"), x="frecuencia", y="ngrama", orientation="h", title=nombre,
                                color_discrete_sequence=[tres[0]], height=420).update_layout(yaxis_title=None))
                    for nombre, t in ngramas.items() if not t.empty]
    t = sent.reindex(fuentes)[["pct_positivo", "pct_negativo"]].rename_axis("fuente").reset_index()
    t = t.melt(id_vars="fuente", var_name="sentimiento", value_name="porcentaje")
    t["sentimiento"] = t["sentimiento"].str.replace("pct_", "")
    g["sentimiento"] = graf(px.bar(t, x="porcentaje", y="fuente", color="sentimiento", pattern_shape="sentimiento",
                                   orientation="h", text_auto=".1f", range_x=[0, 100], category_orders={"fuente": fuentes},
                                   color_discrete_map={"positivo": tres[2], "negativo": tres[0]},
                                   pattern_shape_map={"positivo": "", "negativo": "/"}, title="Sentimiento por fuente (%)"))
    g["polaridad"] = graf(px.histogram(normales, x="polaridad", color="fuente", histnorm="percent", barmode="overlay", nbins=30,
                                       opacity=0.55, category_orders={"fuente": fuentes}, color_discrete_map=color_fuente,
                                       title="Distribución de la polaridad (% de cada fuente)"))
    g["mapa_fuentes"] = graf(px.scatter(muestra, x="x", y="y", color="fuente", symbol="fuente", category_orders={"fuente": fuentes},
                                        color_discrete_map=color_fuente, symbol_sequence=formas, hover_data=hover | {"topico": True},
                                        opacity=0.6, height=560, title="Mapa de comentarios por fuente (color y forma)")
                             .for_each_trace(lambda traza: traza.update(marker_size=11 if traza.name == dmo else 5))
                             .update_layout(xaxis_title=None, yaxis_title=None))
    t = similitud.rename("similitud").rename_axis("fuente").reset_index()
    g["similitud"] = graf(px.bar(t, x="similitud", y="fuente", color="fuente", pattern_shape="fuente", orientation="h",
                                 text_auto=".3f", range_x=[0, 1], color_discrete_map=color_fuente,
                                 title="Similitud de cada fuente con la DMO (0 = distinta, 1 = igual)").update_layout(showlegend=False))
    t = comparacion_sent.drop(dmo).rename_axis("fuente").reset_index()
    t["comparacion"] = np.where(t["diferencia_vs_dmo"] > 0, "más positiva que la DMO", "menos positiva que la DMO")
    g["diferencia"] = graf(px.bar(t, x="diferencia_vs_dmo", y="fuente", color="comparacion", pattern_shape="comparacion",
                                  orientation="h", text_auto=".1f",
                                  color_discrete_map={"más positiva que la DMO": tres[2], "menos positiva que la DMO": tres[0]},
                                  title="Diferencia de % positivo contra la DMO (puntos)"))
    g["confusion"] = "<p>No hay columna de validación en estos datos.</p>"
    if validacion is not None:
        m = confusion_matrix(real, normales.loc[mascara, "sentimiento"], labels=["negativo", "positivo"])
        g["confusion"] = graf(px.imshow(m, x=["predicho negativo", "predicho positivo"], y=["real negativo", "real positivo"],
                                        text_auto=True, color_continuous_scale=pal, title="Validación con estrellas: matriz de confusión"))
    columnas_mapa = fuentes + ["percibida"]
    g["mapa_topicos"] = graf(px.imshow(tabla_fuente[columnas_mapa].values, x=columnas_mapa, text_auto=".1f", aspect="auto",
                                       y=[f"{k}: " + ", ".join(tabla_fuente.loc[k, "palabras"].split(", ")[:4]) for k in tabla_fuente.index],
                                       color_continuous_scale=pal, labels=dict(color="% de textos"),
                                       title="Tópicos por fuente (% de los textos de cada fuente)"))
    g["scatter_topicos"] = graf(px.scatter(muestra, x="x", y="y", color="topico", symbol="sentimiento", category_orders={"topico": orden_topicos},
                                           color_discrete_sequence=colores_topico, symbol_sequence=["circle", "x"], hover_data=hover,
                                           opacity=0.6, height=560, title="Comentarios por tópico (color) y sentimiento (forma)")
                                .update_traces(marker=dict(size=6)).update_layout(xaxis_title=None, yaxis_title=None))
    distribucion = (normales.groupby(["fuente", "sentimiento", "topico"]).size() /
                    normales.groupby("fuente").size()).mul(100).round(1).rename("porcentaje").reset_index()
    distribucion["topico"] = distribucion["topico"].map(lambda c: etiquetas.get(c, c))
    g["comparativa"] = graf(px.bar(distribucion, x="fuente", y="porcentaje", color="topico", pattern_shape="sentimiento",
                                   category_orders={"fuente": fuentes, "topico": orden_topicos}, color_discrete_sequence=colores_topico,
                                   pattern_shape_map={"positivo": "", "negativo": "/"}, height=520,
                                   title="Tópicos y sentimiento: DMO frente a fuentes percibidas").update_layout(yaxis_title="% de comentarios"))
    g["precio"] = graf(px.scatter(muestra, x="similitud_precio", y="polaridad", color="fuente", symbol="fuente",
                                  category_orders={"fuente": fuentes}, color_discrete_map=color_fuente, symbol_sequence=formas,
                                  hover_data=hover | {"similitud_precio": True}, opacity=0.6, height=520,
                                  title="Relación de cada comentario con precio / valor / costo")
                       .add_vline(x=parametros.umbral_precio, line_dash="dash", annotation_text="umbral")
                       .update_layout(xaxis_title="similitud con el concepto", yaxis_title="polaridad (-1 negativo, 1 positivo)"))
    t = precio_por_fuente.rename_axis("fuente").reset_index()
    g["precio_fuente"] = graf(px.bar(t, x="pct_sobre_precio", y="fuente", color="fuente", pattern_shape="fuente", orientation="h",
                                     text_auto=".1f", color_discrete_map=color_fuente, hover_data=["sobre_precio", "pct_positivo_precio"],
                                     category_orders={"fuente": fuentes},
                                     title="% de comentarios de cada fuente que hablan de precio").update_layout(showlegend=False))

    etiqueta_topico = {k: f"{k}: " + ", ".join(tabla_fuente.loc[k, "palabras"].split(", ")[:4]) for k in tabla_fuente.index}
    if not comparacion_fuente.empty:
        pivote = comparacion_fuente.pivot(index="topico", columns="fuente", values="diferencia")
        pivote.index = [etiqueta_topico[k] for k in pivote.index]
        g["brecha_fuente"] = graf(px.imshow(pivote.values, x=list(pivote.columns), y=list(pivote.index), text_auto=".1f",
                                            aspect="auto", color_continuous_scale=pal, color_continuous_midpoint=0,
                                            labels=dict(color="puntos"),
                                            title="Brecha por tópico: % de la fuente menos % de la DMO"))
    else:
        g["brecha_fuente"] = "<p>Se necesita al menos una fuente percibida para calcular brechas por fuente.</p>"
    t = precio_por_fuente.rename_axis("fuente").reset_index()
    g["precio_tono"] = graf(px.bar(t, x="pct_positivo_precio", y="fuente", color="fuente", pattern_shape="fuente", orientation="h",
                                   text_auto=".1f", range_x=[0, 100], color_discrete_map=color_fuente,
                                   category_orders={"fuente": fuentes},
                                   title="% positivo en los comentarios sobre precio").update_layout(showlegend=False))

    frases = []
    for k in tabla_fuente.index:
        d, pc = tabla_fuente.loc[k, dmo], tabla_fuente.loc[k, "percibida"]
        if abs(pc - d) > 5:
            tema = ", ".join(tabla_fuente.loc[k, "palabras"].split(", ")[:4])
            frases.append(f"<li>La DMO dedica {d}% de sus textos a «{tema}» y la imagen percibida {pc}%. "
                          + ("La DMO insiste más en este tema." if d > pc else "Los usuarios hablan más de este tema.") + "</li>")
    frases_html = "<ul>" + "".join(frases) + "</ul>" if frases else "<p>No hay brechas mayores a 5 puntos.</p>"
    contexto_html = "".join(f"<p><b>Contexto en {rol}:</b> {escape(txt)}</p>" for rol, txt in contexto_precio.items())
    return carpeta_html, tres, g, frases_html, contexto_html
