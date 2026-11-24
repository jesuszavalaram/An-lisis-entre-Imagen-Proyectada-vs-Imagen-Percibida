from configuracion import *
def imprimir_resumen(parametros, registros, normales, atipicos_por_fuente, ngramas, atipicos, sent, por_rol, positivos, negativos, validacion, reporte_topicos, tabla_fuente, dmo, indice_alineacion, precio_por_fuente, contexto_precio, top_precio):
    print("\nResumen:", parametros.titulo)
    print("Comentarios:", len(registros), "| outliers:", int(registros["outlier"].sum()), "| normales:", len(normales))
    print("\nOutliers por fuente:")
    print(atipicos_por_fuente)
    for nombre, t in ngramas.items():
        print(f"\n{nombre} en outliers:", ", ".join(f"{g} ({f})" for g, f in t.head(10).values))
    print("\nEjemplos de outliers:")
    for _, r in atipicos.head(5).iterrows():
        print("-", r["fuente"], ":", r["texto_original"][:120])
    print("\nSentimiento por fuente:")
    print(sent)
    for rol, v in por_rol.items():
        print(f"{rol}: {v:.1f}% positivo")
    print("Positivos:", len(positivos), "| Negativos:", len(negativos))
    if validacion is not None:
        print(f"Validación con estrellas: exactitud {validacion.loc['accuracy', 'precision']:.3f}, "
              f"F1 {validacion.loc['macro avg', 'f1-score']:.3f}")
     
    print("\n===== REPORTE DE TOPICOS =====")
    for _, r in reporte_topicos.iterrows():
        print(f"\n[{r['topico']}] {r['sentimiento']} ({r['porcentaje']}% del grupo)")
        print("  Palabras clave:", r["palabras"])
        print("  Fuentes de origen:", r["fuentes"])
        print(f"  Comentario representativo ({r['fuente_ejemplo']}):", r["comentario_representativo"])
    print("\n===== BRECHAS ENTRE LA DMO Y LA IMAGEN PERCIBIDA =====")
    brechas = []
    for k in tabla_fuente.index:
        d, pc = tabla_fuente.loc[k, dmo], tabla_fuente.loc[k, "percibida"]
        if abs(pc - d) > 5:
            brechas.append(f"Topico {k} ({', '.join(tabla_fuente.loc[k, 'palabras'].split(', ')[:5])}): DMO {d}% vs percibida {pc}% -> "
                           + ("la DMO lo comunica mas" if d > pc else "los usuarios lo mencionan mas"))
    brechas.append(f"Sentimiento: DMO {sent.loc[dmo, 'pct_positivo']:.1f}% positivo vs percibida {por_rol.get('percibida', np.nan):.1f}% positivo")
    brechas.append(f"Indice de alineacion: {indice_alineacion}")
    print("\n".join("- " + b for b in brechas))
    print("\n===== PRECIO / VALOR / COSTO =====")
    print(precio_por_fuente)
    for rol, texto in contexto_precio.items():
        print(f"Contexto en {rol}:", texto)
    print("\nLos 5 comentarios mas cercanos al concepto:")
    for _, r in top_precio.iterrows():
        print(f"- [{r['fuente']}, {r['sentimiento']}, similitud {r['similitud_precio']:.3f}]", r["texto_original"][:200])
def crear_reporte(fuentes, colores, tres, reporte_topicos, topicos, tabla_fuente, top_precio, normales, dmo, relacionados, resumen_datos, tit, registros, parametros, g, sent, por_rol, atipicos_por_fuente, indice_alineacion, similitud, validacion, tabla_por_fuente, frases_html, precio_por_fuente, contexto_html, k, pal, carpeta_html):
    tabla = lambda df: df.to_html(classes="tabla", border=0)
    tarjeta = lambda titulo, valor, nota="": f"<div class='kpi'><span>{titulo}</span><b>{valor}</b><small>{nota}</small></div>"
    percibidas = [f for f in fuentes if f != dmo]
    c_oscuro, c_claro, c_pos, c_neg = to_hex(colores(0.0)), to_hex(colores(0.95)), tres[2], tres[0]
    topicos_html = "".join(
        f"<div class='topico {r.sentimiento}'><div class='etiqueta'>{r.topico} · {r.sentimiento} · {r.porcentaje}% del grupo</div>"
        f"<p class='chips'>{''.join(f'<span>{w}</span>' for w in r.palabras.split(', ')[:8])}</p>"
        f"<p class='origen'>Fuentes: {r.fuentes}</p><blockquote>{escape(r.comentario_representativo)}<cite>{r.fuente_ejemplo}</cite></blockquote></div>"
        for r in reporte_topicos.itertuples()) or "".join(f"<p><b>{n}</b>: {t.loc[0, 'palabras']}</p>" for n, t in topicos.items())
    brechas_html = "".join(
        f"<div class='brecha {'dmo' if tabla_fuente.loc[k, dmo] > tabla_fuente.loc[k, 'percibida'] else 'usuarios'}'>"
        f"<b>{', '.join(tabla_fuente.loc[k, 'palabras'].split(', ')[:5])}</b>"
        f"<span>DMO {tabla_fuente.loc[k, dmo]}% · percibida {tabla_fuente.loc[k, 'percibida']}%</span>"
        f"<em>{'La DMO lo comunica más' if tabla_fuente.loc[k, dmo] > tabla_fuente.loc[k, 'percibida'] else 'Los usuarios lo mencionan más'}</em></div>"
        for k in tabla_fuente.index if abs(tabla_fuente.loc[k, dmo] - tabla_fuente.loc[k, "percibida"]) > 5) or "<p>No hay brechas mayores a 5 puntos.</p>"
    precio_html = "".join(f"<blockquote>{escape(r.texto_original[:300])}<cite>{r.fuente} · {r.sentimiento} · similitud {r.similitud_precio:.3f}</cite></blockquote>"
                          for r in top_precio.itertuples())
    es_perc = normales["fuente"] != dmo
    pct_precio_perc = relacionados["fuente"].ne(dmo).sum() / max(1, es_perc.sum()) * 100
    tono_precio_perc = np.nan_to_num((relacionados.loc[relacionados["fuente"] != dmo, "sentimiento"] == "positivo").mean() * 100)
    total = int(resumen_datos["registros"].sum())
    pagina = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(tit)}</title><script>{get_plotlyjs()}</script><style>
:root{{--oscuro:{c_oscuro};--claro:{c_claro};--pos:{c_pos};--neg:{c_neg};--fondo:#f6f2ea;--texto:#1f2430;--suave:#5b6170}}
*{{box-sizing:border-box}}body{{margin:0;font-family:"Segoe UI",Arial,sans-serif;background:var(--fondo);color:var(--texto);line-height:1.5}}
header{{position:sticky;top:0;z-index:10;background:linear-gradient(120deg,var(--oscuro),#0f4c5c 60%,#1c6e5a);color:#fff;padding:18px 4vw 0;box-shadow:0 2px 10px rgba(0,0,0,.2)}}
header h1{{margin:0;font-size:clamp(22px,3vw,32px);line-height:1.2}}header p{{margin:4px 0 12px;opacity:.9;font-size:14px}}
nav{{display:flex;flex-wrap:wrap;gap:4px}}nav button{{border:0;background:transparent;color:#fff;padding:10px 16px;font:600 14px "Segoe UI",Arial,sans-serif;cursor:pointer;border-radius:10px 10px 0 0}}
nav button:hover{{background:rgba(255,255,255,.15)}}nav button.activa{{background:var(--fondo);color:var(--oscuro)}}
main{{max-width:1240px;margin:auto;padding:24px 4vw 40px}}section{{display:none}}section.activa{{display:block}}
h2{{font-size:26px;margin:0 0 6px;color:var(--oscuro);border-left:8px solid var(--claro);padding-left:12px}}.intro{{color:var(--suave);max-width:900px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:18px 0}}
.kpi{{background:#fff;border-radius:14px;padding:16px 18px;box-shadow:0 2px 10px rgba(0,0,0,.06);border-top:5px solid var(--oscuro)}}
.kpi span{{display:block;font-size:13px;color:var(--suave)}}.kpi b{{display:block;font-size:30px;color:var(--oscuro)}}.kpi small{{color:var(--suave)}}
.card{{background:#fff;border-radius:16px;padding:12px;box-shadow:0 2px 12px rgba(0,0,0,.07);margin:14px 0;min-width:0}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}}.grid3{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}}
.topicos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}}
.topico{{background:#fff;border-radius:14px;padding:14px 16px;box-shadow:0 2px 10px rgba(0,0,0,.06);border-left:7px solid var(--pos)}}.topico.negativo{{border-left-color:var(--neg)}}
.etiqueta{{font-weight:700;color:var(--oscuro)}}.chips span{{display:inline-block;background:var(--fondo);border-radius:12px;padding:2px 10px;margin:2px;font-size:13px}}
.origen{{font-size:13px;color:var(--suave);margin:4px 0}}blockquote{{margin:8px 0;padding:10px 14px;background:var(--fondo);border-radius:10px;font-style:italic;font-size:14px}}
cite{{display:block;font-style:normal;font-size:12px;color:var(--suave);margin-top:4px}}
.brechas{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin:14px 0}}
.brecha{{background:#fff;border-radius:14px;padding:14px;box-shadow:0 2px 10px rgba(0,0,0,.06);border-top:6px solid var(--pos)}}.brecha.dmo{{border-top-color:var(--oscuro)}}
.brecha b,.brecha span,.brecha em{{display:block}}.brecha span{{font-size:18px;font-weight:700;color:var(--oscuro)}}.brecha em{{color:var(--suave)}}
details{{margin:8px 0}}summary{{cursor:pointer;color:var(--oscuro);font-weight:600}}.tabla{{border-collapse:collapse;font-size:13px;background:#fff;margin:8px 0;display:block;overflow-x:auto}}
.tabla td,.tabla th{{border:1px solid #e3ddd2;padding:5px 9px;text-align:left}}.tabla th{{background:#efe9df}}
.metodo{{display:table;font-size:14px}}.metodo td{{vertical-align:top}}.metodo td:first-child{{font-weight:600;color:var(--oscuro);white-space:nowrap}}
@media (max-width:700px){{.grid2,.grid3{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>{escape(tit)}</h1><p>Imagen proyectada por la DMO ({dmo}) frente a la imagen percibida en {", ".join(percibidas)}</p>
<nav><button class="activa" data-s="datos">Datos iniciales</button><button data-s="comentarios">Comentarios analizados</button><button data-s="indices">Índices</button><button data-s="resultados">Resultados</button><button data-s="precio">Precio, valor y costo</button><button data-s="metodo">Metodología</button></nav></header><main>
<section id="datos" class="activa"><h2>1. Datos iniciales</h2><p class="intro">Registros recibidos por fuente y cuántos se analizaron después de limpiar el texto y quitar los que estaban en otro idioma o vacíos.</p>
<div class="kpis">{tarjeta("Registros recibidos", f"{total:,}")}{tarjeta("Fuentes", len(fuentes), ", ".join(fuentes))}{tarjeta("Analizados", f"{len(registros):,}", f"idioma: {parametros.idioma}")}{tarjeta("Excluidos", f"{total - len(registros):,}", "otro idioma o vacíos")}</div>
<div class="grid2"><div class="card">{g["registros"]}</div><div class="card">{g["longitud"]}</div></div><details><summary>Ver tabla</summary>{tabla(resumen_datos)}</details></section>
<section id="comentarios"><h2>2. Comentarios analizados</h2><p class="intro">Los comentarios atípicos se detectaron con Isolation Forest por fuente y se resumen con n-gramas. El resto se clasificó como positivo o negativo con pysentimiento.</p>
<div class="kpis">{tarjeta("Comentarios normales", f"{len(normales):,}")}{tarjeta("Atípicos", f"{int(registros['outlier'].sum()):,}", f"contaminacion {parametros.contaminacion}")}{tarjeta("Positivos DMO", f"{sent.loc[dmo, 'pct_positivo']:.1f}%")}{tarjeta("Positivos percibida", f"{por_rol.get('percibida', np.nan):.1f}%")}</div>
<div class="grid3">{"".join(f'<div class="card">{x}</div>' for x in g["ngramas"])}</div>
<div class="grid2"><div class="card">{g["sentimiento"]}</div><div class="card">{g["polaridad"]}</div></div><div class="card">{g["mapa_fuentes"]}</div>
<details><summary>Ver tablas</summary>{tabla(atipicos_por_fuente)}{tabla(sent)}</details></section>
<section id="indices"><h2>3. Índices</h2><p class="intro">Qué tan parecido es lo que dicen las fuentes percibidas a lo que dice la DMO, y qué tan confiable es la clasificación de sentimiento.</p>
<div class="kpis">{tarjeta("Índice de alineación", indice_alineacion, "DMO vs todas las percibidas")}{tarjeta("Fuente más parecida", similitud.idxmax(), f"similitud {similitud.max():.3f}")}{tarjeta("F1 (macro)", f"{validacion.loc['macro avg', 'f1-score']:.3f}" if validacion is not None else "-", "contra estrellas")}{tarjeta("Exactitud", f"{validacion.loc['accuracy', 'precision']:.3f}" if validacion is not None else "-", "contra estrellas")}</div>
<div class="grid2"><div class="card">{g["similitud"]}</div><div class="card">{g["diferencia"]}</div></div><div class="grid2"><div class="card">{g["confusion"]}</div><div class="card"><h3>Cada fuente frente a la DMO</h3>{tabla(tabla_por_fuente)}</div></div></section>
<section id="resultados"><h2>4. Resultados obtenidos</h2><p class="intro">Tópicos encontrados con NMF en comentarios positivos y negativos. El nombre de cada tópico es una interpretación de sus palabras clave.</p>
<h3>Brechas entre la imagen proyectada y la percibida</h3><div class="card">{frases_html}</div><div class="brechas">{brechas_html}</div>
<div class="card">{g["mapa_topicos"]}</div><div class="card">{g["brecha_fuente"]}</div><h3>Tópicos por sentimiento</h3><div class="topicos">{topicos_html}</div>
<div class="card">{g["scatter_topicos"]}</div><div class="card">{g["comparativa"]}</div></section>
<section id="precio"><h2>5. Precio, valor y costo</h2><p class="intro">Comentarios cercanos al concepto precio / valor / costo por similitud semántica (umbral {parametros.umbral_precio}).</p>
<div class="kpis">{tarjeta("DMO que habla de precio", f"{precio_por_fuente.loc[dmo, 'pct_sobre_precio']:.1f}%")}{tarjeta("Percibida que habla de precio", f"{pct_precio_perc:.1f}%")}{tarjeta("Tono positivo en percibida", f"{tono_precio_perc:.1f}%", "comentarios sobre precio")}</div>
<div class="grid2"><div class="card">{g["precio"]}</div><div class="card">{g["precio_fuente"]}</div></div>
<div class="grid2"><div class="card">{g["precio_tono"]}</div><div class="card"><h3>Contexto</h3>{contexto_html}</div></div>
<div class="card"><h3>Los 5 comentarios más cercanos al concepto</h3>{precio_html}</div></section>
<section id="metodo"><h2>6. Metodología</h2><p class="intro">Herramienta y parámetros usados para obtener cada parte del reporte.</p>
<table class="tabla metodo"><tr><th>Sección</th><th>Qué se obtuvo</th><th>Cómo se hizo</th></tr>
<tr><td>Datos iniciales</td><td>Limpieza del texto</td><td>Minúsculas, sin acentos, enlaces ni menciones. Stopwords de NLTK ({parametros.idioma}) y raíces con SnowballStemmer.</td></tr>
<tr><td>Datos iniciales</td><td>Idioma y excluidos</td><td>Idioma por conteo de stopwords (es, en, fr). Se excluyen textos en otro idioma o con menos de 3 palabras útiles.</td></tr>
<tr><td>Comentarios analizados</td><td>Representación de los textos</td><td>TF-IDF (min_df=2, max_df=0.5) reducido con LSA (TruncatedSVD, {k} componentes) y normalizado.</td></tr>
<tr><td>Comentarios analizados</td><td>Comentarios atípicos</td><td>Isolation Forest por fuente, contaminación {parametros.contaminacion}, 200 árboles, semilla {parametros.semilla}. Se resumen con unigramas, bigramas y trigramas (CountVectorizer).</td></tr>
<tr><td>Comentarios analizados</td><td>Sentimiento</td><td>pysentimiento (modelo preentrenado en español). Polaridad = P(positivo) − P(negativo); positivo si la polaridad es ≥ {parametros.umbral_polaridad}.</td></tr>
<tr><td>Comentarios analizados</td><td>Mapa de comentarios</td><td>PCA a 2 dimensiones sobre los vectores LSA. Se muestran hasta 3,000 comentarios por fuente.</td></tr>
<tr><td>Índices</td><td>Validación del sentimiento</td><td>Comparación con las estrellas ({parametros.columna_validacion}): 1–2 negativo, 4–5 positivo, 3 excluido. Exactitud, F1 macro y matriz de confusión.</td></tr>
<tr><td>Índices</td><td>Similitud e índice de alineación</td><td>Similitud coseno entre el vector promedio (LSA) de la DMO y el de cada fuente, o el de todas las percibidas juntas.</td></tr>
<tr><td>Resultados</td><td>Tópicos por sentimiento</td><td>TF-IDF (min_df=5, max_df=0.5) + NMF con 5 componentes y 10 palabras por tópico. Si un grupo tiene menos de {parametros.minimo_topicos} comentarios, se usan las palabras más frecuentes. El comentario representativo es el de mayor peso en el tópico.</td></tr>
<tr><td>Resultados</td><td>Tópicos por fuente y brechas</td><td>NMF con 5 componentes sobre todos los comentarios; % de textos de cada fuente por tópico. Brecha si la diferencia con la DMO es mayor a 5 puntos.</td></tr>
<tr><td>Precio, valor y costo</td><td>Comentarios sobre precio</td><td>El concepto «{CONCEPTO_PRECIO[parametros.idioma]}» se convierte en vector LSA y se mide la similitud coseno con cada comentario. Relacionado si la similitud es ≥ {parametros.umbral_precio}.</td></tr>
<tr><td>Todas</td><td>Gráficas</td><td>Plotly, sin conexión, con la paleta {pal}. Color y forma (o textura) para distinguir fuentes y sentimiento.</td></tr></table></section></main>
<script>document.querySelectorAll("nav button").forEach(b => b.onclick = () => {{
document.querySelectorAll("section, nav button").forEach(e => e.classList.remove("activa"));
document.getElementById(b.dataset.s).classList.add("activa"); b.classList.add("activa"); window.scrollTo(0, 0);
document.querySelectorAll("#" + b.dataset.s + " .js-plotly-plot").forEach(p => Plotly.Plots.resize(p));}});</script></body></html>"""
    (carpeta_html / "reporte.html").write_text(pagina, encoding="utf-8")

    carpeta_graficas = carpeta_html / "graficas"
    carpeta_graficas.mkdir(exist_ok=True)
    figuras = [(nombre, contenido) for nombre, contenido in g.items()
               for contenido in (contenido if isinstance(contenido, list) else [contenido])]
    for i, (nombre_g, fig_p) in enumerate(figuras, 1):
        titulo_g = re.sub(r"[^a-z0-9]+", "_", unicodedata.normalize("NFD", nombre_g.lower())
                          .encode("ascii", "ignore").decode()).strip("_")[:50]
        pagina_g = f"<!doctype html><html><head><meta charset='utf-8'><script>{get_plotlyjs()}</script></head><body>{fig_p}</body></html>"
        (carpeta_graficas / f"{i:02d}_{titulo_g}.html").write_text(pagina_g, encoding="utf-8")
    print("Gráficas individuales en:", carpeta_graficas)
    webbrowser.open((carpeta_html / "reporte.html").as_uri())
    print("\nReporte guardado en:", carpeta_html / "reporte.html")
