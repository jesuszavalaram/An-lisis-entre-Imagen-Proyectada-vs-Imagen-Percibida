from configuracion import *
def clasificar_sentimiento(parametros, normales):
    print("calculando sentimiento")
    analizador = create_analyzer(task="sentiment", lang=parametros.idioma)
    predicciones = [r.probas for r in analizador.predict([t[:1500] for t in normales["texto_original"]])]
    normales["polaridad"] = [p.get("POS", 0) - p.get("NEG", 0) for p in predicciones]
    normales["sentimiento"] = np.where(normales["polaridad"] >= parametros.umbral_polaridad, "positivo", "negativo")
    sent = normales.groupby("fuente").agg(n=("sentimiento", "size"),
                                          pct_positivo=("sentimiento", lambda s: (s == "positivo").mean() * 100),
                                          polaridad_media=("polaridad", "mean"))
    sent["pct_negativo"] = 100 - sent["pct_positivo"]
    por_rol = normales.groupby("rol")["sentimiento"].apply(lambda s: (s == "positivo").mean() * 100)
    sent = sent.round(2)
    columnas = ["fuente", "rol", "polaridad", "sentimiento", "texto_original"]
    positivos = normales.loc[normales["sentimiento"] == "positivo", columnas]
    negativos = normales.loc[normales["sentimiento"] == "negativo", columnas]
    return sent, por_rol, positivos, negativos
def validar_sentimiento(parametros, normales):
    validacion = None
    real, mascara = None, None
    if parametros.columna_validacion and parametros.columna_validacion in normales.columns:
        calificacion = pd.to_numeric(normales[parametros.columna_validacion], errors="coerce")
        medio = (calificacion.min() + calificacion.max()) / 2
        mascara = calificacion.notna() & (calificacion != medio)
        if mascara.sum():
            real = np.where(calificacion[mascara] > medio, "positivo", "negativo")
            validacion = pd.DataFrame(classification_report(real, normales.loc[mascara, "sentimiento"],
                                                            output_dict=True, zero_division=0)).T.round(3)
    return validacion, real, mascara
