import streamlit as st

st.set_page_config(page_title="Pronóstico Baby Planet", layout="wide")
st.title("📦 Pronóstico de demanda – Baby Planet")

# 1) importar librerías con control de error
try:
    import pandas as pd
    import numpy as np
    from prophet import Prophet
    import matplotlib.pyplot as plt
except Exception as e:
    st.error("⚠️ No se pudieron importar las librerías necesarias (pandas / prophet / matplotlib).")
    st.code(str(e))
    st.stop()

st.sidebar.header("📁 Fuente de datos")

uploaded_file = st.sidebar.file_uploader("Sube el archivo de ventas", type=["xlsx", "csv"])
from io import BytesIO


def df_to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Pronostico")
    return output.getvalue()


# 2) intentar leer archivo local si no suben nada
df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, parse_dates=["Fecha"])
        else:
            df = pd.read_excel(uploaded_file, parse_dates=["Fecha"])
        st.success("✅ Archivo cargado desde el navegador")
    except Exception as e:
        st.error("No pude leer el archivo que subiste.")
        st.code(str(e))
        st.stop()
else:
    # probar con el archivo que tienes en la carpeta
    try:
        df = pd.read_excel("ventas_babyplanet_sku_dinamica.xlsx", parse_dates=["Fecha"])
        st.info("📂 No subiste nada, así que estoy usando el archivo local `ventas_babyplanet_sku_dinamica.xlsx`.")
    except Exception as e:
        st.warning("No hay archivo subido y no encontré el archivo local. Sube uno para seguir.")
        st.stop()

# 3) validar columnas
cols_necesarias = {"Fecha", "Lineitem quantity", "Lineitem name"}
if not cols_necesarias.issubset(df.columns):
    st.error(f"Al archivo le faltan columnas. Debe tener al menos: {cols_necesarias}")
    st.write("Columnas que sí encontró:", list(df.columns))
    st.stop()

# 4) mostrar primera vista
st.subheader("👀 Vista previa de los datos")
st.dataframe(df.head())

# 5) seleccionar producto (Lineitem name)
productos = df["Lineitem name"].dropna().unique()
producto_sel = st.selectbox("🔎 Selecciona el producto a pronosticar", options=productos)

# 6) parámetro del horizonte
periods_usuario = st.number_input("Semanas a pronosticar", min_value=1, max_value=52, value=8)

if st.button("🚀 Generar pronóstico"):
    # ========== preparar serie del PRODUCTO (SEMANAL: semanas Lunes–Domingo, marcadas en domingo) ========== #
    df_prod = (
        df[df["Lineitem name"] == producto_sel]
        .groupby(pd.Grouper(key="Fecha", freq="W-SUN"))["Lineitem quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Fecha": "ds", "Lineitem quantity": "y"})
    )

    if df_prod.empty:
        st.error("Este producto no tiene datos suficientes.")
        st.stop()

    # reindexar semanas faltantes (todas las semanas con fin de semana domingo)
    rango = pd.date_range(df_prod["ds"].min(), df_prod["ds"].max(), freq="W-SUN")
    df_prod = (
        df_prod.set_index("ds")
        .reindex(rango)
        .reset_index()
        .rename(columns={"index": "ds"})
    )
    df_prod["y"] = df_prod["y"].fillna(0)

    # ordenar por fecha por si acaso
    df_prod = df_prod.sort_values("ds").reset_index(drop=True)

    # necesitamos suficientes puntos para hacer 80/20
    if len(df_prod) < 10:
        st.warning(
            "Muy pocos datos históricos para hacer una validación 80/20. "
            "Entrenaré con todo, pero el MAPE puede no ser representativo."
        )
        usar_split = False
    else:
        usar_split = True

    # ========== Paso 1: VALIDACIÓN 80/20 (MAPE interno) ========== #
    mape_val = None
    corte_fecha_validacion = None

    if usar_split:
        # índice de corte 80/20
        idx_corte = int(len(df_prod) * 0.8)
        train_80 = df_prod.iloc[:idx_corte].copy()
        test_20 = df_prod.iloc[idx_corte:].copy()

        corte_fecha_validacion = df_prod["ds"].iloc[idx_corte]

        try:
            m_val = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,  # datos ya agregados por semana
                daily_seasonality=False,
            )
            m_val.fit(train_80)

            # pronosticar hasta el final de la serie histórica (tantas semanas como test)
            future_val = m_val.make_future_dataframe(
                periods=len(test_20),
                freq="W-SUN",
            )
            forecast_val = m_val.predict(future_val)

            # calcular MAPE solo en el 20% más reciente
            comp = test_20.merge(
                forecast_val[["ds", "yhat"]],
                on="ds",
                how="left",
            )
            y_true = comp["y"].to_numpy(dtype=float)
            y_pred = comp["yhat"].to_numpy(dtype=float)
            mask = y_true != 0
            if mask.sum() > 0:
                mape_val = (
                    np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
                ).mean() * 100
        except Exception as e:
            st.warning("No se pudo calcular el MAPE con el split 80/20.")
            st.code(str(e))
            mape_val = None

    # ========== Paso 2: ENTRENAR CON EL 100% Y PRONOSTICAR FUTURO ========== #
    ultima_fecha_real = df_prod["ds"].max()

    try:
        m_full = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        m_full.fit(df_prod)
    except Exception as e:
        st.error("❌ Error al entrenar Prophet con el 100% de los datos.")
        st.code(str(e))
        st.stop()

    # pronóstico futuro (historia + periods_usuario semanas más)
    future_full = m_full.make_future_dataframe(
        periods=periods_usuario,
        freq="W-SUN",
    )
    forecast_full = m_full.predict(future_full)

    # calcular próximo domingo después del último dato real
    # (semana Lunes–Domingo, marcada en domingo)
    proximo_domingo = ultima_fecha_real + pd.offsets.Week(weekday=6)

    # ========== mostrar tabla de FUTURO REAL (semanas posteriores al último dato) ========== #
    st.subheader("📋 Pronóstico futuro (semanal)")

    futuro = forecast_full[forecast_full["ds"] >= proximo_domingo][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].rename(
        columns={
            "ds": "Semana",
            "yhat": "Demanda esperada",
            "yhat_lower": "Mín",
            "yhat_upper": "Máx",
        }
    )

    # para que se vea más limpio (solo fecha)
    futuro["Semana"] = futuro["Semana"].dt.date

    st.dataframe(futuro)

    # 👉 botón de descarga en Excel
    excel_bytes = df_to_excel_bytes(futuro)
    st.download_button(
        label="⬇️ Descargar pronóstico en Excel",
        data=excel_bytes,
        file_name=f"pronostico_{producto_sel}_semanal.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ========== mostrar MAPE de validación 80/20 ========== #
    if mape_val is not None:
        st.info(f"📏 MAPE (validación interna 80/20): **{mape_val:.2f}%**")
    else:
        st.info(
            "📏 No se pudo calcular un MAPE confiable (pocos datos o error en la validación)."
        )

    # ========== gráfico ========== #
    st.subheader("📈 Gráfico histórico + pronóstico (semanal)")
    fig, ax = plt.subplots(figsize=(10, 4))

    # histórico completo
    ax.plot(df_prod["ds"], df_prod["y"], color="black", label="Histórico")

    # pronóstico completo (historia + futuro)
    ax.plot(
        forecast_full["ds"],
        forecast_full["yhat"],
        color="red",
        linestyle="--",
        label="Pronóstico Prophet",
    )

    # banda de confianza
    ax.fill_between(
        forecast_full["ds"],
        forecast_full["yhat_lower"],
        forecast_full["yhat_upper"],
        color="red",
        alpha=0.15,
        label="Intervalo de confianza (95%)",
    )

    # línea vertical: inicio del 20% de test (validación) si usamos split
    if corte_fecha_validacion is not None:
        ax.axvline(
            corte_fecha_validacion,
            color="blue",
            linestyle="--",
            linewidth=2,
            label="Inicio período de validación (20% más reciente)",
        )

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Unidades vendidas")
    ax.set_title(f"Proyección de demanda semanal – {producto_sel}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)
