import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set Matplotlib backend to 'agg' to avoid rendering issues in Streamlit
plt.switch_backend('agg')

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Análisis de Retornos Diarios, Semanales y Mensuales",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Función para aplanar las columnas MultiIndex
def flatten_columns(df, ticker):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[0]} {col[1]}" for col in df.columns]
    return df

# Función para descargar y comprimir datos
def download_data(ticker, start, end, compression='Daily'):
    try:
        # Descargar datos diarios de yfinance
        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            st.warning(f"No hay datos disponibles para el ticker **{ticker}** en el rango de fechas seleccionado.")
            return None
        
        # Aplanar columnas
        df = flatten_columns(df, ticker)

        # Mapear la compresión seleccionada a la regla de pandas
        if compression == 'Weekly':
            rule = 'W'  # Semanal
            df = df.resample(rule).agg({
                f'Open {ticker}': 'first',
                f'High {ticker}': 'max',
                f'Low {ticker}': 'min',
                f'Close {ticker}': 'last',
                f'Volume {ticker}': 'sum'
            })
        elif compression == 'Monthly':
            rule = 'M'  # Mensual
            df = df.resample(rule).agg({
                f'Open {ticker}': 'first',
                f'High {ticker}': 'max',
                f'Low {ticker}': 'min',
                f'Close {ticker}': 'last',
                f'Volume {ticker}': 'sum'
            })
        else:
            rule = 'D'  # Diario (sin cambios)
        
        st.write(f"**Columnas para {ticker} ({compression}):** {df.columns.tolist()}")
        return df
    except Exception as e:
        st.error(f"Error al descargar datos para el ticker **{ticker}**: {e}")
        return None

# Función para calcular retornos
def calculate_returns(data, price_column):
    returns = data[price_column].pct_change() * 100  # Retorno porcentual
    return returns

# Función para analizar la estrategia basada en percentiles de retornos
def analyze_returns_percentile_strategy(data, price_column, look_forward_days, low_percentile, high_percentile):
    if len(data) < look_forward_days + 1:
        st.error(f"El conjunto de datos es demasiado corto para analizar con {look_forward_days} días de proyección. Se necesitan al menos {look_forward_days + 1} días de datos.")
        return pd.DataFrame()

    # Calcular retornos
    data['Returns'] = calculate_returns(data, price_column)
    
    # Determinar los percentiles de retornos históricos
    returns_data = data['Returns'].dropna()
    if len(returns_data) < 10:  # Asegurarse de que haya suficientes datos
        st.warning("No hay suficientes datos de retornos para analizar.")
        return pd.DataFrame()
    
    low_threshold = np.percentile(returns_data, low_percentile)
    high_threshold = np.percentile(returns_data, high_percentile)
    
    # Identificar señales de compra y venta basadas en percentiles
    buy_signals = 0
    sell_signals = 0
    buy_successes = 0
    sell_successes = 0
    buy_gains = []
    sell_gains = []
    
    for i in range(len(data) - look_forward_days):
        current_return = data['Returns'].iloc[i]
        initial_price = data[price_column].iloc[i]
        future_prices = data[price_column].iloc[i:i + look_forward_days + 1]
        
        # Verificar que los datos sean válidos
        if pd.isna(current_return) or pd.isna(initial_price) or initial_price == 0 or future_prices.isna().any():
            continue
        
        # Señal de compra: retorno por debajo del percentil bajo
        if current_return <= low_threshold:
            buy_signals += 1
            future_max = future_prices.max()
            gain = (future_max - initial_price) / initial_price * 100
            buy_gains.append(gain)
            if future_max > initial_price:  # Éxito si el precio sube
                buy_successes += 1
        
        # Señal de venta: retorno por encima del percentil alto
        elif current_return >= high_threshold:
            sell_signals += 1
            future_min = future_prices.min()
            loss = (future_min - initial_price) / initial_price * 100
            sell_gains.append(loss)
            if future_min < initial_price:  # Éxito si el precio baja
                sell_successes += 1
    
    if buy_signals == 0 and sell_signals == 0:
        st.warning("No se encontraron señales de compra o venta con los percentiles seleccionados.")
        return pd.DataFrame()
    
    buy_success_rate = (buy_successes / buy_signals * 100) if buy_signals > 0 else 0
    sell_success_rate = (sell_successes / sell_signals * 100) if sell_signals > 0 else 0
    avg_buy_gain = np.mean(buy_gains) if buy_gains else 0
    avg_sell_gain = np.mean(sell_gains) if sell_gains else 0
    
    results = [{
        'Buy_Signals': buy_signals,
        'Buy_Success_Rate (%)': buy_success_rate,
        'Avg_Buy_Gain (%)': avg_buy_gain,
        'Sell_Signals': sell_signals,
        'Sell_Success_Rate (%)': sell_success_rate,
        'Avg_Sell_Gain (%)': avg_sell_gain
    }]
    
    return pd.DataFrame(results)

# Título de la aplicación
st.title("📈 Análisis de Retornos Diarios, Semanales y Mensuales")
st.markdown("### 🚀 Sigue nuestro trabajo en [Twitter](https://twitter.com/MTaurus_ok)")

# Crear pestañas
tab1, tab2 = st.tabs(["Análisis de Retornos", "Análisis de Trading con Percentiles de Retornos"])

# Pestaña 1: Análisis de Retornos
with tab1:
    ticker = st.text_input("🖊️ Ingrese el símbolo del ticker", value="AAPL", key="ticker_original").upper()
    
    if ticker:
        start_date = st.date_input(
            "📅 Seleccione la fecha de inicio",
            value=pd.to_datetime('2000-01-01'),
            min_value=pd.to_datetime('1900-01-01'),
            max_value=pd.to_datetime('today'),
            key="start_original"
        )
        end_date = st.date_input(
            "📅 Seleccione la fecha de fin",
            value=pd.to_datetime('today') + pd.DateOffset(days=1),
            min_value=pd.to_datetime('1900-01-01'),
            max_value=pd.to_datetime('today') + pd.DateOffset(days=1),
            key="end_original"
        )
        compression = st.selectbox("📅 Seleccione la compresión de datos", ["Daily", "Weekly", "Monthly"], key="compression_original")
        apply_ratio = st.checkbox("🔄 Ajustar precio por el ratio YPFD.BA/YPF (CCL)", key="ratio_original")

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        if start_date > end_date:
            st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
        else:
            data = download_data(ticker, start_date, end_date, compression=compression)

            if data is not None:
                close_col_main = f"Close {ticker}"

                if apply_ratio:
                    st.subheader("🔄 Aplicando ajuste por ratio YPFD.BA/YPF (CCL)")
                    ypfd_ba_ticker = "YPFD.BA"
                    ypf_ticker = "YPF"
                    ypfd_ba_data = download_data(ypfd_ba_ticker, start_date, end_date, compression=compression)
                    ypf_data = download_data(ypf_ticker, start_date, end_date, compression=compression)

                    if ypfd_ba_data is not None and ypf_data is not None:
                        close_col_ypfd = f"Close {ypfd_ba_ticker}"
                        close_col_ypf = f"Close {ypf_ticker}"

                        if close_col_ypfd in ypfd_ba_data.columns and close_col_ypf in ypf_data.columns:
                            ypfd_ba_data = ypfd_ba_data.fillna(method='ffill').fillna(method='bfill')
                            ypf_data = ypf_data.fillna(method='ffill').fillna(method='bfill')
                            ratio = ypfd_ba_data[close_col_ypfd] / ypf_data[close_col_ypf]
                            ratio = ratio.reindex(data.index).fillna(method='ffill').fillna(method='bfill')

                            data['Close Ajustado'] = data[close_col_main] / ratio
                        else:
                            st.error(f"No se encontraron columnas de precio válidas para {ypfd_ba_ticker} o {ypf_ticker}.")
                    else:
                        st.error("No se pudieron descargar los datos necesarios para aplicar el ratio.")
                else:
                    data['Close Original'] = data[close_col_main]

                price_column = (
                    'Close Ajustado' if (apply_ratio and 'Close Ajustado' in data.columns)
                    else 'Close Original' if 'Close Original' in data.columns
                    else close_col_main
                )

                if price_column not in data.columns:
                    st.error(f"La columna **{price_column}** no existe en los datos.")
                else:
                    # Calcular retornos
                    data['Returns'] = calculate_returns(data, price_column)

                    # Visualización 1: Precio Histórico
                    st.write(f"### 📈 Precio Histórico ({compression})")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name='Precio de Cierre'))
                    fig.add_annotation(text="MTaurus. X: mtaurus_ok", xref="paper", yref="paper", x=0.95, y=0.05, showarrow=False, font=dict(size=14, color="gray"), opacity=0.5)
                    fig.update_layout(
                        title=f"Precio Histórico de {ticker} ({compression})",
                        xaxis_title="Fecha", yaxis_title="Precio (USD)", legend_title="Leyenda", template="plotly_dark", hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Visualización 2: Retornos Históricos
                    st.write(f"### 📉 Retornos Históricos ({compression})")

                    if data['Returns'].dropna().empty:
                        st.error("No hay datos válidos de retornos para graficar.")
                    else:
                        fig_returns = go.Figure()
                        fig_returns.add_trace(go.Scatter(
                            x=data.index, 
                            y=data['Returns'], 
                            mode='lines', 
                            name='Retornos (%)',
                            line=dict(color='lightgrey')
                        ))

                        # Línea de promedio histórico
                        historical_mean = data['Returns'].mean()
                        if not pd.isna(historical_mean):
                            fig_returns.add_shape(
                                type="line", 
                                x0=data.index.min(), 
                                x1=data.index.max(), 
                                y0=historical_mean, 
                                y1=historical_mean,
                                line=dict(color="lightblue", width=1, dash="dash"),
                            )
                            fig_returns.add_trace(go.Scatter(
                                x=[None], y=[None], mode='lines',
                                line=dict(color="lightblue", width=1, dash="dash"),
                                name=f"Promedio: {historical_mean:.2f}%",
                                showlegend=True,
                                opacity=0
                            ))
                            fig_returns.add_annotation(
                                x=data.index.max(), 
                                y=historical_mean, 
                                text=f"Promedio: {historical_mean:.2f}%",
                                showarrow=True, 
                                arrowhead=1, 
                                ax=20, 
                                ay=-20, 
                                font=dict(color="lightblue")
                            )

                        # Percentiles dinámicos
                        lower_percentile = st.slider("Seleccione el percentil inferior", min_value=1, max_value=49, value=5, key="lower_percentile")
                        upper_percentile = st.slider("Seleccione el percentil superior", min_value=51, max_value=99, value=95, key="upper_percentile")

                        returns_data = data['Returns'].dropna()
                        lower_value = np.percentile(returns_data, lower_percentile)
                        upper_value = np.percentile(returns_data, upper_percentile)

                        # Línea de percentil inferior
                        fig_returns.add_shape(
                            type="line", 
                            x0=data.index.min(), 
                            x1=data.index.max(), 
                            y0=lower_value, 
                            y1=lower_value,
                            line=dict(color="red", width=1, dash="dash"),
                        )
                        fig_returns.add_trace(go.Scatter(
                            x=[None], y=[None], mode='lines',
                            line=dict(color="red", width=1, dash="dash"),
                            name=f"P{lower_percentile}: {lower_value:.2f}%",
                            showlegend=True,
                            opacity=0
                        ))
                        fig_returns.add_annotation(
                            x=data.index.max(), 
                            y=lower_value, 
                            text=f"P{lower_percentile}: {lower_value:.2f}%",
                            showarrow=True, 
                            arrowhead=1, 
                            ax=20, 
                            ay=20, 
                            font=dict(color="red")
                        )

                        # Línea de percentil superior
                        fig_returns.add_shape(
                            type="line", 
                            x0=data.index.min(), 
                            x1=data.index.max(), 
                            y0=upper_value, 
                            y1=upper_value,
                            line=dict(color="green", width=1, dash="dash"),
                        )
                        fig_returns.add_trace(go.Scatter(
                            x=[None], y=[None], mode='lines',
                            line=dict(color="green", width=1, dash="dash"),
                            name=f"P{upper_percentile}: {upper_value:.2f}%",
                            showlegend=True,
                            opacity=0
                        ))
                        fig_returns.add_annotation(
                            x=data.index.max(), 
                            y=upper_value, 
                            text=f"P{upper_percentile}: {upper_value:.2f}%",
                            showarrow=True, 
                            arrowhead=1, 
                            ax=20, 
                            ay=-20, 
                            font=dict(color="green")
                        )

                        # Línea cero
                        fig_returns.add_shape(
                            type="line", 
                            x0=data.index.min(), 
                            x1=data.index.max(), 
                            y0=0, 
                            y1=0, 
                            line=dict(color="red", width=2)
                        )
                        fig_returns.add_trace(go.Scatter(
                            x=[None], y=[None], mode='lines',
                            line=dict(color="red", width=2),
                            name="Línea Cero",
                            showlegend=True,
                            opacity=0
                        ))

                        fig_returns.add_annotation(
                            text="MTaurus. X: mtaurus_ok", 
                            xref="paper", 
                            yref="paper", 
                            x=0.95, 
                            y=0.05,
                            showarrow=False, 
                            font=dict(size=14, color="gray"), 
                            opacity=0.5
                        )

                        fig_returns.update_layout(
                            title=f"Retornos Históricos de {ticker} ({compression})",
                            xaxis_title="Fecha", 
                            yaxis_title="Retornos (%)", 
                            legend_title="Leyenda",
                            template="plotly_dark", 
                            hovermode="x unified",
                            showlegend=True
                        )

                        st.plotly_chart(fig_returns, use_container_width=True)

                    # Visualización 3: Histograma con Seaborn/Matplotlib
                    st.write(f"### 📊 Histograma de Retornos con Percentiles ({compression})")
                    percentiles = [95, 85, 75, 50, 25, 15, 5]
                    percentile_values = np.percentile(data['Returns'].dropna(), percentiles)
                    
                    # Input para fechas específicas
                    st.write("#### Seleccionar fechas específicas para destacar en el histograma")
                    num_dates = st.number_input("Número de fechas a destacar", min_value=0, max_value=10, value=0, key="num_dates_hist")
                    selected_dates = []
                    returns_values = []
                    if num_dates > 0:
                        for i in range(num_dates):
                            date = st.date_input(
                                f"Seleccione la fecha {i+1}",
                                value=data.index[-1],
                                min_value=data.index[0],
                                max_value=data.index[-1],
                                key=f"hist_date_{i}"
                            )
                            date = pd.to_datetime(date)
                            if date in data.index:
                                selected_dates.append(date)
                                ret_value = data.loc[date, 'Returns']
                                if not pd.isna(ret_value):
                                    returns_values.append(ret_value)
                                else:
                                    st.warning(f"No hay datos de retornos para la fecha {date.strftime('%Y-%m-%d')}.")
                            else:
                                st.warning(f"La fecha {date.strftime('%Y-%m-%d')} no está en el rango de datos.")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(data['Returns'].dropna(), kde=True, color='blue', bins=100, ax=ax)
                    for percentile, value in zip(percentiles, percentile_values):
                        ax.axvline(value, color='red', linestyle='--')
                        ax.text(value, ax.get_ylim()[1] * 0.9, f'{percentile}º Percentil', color='red', rotation='vertical', verticalalignment='center', horizontalalignment='right')
                    for date, ret_value in zip(selected_dates, returns_values):
                        ax.axvline(ret_value, color='green', linestyle='-', alpha=0.5)
                        ax.text(ret_value, ax.get_ylim()[1] * 0.95, f"{date.strftime('%Y-%m-%d')}\n{ret_value:.2f}%", 
                                color='green', rotation='vertical', verticalalignment='center', horizontalalignment='left')
                    ax.text(0.95, 0.05, "MTaurus. X: mtaurus_ok", fontsize=14, color='gray', ha='right', va='center', alpha=0.5, transform=fig.transFigure)
                    ax.set_title(f'Retornos de {ticker} ({compression})')
                    ax.set_xlabel('Retornos (%)')
                    ax.set_ylabel('Frecuencia')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Visualización 4: Histograma Personalizable con Plotly
                    st.write(f"### 🎨 Personalización del Histograma ({compression})")
                    num_bins = st.slider("Seleccione el número de bins para el histograma", min_value=10, max_value=100, value=50, key="bins_original")
                    hist_color = st.color_picker("Elija un color para el histograma", value='#1f77b4', key="color_original")
                    st.write(f"### 📊 Histograma de Retornos ({compression})")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(x=data['Returns'].dropna(), nbinsx=num_bins, marker_color=hist_color, opacity=0.75, name="Histograma"))
                    for percentile, value in zip(percentiles, percentile_values):
                        fig_hist.add_vline(x=value, line=dict(color="red", width=2, dash="dash"), 
                                           annotation_text=f'{percentile}º Percentil', annotation_position="top", 
                                           annotation=dict(textangle=-90, font=dict(color="red")))
                    for date, ret_value in zip(selected_dates, returns_values):
                        fig_hist.add_vline(x=ret_value, line=dict(color="green", width=2, dash="solid"), 
                                           annotation_text=f"{date.strftime('%Y-%m-%d')}\n{ret_value:.2f}%", 
                                           annotation_position="top", annotation=dict(textangle=-90, font=dict(color="green")))
                    fig_hist.add_annotation(text="MTaurus. X: mtaurus_ok", xref="paper", yref="paper", x=0.95, y=0.05, 
                                            showarrow=False, font=dict(size=14, color="gray"), opacity=0.5)
                    fig_hist.update_layout(
                        title=f'Histograma de Retornos de {ticker} ({compression})',
                        xaxis_title='Retornos (%)', yaxis_title='Frecuencia', bargap=0.1, 
                        template="plotly_dark", hovermode="x unified"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("⚠️ Por favor, ingrese un símbolo de ticker válido para comenzar el análisis.")

# Pestaña 2: Análisis de Trading con Percentiles de Retornos
with tab2:
    st.header("Análisis de Trading con Percentiles de Retornos")
    
    st.markdown("""
    ### ¿Qué hace esta pestaña?
    Esta herramienta evalúa una estrategia de trading basada en los percentiles de los retornos históricos:
    - **Señales de Compra**: Cuando el retorno cae por debajo de un percentil bajo (e.g., 5º percentil), indicando un movimiento bajista extremo que podría preceder una recuperación.
    - **Señales de Venta**: Cuando el retorno sube por encima de un percentil alto (e.g., 95º percentil), indicando un movimiento alcista extremo que podría preceder una caída.
    Para cada señal, calculamos:
    - La **tasa de éxito** de las señales de compra (qué tan seguido el precio sube después de una señal) y de venta (qué tan seguido el precio baja).
    - La **ganancia promedio** después de una señal de compra y la **pérdida promedio** después de una señal de venta (en los próximos N períodos).
    Esto te ayuda a evaluar si los extremos en los retornos pueden ser usados para generar señales de trading confiables.
    """)

    ticker_ma = st.text_input("🖊️ Ingrese el símbolo del ticker", value="AAPL", key="ticker_ma").upper()
    
    if ticker_ma:
        start_date_ma = st.date_input(
            "📅 Seleccione la fecha de inicio",
            value=pd.to_datetime('2000-01-01'),
            min_value=pd.to_datetime('1900-01-01'),
            max_value=pd.to_datetime('today'),
            key="start_ma"
        )
        end_date_ma = st.date_input(
            "📅 Seleccione la fecha de fin",
            value=pd.to_datetime('today') + pd.DateOffset(days=1),
            min_value=pd.to_datetime('1900-01-01'),
            max_value=pd.to_datetime('today') + pd.DateOffset(days=1),
            key="end_ma"
        )
        look_forward_days = st.number_input("Períodos de proyección (N períodos después de la señal)", min_value=1, value=5, key="look_forward_days")
        low_percentile = st.slider("Percentil bajo para señales de compra", min_value=1, max_value=49, value=5, key="low_percentile_ma")
        high_percentile = st.slider("Percentil alto para señales de venta", min_value=51, max_value=99, value=95, key="high_percentile_ma")
        compression_ma = st.selectbox("📅 Seleccione la compresión de datos", ["Daily", "Weekly", "Monthly"], key="compression_ma")
        apply_ratio_ma = st.checkbox("🔄 Ajustar precio por el ratio YPFD.BA/YPF (CCL)", key="ratio_ma")

        start_date_ma = pd.to_datetime(start_date_ma)
        end_date_ma = pd.to_datetime(end_date_ma)

        if start_date_ma > end_date_ma:
            st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
        else:
            if st.button("Confirmar Análisis", key="confirm_ma"):
                data_ma = download_data(ticker_ma, start_date_ma, end_date_ma, compression=compression_ma)

                if data_ma is not None:
                    close_col_main = f"Close {ticker_ma}"

                    if apply_ratio_ma:
                        st.subheader("🔄 Aplicando ajuste por ratio YPFD.BA/YPF (CCL)")
                        ypfd_ba_ticker = "YPFD.BA"
                        ypf_ticker = "YPF"
                        ypfd_ba_data = download_data(ypfd_ba_ticker, start_date_ma, end_date_ma, compression=compression_ma)
                        ypf_data = download_data(ypf_ticker, start_date_ma, end_date_ma, compression=compression_ma)

                        if ypfd_ba_data is not None and ypf_data is not None:
                            close_col_ypfd = f"Close {ypfd_ba_ticker}"
                            close_col_ypf = f"Close {ypf_ticker}"

                            if close_col_ypfd in ypfd_ba_data.columns and close_col_ypf in ypf_data.columns:
                                ypfd_ba_data = ypfd_ba_data.fillna(method='ffill').fillna(method='bfill')
                                ypf_data = ypf_data.fillna(method='ffill').fillna(method='bfill')
                                ratio = ypfd_ba_data[close_col_ypfd] / ypf_data[close_col_ypf]
                                ratio = ratio.reindex(data_ma.index).fillna(method='ffill').fillna(method='bfill')

                                data_ma['Close Ajustado'] = data_ma[close_col_main] / ratio
                            else:
                                st.error(f"No se encontraron columnas de precio válidas para {ypfd_ba_ticker} o {ypf_ticker}.")
                        else:
                            st.error("No se pudieron descargar los datos necesarios para aplicar el ratio.")
                    else:
                        data_ma['Close Original'] = data_ma[close_col_main]

                    price_column_ma = (
                        'Close Ajustado' if (apply_ratio_ma and 'Close Ajustado' in data_ma.columns)
                        else 'Close Original' if 'Close Original' in data_ma.columns
                        else close_col_main
                    )

                    if price_column_ma not in data_ma.columns:
                        st.error(f"La columna **{price_column_ma}** no existe en los datos.")
                    else:
                        trading_df = analyze_returns_percentile_strategy(
                            data_ma, price_column_ma, look_forward_days, low_percentile, high_percentile
                        )

                        if not trading_df.empty:
                            st.write("### Resultados del Análisis de Trading")
                            st.markdown("""
                            Aquí tienes los resultados:
                            - **Buy_Signals**: Cuántas veces el retorno cayó por debajo del percentil bajo (señal de compra).
                            - **Buy_Success_Rate (%)**: Porcentaje de señales de compra que resultaron en un aumento del precio.
                            - **Avg_Buy_Gain (%)**: Ganancia promedio después de una señal de compra (en los próximos N períodos).
                            - **Sell_Signals**: Cuántas veces el retorno subió por encima del percentil alto (señal de venta).
                            - **Sell_Success_Rate (%)**: Porcentaje de señales de venta que resultaron en una caída del precio.
                            - **Avg_Sell_Gain (%)**: Pérdida promedio después de una señal de venta (en los próximos N períodos).
                            """)
                            st.dataframe(trading_df)

                            # Visualización: Resultados de la Estrategia
                            fig_trading = go.Figure()
                            fig_trading.add_trace(go.Bar(
                                x=['Compra'],
                                y=[trading_df['Buy_Success_Rate (%)'].iloc[0]],
                                name='Tasa de Éxito Compra (%)',
                                marker_color='green'
                            ))
                            fig_trading.add_trace(go.Bar(
                                x=['Venta'],
                                y=[trading_df['Sell_Success_Rate (%)'].iloc[0]],
                                name='Tasa de Éxito Venta (%)',
                                marker_color='red'
                            ))
                            fig_trading.add_trace(go.Bar(
                                x=['Compra'],
                                y=[trading_df['Avg_Buy_Gain (%)'].iloc[0]],
                                name='Ganancia Promedio Compra (%)',
                                marker_color='blue',
                                opacity=0.5
                            ))
                            fig_trading.add_trace(go.Bar(
                                x=['Venta'],
                                y=[trading_df['Avg_Sell_Gain (%)'].iloc[0]],
                                name='Pérdida Promedio Venta (%)',
                                marker_color='orange',
                                opacity=0.5
                            ))
                            fig_trading.add_annotation(
                                text="MTaurus. X: mtaurus_ok", 
                                xref="paper", 
                                yref="paper", 
                                x=0.95, 
                                y=0.05, 
                                showarrow=False, 
                                font=dict(size=14, color="gray"), 
                                opacity=0.5
                            )
                            fig_trading.update_layout(
                                title=f"Resultados de la Estrategia de Trading para {ticker_ma} ({compression_ma})",
                                xaxis_title="Tipo de Señal",
                                yaxis_title="Porcentaje (%)",
                                template="plotly_dark",
                                hovermode="x unified",
                                showlegend=True,
                                barmode='group'
                            )
                            st.plotly_chart(fig_trading, use_container_width=True)

                            # Resumen de la estrategia
                            st.markdown(f"""
                            ### Resumen de la Estrategia
                            Para {ticker_ma} ({compression_ma}), usando retornos en el {low_percentile}º percentil para compras y el {high_percentile}º percentil para ventas:
                            - **Señales de Compra**: {int(trading_df['Buy_Signals'].iloc[0])} señales, con una tasa de éxito de {trading_df['Buy_Success_Rate (%)'].iloc[0]:.2f}% y una ganancia promedio de {trading_df['Avg_Buy_Gain (%)'].iloc[0]:.2f}% en los próximos {look_forward_days} períodos.
                            - **Señales de Venta**: {int(trading_df['Sell_Signals'].iloc[0])} señales, con una tasa de éxito de {trading_df['Sell_Success_Rate (%)'].iloc[0]:.2f}% y una pérdida promedio de {trading_df['Avg_Sell_Gain (%)'].iloc[0]:.2f}% en los próximos {look_forward_days} períodos.
                            Esto sugiere que podrías comprar después de retornos extremadamente bajos y vender después de retornos extremadamente altos, con las tasas de éxito y ganancias/pérdidas promedio indicadas.
                            """)
    else:
        st.warning("⚠️ Por favor, ingrese un símbolo de ticker válido para comenzar el análisis.")

# Footer
st.markdown("---")
st.markdown("© 2024 MTaurus. Todos los derechos reservados.")
