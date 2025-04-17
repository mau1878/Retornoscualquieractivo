import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

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
    """
    Flatten DataFrame columns and prepend ticker name to column names.
    
    Parameters:
    - df: DataFrame from yf.download, possibly with MultiIndex or single-level columns
    - ticker: String ticker symbol (e.g., 'YPFD.BA')
    
    Returns:
    - DataFrame with columns renamed to 'Open {ticker}', 'Close {ticker}', etc.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Handle MultiIndex (multiple tickers)
        df.columns = [f"{col[0]} {col[1]}" if col[1] else col[0] for col in df.columns]
    else:
        # Handle single-level columns (single ticker)
        df.columns = [f"{col} {ticker}" for col in df.columns]
    return df

# Función para evaluar expresiones con tickers
def evaluate_ticker_expression(expression, data_dict, index):
    """
    Evaluate a mathematical expression involving tickers' closing prices.
    
    Parameters:
    - expression: String, e.g., "GGAL.BA*10/GGAL"
    - data_dict: Dict of DataFrames, where keys are tickers and values are DataFrames with closing prices
    - index: DatetimeIndex to align the resulting series
    
    Returns:
    - pd.Series with the evaluated prices, or None if evaluation fails
    """
    try:
        # Extract potential tickers (start with letter, allow letters, numbers, dots)
        potential_tickers = set(re.findall(r'\b[A-Za-z][A-Za-z0-9\._]*\b', expression))
        # Only include tickers that exist in data_dict
        tickers = [t for t in potential_tickers if t in data_dict]
        
        if not tickers:
            st.error("No se encontraron tickers válidos en la expresión.")
            return None
        
        # Create a local dictionary for evaluation
        local_vars = {}
        for ticker in tickers:
            close_col = f"Close {ticker}"
            if close_col not in data_dict[ticker].columns:
                st.error(f"No se encontró la columna de cierre para el ticker {ticker}.")
                return None
            local_vars[ticker.replace('.', '_')] = data_dict[ticker][close_col].reindex(index).fillna(method='ffill').fillna(method='bfill')
        
        # Replace ticker names in the expression
        eval_expression = expression
        for ticker in tickers:
            eval_expression = eval_expression.replace(ticker, ticker.replace('.', '_'))
        
        # Evaluate the expression
        result = pd.eval(eval_expression, local_dict=local_vars, engine='python')
        
        # Ensure the result is a Series
        result = pd.Series(result, index=index).fillna(method='ffill').fillna(method='bfill')
        
        if result.isna().all() or result.isin([np.inf, -np.inf]).any():
            st.error("La evaluación de la expresión resultó en valores inválidos (NaN o infinitos).")
            return None
        
        return result
    
    except Exception as e:
        st.error(f"Error al evaluar la expresión: {e}")
        return None
# Función para descargar y comprimir datos
def download_data(tickers, start, end, compression='Daily', expression=None):
    """
    Download and process data for one or more tickers, or compute a derived price from an expression.
    
    Parameters:
    - tickers: List of tickers or a single ticker string
    - start, end: Datetime for data range
    - compression: 'Daily', 'Weekly', or 'Monthly'
    - expression: Optional string expression (e.g., "YPFD.BA/YPF")
    
    Returns:
    - DataFrame with price data or computed prices
    """
    try:
        if isinstance(tickers, str):
            tickers = [tickers]
        
        data_dict = {}
        for ticker in tickers:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                st.error(f"No hay datos disponibles para el ticker **{ticker}** en el rango de fechas {start.strftime('%Y-%m-%d')} a {end.strftime('%Y-%m-%d')}. Verifique el ticker o el rango de fechas.")
                return None
            df = flatten_columns(df, ticker)
            # Debug: Log column names
            st.write(f"Columnas para {ticker} después de flatten_columns: {df.columns.tolist()}")
            data_dict[ticker] = df
        
        # Determine the common index
        common_index = data_dict[tickers[0]].index
        for ticker in tickers[1:]:
            common_index = common_index.intersection(data_dict[ticker].index)
        
        if len(common_index) == 0:
            st.error("No hay fechas comunes entre los tickers seleccionados. Intente un rango de fechas más reciente.")
            return None
        
        # Resample data based on compression
        if compression == 'Weekly':
            rule = 'W'
        elif compression == 'Monthly':
            rule = 'M'
        else:
            rule = 'D'
        
        if rule != 'D':
            for ticker in tickers:
                # Build aggregation dictionary for this ticker only
                available_columns = data_dict[ticker].columns
                ticker_agg_dict = {
                    col: func for col, func in {
                        f'Open {ticker}': 'first',
                        f'High {ticker}': 'max',
                        f'Low {ticker}': 'min',
                        f'Close {ticker}': 'last',
                        f'Volume {ticker}': 'sum'
                    }.items() if col in available_columns
                }
                if not ticker_agg_dict:
                    st.error(f"No se encontraron columnas válidas para la agregación de {ticker}. Columnas disponibles: {available_columns.tolist()}")
                    return None
                # Debug: Log aggregation dictionary
                st.write(f"Agregación para {ticker}: {ticker_agg_dict}")
                data_dict[ticker] = data_dict[ticker].resample(rule).agg(ticker_agg_dict).dropna()
            common_index = data_dict[tickers[0]].index
            for ticker in tickers[1:]:
                common_index = common_index.intersection(data_dict[ticker].index)
        
        # Compute derived price if expression is provided
        if expression:
            derived_prices = evaluate_ticker_expression(expression, data_dict, common_index)
            if derived_prices is None:
                return None
            result_df = pd.DataFrame({'Derived Price': derived_prices}, index=common_index)
        else:
            result_df = data_dict[tickers[0]].reindex(common_index)
        
        st.write(f"**Columnas disponibles ({compression}):** {result_df.columns.tolist()}")
        return result_df
    
    except Exception as e:
        st.error(f"Error al descargar o procesar datos: {e}")
        return None

# Función para calcular retornos
def calculate_returns(data, price_column='Derived Price'):
    """
    Calculate percentage returns from a price series.
    
    Parameters:
    - data: DataFrame with price data
    - price_column: Name of the price column to use
    
    Returns:
    - Series with percentage returns
    """
    if price_column not in data.columns:
        st.error(f"La columna **{price_column}** no existe en los datos.")
        return pd.Series()
    returns = data[price_column].pct_change() * 100
    return returns

# Función para analizar la estrategia basada en percentiles de retornos
def analyze_returns_percentile_strategy(data, price_column, look_forward_days, low_percentile, high_percentile):
    """
    Analyze a trading strategy based on return percentiles.
    """
    if len(data) < look_forward_days + 1:
        st.error(f"El conjunto de datos es demasiado corto para analizar con {look_forward_days} días de proyección.")
        return pd.DataFrame()

    data['Returns'] = calculate_returns(data, price_column)
    
    returns_data = data['Returns'].dropna()
    if len(returns_data) < 10:
        st.warning("No hay suficientes datos de retornos para analizar.")
        return pd.DataFrame()
    
    low_threshold = np.percentile(returns_data, low_percentile)
    high_threshold = np.percentile(returns_data, high_percentile)
    
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
        
        if pd.isna(current_return) or pd.isna(initial_price) or initial_price == 0 or future_prices.isna().any():
            continue
        
        if current_return <= low_threshold:
            buy_signals += 1
            future_max = future_prices.max()
            gain = (future_max - initial_price) / initial_price * 100
            buy_gains.append(gain)
            if future_max > initial_price:
                buy_successes += 1
        
        elif current_return >= high_threshold:
            sell_signals += 1
            future_min = future_prices.min()
            loss = (future_min - initial_price) / initial_price * 100
            sell_gains.append(loss)
            if future_min < initial_price:
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
    st.write("### Ingrese un ticker o una expresión")
    input_type = st.radio("Seleccione el tipo de entrada", ["Ticker único", "Expresión personalizada"], key="input_type_original")
    
    if input_type == "Ticker único":
        ticker = st.text_input("🖊️ Ingrese el símbolo del ticker", value="AAPL", key="ticker_original").upper()
        expression = None
        tickers = [ticker] if ticker and re.match(r'^[A-Za-z][A-Za-z0-9\._]*$', ticker) else []
    else:
        expression = st.text_input("🖊️ Ingrese la expresión (e.g., GGAL.BA*10/GGAL)", value="GGAL.BA*10/GGAL", key="expression_original")
        tickers = list(set(re.findall(r'\b[A-Za-z][A-Za-z0-9\._]*\b', expression))) if expression else []
        ticker = "Expresión Personalizada" if expression else ""
    
    if input_type == "Ticker único" and not tickers:
        st.warning("⚠️ Por favor, ingrese un ticker válido.")
    elif input_type == "Expresión personalizada" and not expression:
        st.warning("⚠️ Por favor, ingrese una expresión válida.")
    else:
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

        # Outlier threshold inputs
        st.write("### Filtrar valores extremos (outliers) para las visualizaciones")
        filter_outliers = st.checkbox("Excluir valores extremos de los gráficos", value=False, key="filter_outliers")
        if filter_outliers:
            col1, col2 = st.columns(2)
            with col1:
                lower_bound = st.number_input(
                    "Límite inferior para retornos (%)",
                    min_value=-1000.0,
                    max_value=0.0,
                    value=-100.0,
                    step=1.0,
                    key="lower_bound"
                )
            with col2:
                upper_bound = st.number_input(
                    "Límite superior para retornos (%)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=100.0,
                    step=1.0,
                    key="upper_bound"
                )
        else:
            lower_bound = None
            upper_bound = None

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        if start_date > end_date:
            st.error("La fecha de inicio no puede ser posterior a la fecha de fin.")
        else:
            if tickers or expression:
                data = download_data(tickers, start_date, end_date, compression=compression, expression=expression)

                if data is not None:
                    price_column = 'Derived Price' if expression else f'Close {tickers[0]}'

                    if apply_ratio:
                        st.subheader("🔄 Aplicando ajuste por ratio YPFD.BA/YPF (CCL)")
                        ypfd_ba_ticker = "YPFD.BA"
                        ypf_ticker = "YPF"
                        ypfd_ba_data = download_data([ypfd_ba_ticker], start_date, end_date, compression=compression)
                        ypf_data = download_data([ypf_ticker], start_date, end_date, compression=compression)

                        if ypfd_ba_data is not None and ypf_data is not None:
                            close_col_ypfd = f"Close {ypfd_ba_ticker}"
                            close_col_ypf = f"Close {ypf_ticker}"

                            if close_col_ypfd in ypfd_ba_data.columns and close_col_ypf in ypf_data.columns:
                                ypfd_ba_data = ypfd_ba_data.fillna(method='ffill').fillna(method='bfill')
                                ypf_data = ypf_data.fillna(method='ffill').fillna(method='bfill')
                                ratio = ypfd_ba_data[close_col_ypfd] / ypf_data[close_col_ypf]
                                ratio = ratio.reindex(data.index).fillna(method='ffill').fillna(method='bfill')

                                data['Close Ajustado'] = data[price_column] / ratio
                                price_column = 'Close Ajustado'
                            else:
                                st.error(f"No se encontraron columnas de precio válidas para {ypfd_ba_ticker} o {ypf_ticker}.")
                        else:
                            st.error("No se pudieron descargar los datos necesarios para aplicar el ratio.")

                    # Calcular retornos
                    data['Returns'] = calculate_returns(data, price_column)

                    # Apply outlier filtering for visualizations
                    if filter_outliers and lower_bound is not None and upper_bound is not None:
                        plot_data = data[(data['Returns'] >= lower_bound) & (data['Returns'] <= upper_bound)].copy()
                        if plot_data['Returns'].dropna().empty:
                            st.warning("⚠️ Los límites de outliers excluyen todos los datos de retornos. Ajuste los límites.")
                    else:
                        plot_data = data.copy()

                    # Visualización 1: Precio Histórico
                    st.write(f"### 📈 Precio Histórico ({compression})")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data[price_column], mode='lines', name='Precio'))
                    fig.add_annotation(text="MTaurus. X: mtaurus_ok", xref="paper", yref="paper", x=0.95, y=0.05, showarrow=False, font=dict(size=14, color="gray"), opacity=0.5)
                    fig.update_layout(
                        title=f"Precio Histórico de {ticker} ({compression})",
                        xaxis_title="Fecha", yaxis_title="Precio", legend_title="Leyenda", template="plotly_dark", hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Visualización 2: Retornos Históricos
                    st.write(f"### 📉 Retornos Históricos ({compression})")
                    if plot_data['Returns'].dropna().empty:
                        st.error("No hay datos válidos de retornos para graficar después de filtrar outliers.")
                    else:
                        fig_returns = go.Figure()
                        fig_returns.add_trace(go.Scatter(
                            x=plot_data.index, 
                            y=plot_data['Returns'], 
                            mode='lines', 
                            name='Retornos (%)',
                            line=dict(color='lightgrey')
                        ))

                        historical_mean = plot_data['Returns'].mean()
                        if not pd.isna(historical_mean):
                            fig_returns.add_shape(
                                type="line", 
                                x0=plot_data.index.min(), 
                                x1=plot_data.index.max(), 
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
                                x=plot_data.index.max(), 
                                y=historical_mean, 
                                text=f"Promedio: {historical_mean:.2f}%",
                                showarrow=True, 
                                arrowhead=1, 
                                ax=20, 
                                ay=-20, 
                                font=dict(color="lightblue")
                            )

                        lower_percentile = st.slider("Seleccione el percentil inferior", min_value=1, max_value=49, value=5, key="lower_percentile")
                        upper_percentile = st.slider("Seleccione el percentil superior", min_value=51, max_value=99, value=95, key="upper_percentile")

                        returns_data = plot_data['Returns'].dropna()
                        if not returns_data.empty:
                            lower_value = np.percentile(returns_data, lower_percentile)
                            upper_value = np.percentile(returns_data, upper_percentile)

                            fig_returns.add_shape(
                                type="line", 
                                x0=plot_data.index.min(), 
                                x1=plot_data.index.max(), 
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
                                x=plot_data.index.max(), 
                                y=lower_value, 
                                text=f"P{lower_percentile}: {lower_value:.2f}%",
                                showarrow=True, 
                                arrowhead=1, 
                                ax=20, 
                                ay=20, 
                                font=dict(color="red")
                            )

                            fig_returns.add_shape(
                                type="line", 
                                x0=plot_data.index.min(), 
                                x1=plot_data.index.max(), 
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
                                x=plot_data.index.max(), 
                                y=upper_value, 
                                text=f"P{upper_percentile}: {upper_value:.2f}%",
                                showarrow=True, 
                                arrowhead=1, 
                                ax=20, 
                                ay=-20, 
                                font=dict(color="green")
                            )

                        fig_returns.add_shape(
                            type="line", 
                            x0=plot_data.index.min(), 
                            x1=plot_data.index.max(), 
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
                    percentile_values = np.percentile(plot_data['Returns'].dropna(), percentiles) if not plot_data['Returns'].dropna().empty else []
                    
                    st.write("#### Seleccionar fechas específicas para destacar en el histograma")
                    num_dates = st.number_input("Número de fechas a destacar", min_value=0, max_value=10, value=0, key="num_dates_hist")
                    selected_dates = []
                    returns_values = []
                    if num_dates > 0:
                        for i in range(num_dates):
                            date = st.date_input(
                                f"Seleccione la fecha {i+1}",
                                value=plot_data.index[-1] if not plot_data.empty else pd.to_datetime('today'),
                                min_value=plot_data.index[0] if not plot_data.empty else pd.to_datetime('1900-01-01'),
                                max_value=plot_data.index[-1] if not plot_data.empty else pd.to_datetime('today'),
                                key=f"hist_date_{i}"
                            )
                            date = pd.to_datetime(date)
                            if date in plot_data.index:
                                selected_dates.append(date)
                                ret_value = plot_data.loc[date, 'Returns']
                                if not pd.isna(ret_value):
                                    returns_values.append(ret_value)
                                else:
                                    st.warning(f"No hay datos de retornos para la fecha {date.strftime('%Y-%m-%d')}.")
                            else:
                                st.warning(f"La fecha {date.strftime('%Y-%m-%d')} no está en el rango de datos.")
                    
                    if plot_data['Returns'].dropna().empty:
                        st.error("No hay datos válidos de retornos para graficar después de filtrar outliers.")
                    else:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.histplot(plot_data['Returns'].dropna(), kde=True, color='blue', bins=100, ax=ax)
                        for percentile, value in zip(percentiles, percentile_values):
                            ax.axvline(value, color='red', linestyle='--')
                            ax.text(value, ax.get_ylim()[1] * 0.9, f'{percentile}º Percentil', color='red', rotation='vertical', verticalalignment='center', horizontalalignment='right')
                        for date, ret_value in zip(selected_dates, returns_values):
                            ax.axvline(ret_value, color='green', linestyle='-', alpha=0.5)
                            ax.text(ret_value, ax.get_ylim()[1] * 0.95, f"{date.strftime('%Y-%m-%d')} {ret_value:.2f}%", 
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
                    if plot_data['Returns'].dropna().empty:
                        st.error("No hay datos válidos de retornos para graficar después de filtrar outliers.")
                    else:
                        fig_hist = go.Figure()
                        fig_hist.add_trace(go.Histogram(x=plot_data['Returns'].dropna(), nbinsx=num_bins, marker_color=hist_color, opacity=0.75, name="Histograma"))
                        for percentile, value in zip(percentiles, percentile_values):
                            fig_hist.add_vline(x=value, line=dict(color="red", width=2, dash="dash"), 
                                               annotation_text=f'{percentile}º Percentil', annotation_position="top", 
                                               annotation=dict(textangle=-90, font=dict(color="red")))
                        for i, (date, ret_value) in enumerate(zip(selected_dates, returns_values)):
                            annotation_y_position = 0.95 - (i * 0.05)
                            fig_hist.add_vline(x=ret_value, line=dict(color="green", width=2, dash="solid"))
                            fig_hist.add_annotation(
                                x=ret_value,
                                y=annotation_y_position,
                                yref="paper",
                                text=f"{date.strftime('%Y-%m-%d')} {ret_value:.2f}%",
                                showarrow=True,
                                arrowhead=1,
                                ax=10,
                                ay=0,
                                textangle=90,
                                font=dict(color="green", size=10),
                                align="right"
                            )
                        fig_hist.add_annotation(text="MTaurus. X: mtaurus_ok", xref="paper", yref="paper", x=0.95, y=0.05, 
                                                showarrow=False, font=dict(size=14, color="gray"), opacity=0.5)
                        fig_hist.update_layout(
                            title=f'Histograma de Retornos de {ticker} ({compression})',
                            xaxis_title='Retornos (%)', 
                            yaxis_title='Frecuencia', 
                            bargap=0.1, 
                            template="plotly_dark", 
                            hovermode="x unified",
                            margin=dict(t=100),
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
# Pestaña 2: Análisis de Trading con Percentiles de Retornos
with tab2:
    st.header("Análisis de Trading con Percentiles de Retornos")
    
    st.markdown("""
    ### ¿Qué hace esta pestaña?
    Evalúa una estrategia de trading basada en los percentiles de los retornos históricos, ya sea para un ticker único o una expresión personalizada (e.g., GGAL.BA*10/GGAL).
    - **Señales de Compra**: Retorno por debajo de un percentil bajo (e.g., 5º), indicando posible recuperación.
    - **Señales de Venta**: Retorno por encima de un percentil alto (e.g., 95º), indicando posible caída.
    Calcula tasas de éxito y ganancias/pérdidas promedio en los próximos N períodos.
    """)

    st.write("### Ingrese un ticker o una expresión")
    input_type_ma = st.radio("Seleccione el tipo de entrada", ["Ticker único", "Expresión personalizada"], key="input_type_ma")
    
    if input_type_ma == "Ticker único":
        ticker_ma = st.text_input("🖊️ Ingrese el símbolo del ticker", value="AAPL", key="ticker_ma").upper()
        expression_ma = None
        tickers_ma = [ticker_ma] if ticker_ma else []
    else:
        expression_ma = st.text_input("🖊️ Ingrese la expresión (e.g., GGAL.BA*10/GGAL)", value="GGAL.BA*10/GGAL", key="expression_ma")
        tickers_ma = set(re.findall(r'[A-Za-z0-9\._]+', expression_ma)) if expression_ma else []
        ticker_ma = "Expresión Personalizada"

    if tickers_ma or expression_ma:
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
        look_forward_days = st.number_input("Períodos de proyección", min_value=1, value=5, key="look_forward_days")
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
                data_ma = download_data(tickers_ma, start_date_ma, end_date_ma, compression=compression_ma, expression=expression_ma)

                if data_ma is not None:
                    price_column_ma = 'Derived Price' if expression_ma else f'Close {tickers_ma[0]}'

                    if apply_ratio_ma:
                        st.subheader("🔄 Aplicando ajuste por ratio YPFD.BA/YPF (CCL)")
                        ypfd_ba_ticker = "YPFD.BA"
                        ypf_ticker = "YPF"
                        ypfd_ba_data = download_data([ypfd_ba_ticker], start_date_ma, end_date_ma, compression=compression_ma)
                        ypf_data = download_data([ypf_ticker], start_date_ma, end_date_ma, compression=compression_ma)

                        if ypfd_ba_data is not None and ypf_data is not None:
                            close_col_ypfd = f"Close {ypfd_ba_ticker}"
                            close_col_ypf = f"Close {ypf_ticker}"

                            if close_col_ypfd in ypfd_ba_data.columns and close_col_ypf in ypf_data.columns:
                                ypfd_ba_data = ypfd_ba_data.fillna(method='ffill').fillna(method='bfill')
                                ypf_data = ypf_data.fillna(method='ffill').fillna(method='bfill')
                                ratio = ypfd_ba_data[close_col_ypfd] / ypf_data[close_col_ypf]
                                ratio = ratio.reindex(data_ma.index).fillna(method='ffill').fillna(method='bfill')

                                data_ma['Close Ajustado'] = data_ma[price_column_ma] / ratio
                                price_column_ma = 'Close Ajustado'
                            else:
                                st.error(f"No se encontraron columnas de precio válidas para {ypfd_ba_ticker} o {ypf_ticker}.")
                        else:
                            st.error("No se pudieron descargar los datos necesarios para aplicar el ratio.")

                    trading_df = analyze_returns_percentile_strategy(
                        data_ma, price_column_ma, look_forward_days, low_percentile, high_percentile
                    )

                    if not trading_df.empty:
                        st.write("### Resultados del Análisis de Trading")
                        st.markdown("""
                        Aquí tienes los resultados:
                        - **Buy_Signals**: Señales de compra (retorno por debajo del percentil bajo).
                        - **Buy_Success_Rate (%)**: Porcentaje de señales de compra exitosas.
                        - **Avg_Buy_Gain (%)**: Ganancia promedio tras señal de compra.
                        - **Sell_Signals**: Señales de venta (retorno por encima del percentil alto).
                        - **Sell_Success_Rate (%)**: Porcentaje de señales de venta exitosas.
                        - **Avg_Sell_Gain (%)**: Pérdida promedio tras señal de venta.
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
                        Esto sugiere que podrías comprar después de retornos extremadamente bajos y vender después de retornos extremadamente altos.
                        """)
    else:
        st.warning("⚠️ Por favor, ingrese un ticker o una expresión válida.")

# Footer
st.markdown("---")
st.markdown("© 2024 MTaurus. Todos los derechos reservados.")
