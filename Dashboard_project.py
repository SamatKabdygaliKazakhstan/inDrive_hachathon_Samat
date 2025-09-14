# =====================================================================================
# Фаза 4: Прототипирование и "RL-Фасад"
# Цель: Создать интерактивный дашборд для демонстрации решения.
# =====================================================================================
# --- 1. Установка и импорт библиотек ---
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import h3
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 2. Настройка страницы и заголовков ---
st.set_page_config(layout="wide", page_title="inDrive Geo-Behavioral Engine")
st.title("Движок Геоповеденческого Профилирования 2.0")
st.markdown("""
Добро пожаловать в прототип аналитического движка для inDrive.
Это решение анализирует анонимные геотреки для создания поведенческих профилей поездок ("ДНК")
и построения карты "поведенческих рисков" города.
""")

# --- 3. Функции для загрузки данных (с кэшированием) ---
@st.cache_data
def load_data(path):
    """Универсальная функция для загрузки данных."""
    if not os.path.exists(path):
        st.error(f"❌ Файл не найден по пути: {path}")
        return None
    try:
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
        elif path.endswith('.geojson'):
            return gpd.read_file(path)
        elif path.endswith('.gz'):
            return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке файла {path}: {e}")
        return None
    return None

# --- 4. Загрузка всех необходимых данных ---
BASE_PATH = r'C:\Users\dayrb\Downloads\Decetrathon 4.0_InDrive Case#2_Samat Kabdygali\processed_data'
DNA_RESULTS_PATH = os.path.join(BASE_PATH, 'dna_results_baseline.parquet')
FEATURES_PATH = os.path.join(BASE_PATH, 'features_spatial.parquet')
H3_GRID_PATH = os.path.join(BASE_PATH, 'h3_grid_scores.geojson')
SCALER_PATH = os.path.join(BASE_PATH, 'dna_scaler.gz')

with st.spinner('Загрузка аналитических данных...'):
    df_dna = load_data(DNA_RESULTS_PATH)
    df_features = load_data(FEATURES_PATH)
    gdf_h3 = load_data(H3_GRID_PATH)
    scaler = load_data(SCALER_PATH)

data_loaded_successfully = all(df is not None for df in [df_dna, df_features, gdf_h3, scaler])

# --------------------------------------------------------------------------
if data_loaded_successfully:

    # ИСПРАВЛЕНИЕ: GeoJSON при чтении превращает индекс в колонку.
    if 'h3_09' in gdf_h3.columns:
        gdf_h3 = gdf_h3.rename(columns={'h3_09': 'h3_index'})
    else:
        st.warning("В файле GeoJSON не найдена ожидаемая колонка 'h3_09'. Карта может не отобразиться.")

    # Воссоздаем PCA-компоненты для визуализации
    try:
        df_features_aligned = df_features.reindex(columns=scaler.feature_names_in_).fillna(0)
        features_scaled = scaler.transform(df_features_aligned)
        pca = PCA(n_components=2, random_state=42)
        features_pca = pca.fit_transform(features_scaled)
        df_dna['pca_1'] = features_pca[:, 0]
        df_dna['pca_2'] = features_pca[:, 1]
    except Exception as e:
        st.error(f"Ошибка при воссоздании PCA: {e}")
        data_loaded_successfully = False

if data_loaded_successfully:
    # --- 5. Логика "RL-Фасада" ---
    def get_rl_recommendations(gdf, selected_hex_id, risk_weight, demand_weight, dist_weight):
        if selected_hex_id not in gdf['h3_index'].values:
            return None

        neighbors = h3.grid_disk(selected_hex_id, 1)
        recommendations = []

        gdf_indexed = gdf.set_index('h3_index')

        for neighbor_id in neighbors:
            if neighbor_id != selected_hex_id and neighbor_id in gdf_indexed.index:
                neighbor_data = gdf_indexed.loc[neighbor_id]
                utility = (
                    demand_weight * neighbor_data['demand_score'] -
                    risk_weight * neighbor_data['risk_score'] -
                    dist_weight * 1
                )
                recommendations.append({'from_id': selected_hex_id, 'to_id': neighbor_id, 'utility': utility})

        if not recommendations:
            return None

        reco_df = pd.DataFrame(recommendations).sort_values('utility', ascending=False).head(3)

        # --- Используем новую функцию вместо h3_to_geo ---
        reco_df['from_lat'], reco_df['from_lng'] = h3.cell_to_latlng(selected_hex_id)
        reco_df['to_lat'] = reco_df['to_id'].apply(lambda x: h3.cell_to_latlng(x)[0])
        reco_df['to_lng'] = reco_df['to_id'].apply(lambda x: h3.cell_to_latlng(x)[1])

        return reco_df

    # --- 6. Создание вкладок в интерфейсе ---
    tab1, tab2 = st.tabs(["Анализ Водителей (ДНК)", "Карта Рисков и Рекомендации"])

    with tab1:
        st.header("Анализ стилей вождения на основе 'ДНК' поездок")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Профили стилей вождения")
            st.write("Здесь все поездки спроецированы в 2D-пространство для визуального анализа.")
            color_mode = st.radio("Отобразить по:", ('Кластерам', 'Аномалиям'), horizontal=True)

            color_col = 'cluster' if color_mode == 'Кластерам' else 'is_anomaly'
            title = 'Визуализация кластеров (PCA)' if color_mode == 'Кластерам' else 'Визуализация аномалий (PCA)'

            fig_pca = px.scatter(
                df_dna.sort_values(by='is_anomaly'), x='pca_1', y='pca_2', color=color_col,
                title=title, hover_name=df_dna.index,
                labels={'pca_1': 'Главная компонента 1', 'pca_2': 'Главная компонента 2'},
                template='plotly_white'
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        with col2:
            st.subheader("Индивидуальная 'ДНК' поездки")
            selected_trip_id = st.selectbox("Выберите ID Поездки для анализа:", df_features.index)

            if selected_trip_id:
                trip_features = df_features.loc[selected_trip_id]
                trip_cluster = df_dna.loc[selected_trip_id, 'cluster']
                trip_anomaly = "Да" if df_dna.loc[selected_trip_id, 'is_anomaly'] == 1 else "Нет"

                st.metric(label="Кластер стиля вождения", value=f"Кластер {trip_cluster}")
                st.metric(label="Является аномалией?", value=trip_anomaly)

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=trip_features.values, theta=trip_features.index, fill='toself', name=f'ID: {selected_trip_id}'))
                fig_radar.update_layout(title=f"Радарная диаграмма 'ДНК' для поездки {selected_trip_id}", polar=dict(radialaxis=dict(visible=True)), showlegend=False)
                st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        st.header("Интерактивная карта поведенческих рисков и спроса")

        st.sidebar.title("⚙️ Настройки карты")
        map_metric = st.sidebar.radio("Показать на карте:", ('Уровень риска (Risk Score)', 'Уровень спроса (Demand Score)'))
        st.sidebar.subheader("Настройки 'RL-Фасада'")
        risk_w = st.sidebar.slider("Вес риска (штраф)", 0.0, 2.0, 1.0, 0.1)
        demand_w = st.sidebar.slider("Вес спроса (награда)", 0.0, 2.0, 1.2, 0.1)
        dist_w = st.sidebar.slider("Штраф за расстояние", 0.0, 1.0, 0.5, 0.1)

        get_color_weight = 'risk_score' if map_metric == 'Уровень риска (Risk Score)' else 'demand_score'

        view_state = pdk.ViewState(latitude=51.1, longitude=71.45, zoom=11, pitch=50)

        h3_layer = pdk.Layer(
            "H3HexagonLayer", gdf_h3, pickable=True, stroked=True, filled=True, extruded=True,
            get_hexagon="h3_index",
            get_fill_color=f"[255, (1 - {get_color_weight}) * 255, 0, 140]",
            get_elevation=f"{get_color_weight} * 500",
            wireframe=True,
            line_width_min_pixels=2,
        )

        deck = pdk.Deck(
            layers=[h3_layer],
            initial_view_state=view_state,
            map_style="mapbox://styles/mapbox/light-v9",
            tooltip={"text": "H3 ID: {h3_index}\nRisk Score: {risk_score:.3f}\nDemand Score: {demand_score:.3f}"}
        )
        st.pydeck_chart(deck, use_container_width=True)

        st.info("ℹ️ Ниже приведен пример работы 'RL-Фасада' для ячейки с самым высоким риском.")
        example_hex = gdf_h3.sort_values('risk_score', ascending=False).iloc[0]['h3_index']
        st.write(f"**Пример рекомендаций для ячейки `{example_hex}`:**")

        reco = get_rl_recommendations(gdf_h3, example_hex, risk_w, demand_w, dist_w)

        if reco is not None and not reco.empty:
            st.write("Топ-3 рекомендуемых соседних ячейки для перемещения (чем выше 'utility', тем лучше):")
            st.dataframe(reco[['to_id', 'utility']])
        else:
            st.write("Не удалось найти выгодных соседей по заданным параметрам.")
else:
    st.warning("Не удалось загрузить один или несколько файлов данных. Пожалуйста, проверьте пути к файлам и убедитесь, что все предыдущие фазы были выполнены.")
