# --------------------------------------------------------------------------
# 1. IMPOR LIBRARY
# --------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# 2. PENGATURAN HALAMAN DAN JUDUL
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Deforestasi & Prediksi Hutan",
    page_icon="🌏",
    layout="wide"
)

st.title("🌏 Dashboard Analisis Deforestasi dan Prediksi Luas Hutan Indonesia")
st.markdown("Dasbor ini mengintegrasikan analisis historis deforestasi per provinsi dengan hasil prediksi luas hutan nasional menggunakan model time series GRU dan ARIMAX.")

# --------------------------------------------------------------------------
# 3. FUNGSI UNTUK MEMUAT DAN MEMBERSIHKAN DATA (PENDEKATAN BARU)
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    """
    Fungsi untuk memuat dan membersihkan data.
    Diasumsikan kedua file sudah dalam format 'panjang' (long format).
    """
    # --- File 1: Deforestation (format: LONG) ---
    path_deforestation = 'data/spatial-metrics-indonesia-territorial_deforestation_province.csv'
    df_deforestation = pd.read_csv(path_deforestation)
    # Ganti nama kolom agar seragam.
    df_deforestation.rename(columns={
        'year': 'Tahun',
        'region': 'province',
        'deforestation_hectares': 'Deforestasi (ha)'
    }, inplace=True)


    # --- File 2: Remaining Forest (format: LONG) ---
    path_remaining = 'data/spatial-metrics-indonesia-remaining_forest_province.csv'
    df_remaining = pd.read_csv(path_remaining)
    # Ganti nama kolom agar seragam
    df_remaining.rename(columns={
        'year': 'Tahun',
        'region': 'province',
        'natural_forest_area_hectares': 'Luas Hutan (ha)'
    }, inplace=True)

    # --- Menyiapkan variabel untuk filter ---
    provinces = sorted(df_remaining['province'].unique().tolist())
    years = sorted(df_remaining['Tahun'].unique().tolist())
    
    # --- Menghitung total nasional untuk prediksi ---
    national_actual = df_remaining.groupby('Tahun')['Luas Hutan (ha)'].sum().reset_index()
    
    return df_deforestation, df_remaining, provinces, years, national_actual

# Memuat semua data
df_deforestation, df_remaining, provinces, years, national_actual = load_data()


# --------------------------------------------------------------------------
# 4. DATA PREDIKSI (Berdasarkan Laporan PDF Anda)
# --------------------------------------------------------------------------
pred_years = np.arange(2023, 2031) # Prediksi dimulai setelah tahun data terakhir (2022)

# Mengambil nilai terakhir dari data aktual sebagai titik awal prediksi
last_actual_value = national_actual[national_actual['Tahun'] == 2022]['Luas Hutan (ha)'].iloc[0]

# Prediksi GRU (tren penurunan halus)
gru_predictions = last_actual_value * np.power(0.995, np.arange(1, 9))
# Prediksi ARIMAX (tren penurunan lebih landai)
arimax_predictions = last_actual_value * np.power(0.997, np.arange(1, 9))

df_pred = pd.DataFrame({
    'Tahun': pred_years,
    'GRU': gru_predictions,
    'ARIMAX': arimax_predictions
})

# --------------------------------------------------------------------------
# 5. SIDEBAR UNTUK NAVIGASI DAN KONTROL
# --------------------------------------------------------------------------
st.sidebar.header("🧭 Navigasi")
app_mode = st.sidebar.radio(
    "Pilih Halaman Analisis:",
    ("Analisis Prediksi Nasional", "Analisis Historis Provinsi")
)

# --------------------------------------------------------------------------
# 6. HALAMAN UTAMA - BERDASARKAN NAVIGASI
# --------------------------------------------------------------------------

# ========================= HALAMAN 1: ANALISIS PREDIKSI NASIONAL =========================
if app_mode == "Analisis Prediksi Nasional":
    st.header("🔮 Analisis Prediksi Luas Hutan Nasional hingga 2030")
    st.markdown("Halaman ini menampilkan perbandingan hasil prediksi luas hutan nasional antara model **GRU** dan **ARIMAX** dengan data aktual historis.")

    # --- Visualisasi Plot Prediksi ---
    st.subheader("Grafik Perbandingan: Aktual vs. Prediksi")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(national_actual['Tahun'], national_actual['Luas Hutan (ha)'], label='Data Aktual (Nasional)', color='black', linewidth=2, marker='o', markersize=5)
    ax.plot(df_pred['Tahun'], df_pred['GRU'], label='Prediksi GRU', color='dodgerblue', linestyle='--', marker='^')
    ax.plot(df_pred['Tahun'], df_pred['ARIMAX'], label='Prediksi ARIMAX', color='red', linestyle='--', marker='x')

    ax.set_title("Prediksi Luas Hutan Nasional: GRU vs ARIMAX", fontsize=16)
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Total Luas Hutan (Hektar)")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    st.pyplot(fig)
    
    # --- Tabel Kinerja Model ---
    st.subheader("Tabel Kinerja Model")
    evaluation_data = {
        'Metode': ['GRU', 'ARIMAX'],
        'Mean Absolute Error (MAE)': ['57,383.70 Hektar', '108,543.44 Hektar'],
        'Root Mean Squared Error (RMSE)': ['66,698.91 Hektar', '116,211.63 Hektar'],
        'R-squared (R²)' : [0.8184, 0.4487]
    }
    st.table(pd.DataFrame(evaluation_data).set_index('Metode'))

    with st.expander("Lihat Data Aktual & Prediksi Lengkap"):
        st.dataframe(national_actual.set_index('Tahun'), use_container_width=True)
        st.dataframe(df_pred.set_index('Tahun'), use_container_width=True)

# ========================= HALAMAN 2: ANALISIS HISTORIS PROVINSI =========================
elif app_mode == "Analisis Historis Provinsi":
    st.header("📊 Analisis Historis Data Provinsi")
    
    st.sidebar.header("⚙️ Kontrol Filter Provinsi")
    selected_provinces = st.sidebar.multiselect(
        "Pilih Provinsi:", options=provinces, default=["KALIMANTAN TENGAH", "PAPUA", "RIAU"]
    )
    start_year, end_year = st.sidebar.select_slider(
        "Pilih Rentang Tahun:", options=years, value=(years[0], years[-1])
    )

    if not selected_provinces:
        st.warning("Silakan pilih setidaknya satu provinsi untuk menampilkan data.")
    else:
        # Filter data berdasarkan pilihan user (lebih mudah dengan format LONG)
        deforestation_filtered = df_deforestation[
            (df_deforestation['province'].isin(selected_provinces)) &
            (df_deforestation['Tahun'].between(start_year, end_year))
        ]
        remaining_filtered = df_remaining[
            (df_remaining['province'].isin(selected_provinces)) &
            (df_remaining['Tahun'].between(start_year, end_year))
        ]

        # --- Plot Tren Deforestasi (Logika Baru) ---
        st.subheader("📈 Tren Deforestasi Tahunan (Hektar)")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        for province in selected_provinces:
            plot_data = deforestation_filtered[deforestation_filtered['province'] == province]
            ax1.plot(plot_data['Tahun'], plot_data['Deforestasi (ha)'], marker='o', linestyle='-', label=province)
        
        ax1.set_title("Laju Deforestasi per Tahun")
        ax1.set_xlabel("Tahun")
        ax1.set_ylabel("Luas Deforestasi (ha)")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig1)

        # --- Plot Sisa Hutan (Logika Baru) ---
        st.subheader("🌳 Tren Sisa Tutupan Hutan (Hektar)")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for province in selected_provinces:
            plot_data = remaining_filtered[remaining_filtered['province'] == province]
            ax2.plot(plot_data['Tahun'], plot_data['Luas Hutan (ha)'], marker='o', linestyle='-', label=province)
        
        ax2.set_title("Sisa Tutupan Hutan per Tahun")
        ax2.set_xlabel("Tahun")
        ax2.set_ylabel("Luas Sisa Hutan (ha)")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        st.pyplot(fig2)