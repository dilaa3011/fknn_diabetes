import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io
import requests

# Daftar nilai K yang akan diuji
K_VALUES = [3, 5, 7, 9]

# --- 1. Definisi Model FKNN (HANYA EUCLIDEAN) ---
class FuzzyKNN:
    def __init__(self, k=3, m=2):
        self.k = k
        self.m = m

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y)
        self.classes = np.unique(y)
        N = len(y)

        self.membership = np.zeros((N, len(self.classes)), dtype=np.float32)
        for i, c in enumerate(self.classes):
            n_c = np.sum(y == c)
            self.membership[:, i] = np.where(
                y == c,
                0.51,
                0.49 * (n_c / N)
            )

    def predict(self, X_test):
        X_test = np.asarray(X_test, dtype=np.float32)
        preds = []
        memberships = []

        X_train = self.X
        membership_train = self.membership
        classes = self.classes
        k = self.k
        m = self.m

        for x in X_test:
            # Euclidean distance (cepat)
            distances = np.linalg.norm(X_train - x, axis=1)

            # Ambil k tetangga tercepat
            idx = np.argpartition(distances, k)[:k]
            d = distances[idx]
            d[d == 0] = 1e-6

            weights = 1.0 / (d ** (2 / (m - 1)))
            weights_sum = np.sum(weights)

            u = {}
            for i, c in enumerate(classes):
                u[c] = np.sum(membership_train[idx, i] * weights) / weights_sum

            preds.append(max(u, key=u.get))
            memberships.append(u)

        return np.array(preds), memberships

# --- 2. Fungsi Pemuatan Data dan Pelatihan Model ---
@st.cache_resource
def load_data_and_train_model():
    file_id = '1uM8YN9Cfgk3YgFMY2rae6nnHvgUHwil_'
    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    
    try:
        response = requests.get(download_url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    except Exception as e:
        st.error(f"Gagal memuat data dari Google Drive: {e}")
        return None, None, None, None, None, None, None

    # Pemilihan Fitur (Sesuai kode asli)
    feature_cols = ['gender', 'age', 'hypertension', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    target_col = 'diabetes'
    
    df = df[feature_cols + [target_col]].copy()
    original_df = df.copy() 

    # Encoding Gender (Sesuai kode asli)
    df['gender'] = df['gender'].astype(str).str.strip().str.title()
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})
    
    # Konversi Target ke Int (Sesuai kode asli)
    df[target_col] = df[target_col].astype(int)

    # Pisahkan Fitur dan Target
    X = df[feature_cols]
    y = df[target_col]

    # Normalisasi Min-Max
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    df_normalized = pd.DataFrame(X_scaled, columns=feature_cols)
    df_normalized[target_col] = y.values

    # Split Data Train-Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Undersampling
    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    # 🔥 BATAS DATA LATIH (OPTIMASI)
    MAX_TRAIN = 1500
    if len(X_train_res) > MAX_TRAIN:
        X_train_res = X_train_res[:MAX_TRAIN]
        y_train_res = y_train_res[:MAX_TRAIN]

    # --- Melatih dan Mengevaluasi Model FKNN Terbaik ---
    metrics_list = []
    best_accuracy = -1
    best_model_fknn = None
    best_k = None
    
    for k in K_VALUES:
        model = FuzzyKNN(k=k, m=2)
        model.fit(X_train_res, y_train_res.to_numpy())
        
        y_pred, _ = model.predict(X_test)
        accuracy = round(accuracy_score(y_test, y_pred), 4)
        report = classification_report(y_test, y_pred, output_dict=True)

        metrics_list.append({
            'k': k,
            'Akurasi': accuracy,
            'Presisi (Diabetes)': round(report['1']['precision'], 4),
            'Recall (Diabetes)': round(report['1']['recall'], 4),
            'F1-Score (Diabetes)': round(report['1']['f1-score'], 4),
        })
        
        # Cek jika model ini adalah yang terbaik
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_model_fknn = model

    metrics_df = pd.DataFrame(metrics_list)
    
    # Mengembalikan 7 nilai
    return best_model_fknn, scaler, feature_cols, original_df, df_normalized, metrics_df, best_k


# ==============================================================================
# 3. Fungsi untuk Halaman (Pages)
# ==============================================================================

def data_understanding_page(df_original):
    st.header("🔬 Data Understanding")
    st.markdown("Analisis data mentah dari *dataset* diagnosis diabetes.")
    
    st.subheader("Struktur Data Awal")
    st.markdown(f"Data awal memiliki **{len(df_original)}** baris dan **{len(df_original.columns)}** kolom (setelah pemilihan fitur).")
    st.dataframe(df_original.head())
    
    st.subheader("Informasi Fitur yang Digunakan")
    st.markdown("Fitur yang digunakan untuk pelatihan model:")
    st.code(", ".join(df_original.columns.tolist()))
    
    st.subheader("Distribusi Kelas Target ('diabetes')")
    class_counts = df_original['diabetes'].value_counts()
    st.markdown(f"* Non-Diabetes (0): **{class_counts.get(0, 0)}** data")
    st.markdown(f"* Diabetes (1): **{class_counts.get(1, 0)}** data")


def preprocessing_page(df_normalized):
    st.header("⚙️ Preprocessing")
    st.markdown("Langkah-langkah yang dilakukan untuk menyiapkan data sebelum dimasukkan ke model FKNN.")
    
    st.subheader("Langkah Preprocessing")
    st.markdown(
        """
        1.  **Pemilihan Fitur**: Hanya menggunakan `gender`, `age`, `hypertension`, `bmi`, `HbA1c_level`, `blood_glucose_level`, dan `diabetes`.
        2.  **Encoding Data**: Kolom `gender` diubah menjadi angka (`Male`: 1, `Female`: 0, `Other`: 2).
        3.  **Normalisasi Min-Max**: Semua fitur diubah skalanya menjadi rentang **0 hingga 1**.
        4.  **Undersampling**: Data latih diseimbangkan jumlah kelasnya menggunakan **Random Under-Sampler** (RUS).
        """
    )
    
    st.subheader("Data Setelah Normalisasi Min-Max")
    st.markdown("Tampilan beberapa baris data fitur setelah normalisasi.")
    st.dataframe(df_normalized.head())
    

def modelling_page(metrics_df, best_k, best_accuracy):
    st.header("🤖 Modelling & Evaluasi (Hanya Jarak Euclidean)")
    st.markdown("Model yang digunakan adalah **Fuzzy K-Nearest Neighbors (FKNN)** dengan parameter *fuzziness* $m=2$ dan **Jarak Euclidean** ($p=2$).")
    
    st.subheader("Konfigurasi Optimal")
    st.success(f"Model Terbaik: **K={best_k}**")
    st.info(f"💡 Akurasi Tertinggi yang Dicapai: **{best_accuracy*100:.2f}%**")
    
    st.markdown("---")
    
    st.subheader("Perbandingan Performa Model FKNN untuk Berbagai Nilai $k$")
    st.markdown("Tabel di bawah menunjukkan performa Akurasi, Presisi, Recall, dan F1-Score (untuk kelas Diabetes/1) untuk setiap $k$ yang diuji.")
    st.dataframe(metrics_df.set_index('k'))


def prediction_page(best_model_fknn, scaler, feature_cols, best_k):
    
    # --- Input Sidebar untuk Diagnosis ---
    st.sidebar.title("Input Data Pasien")
    
    # Input Data di sidebar (SESUAIKAN DENGAN feature_cols)
    st.sidebar.subheader("Data Demografi & Klinis")
    
    # gender: 'Male': 1, 'Female': 0, 'Other': 2
    st.sidebar.selectbox(
        "Jenis Kelamin", 
        options=[0, 1, 2], 
        format_func=lambda x: "Female (0)" if x == 0 else ("Male (1)" if x == 1 else "Other (2)"), 
        key='gender_in'
    )
    st.sidebar.slider("Umur (Tahun)", 0, 100, 50, 1, key='age_in')
    st.sidebar.selectbox("Hipertensi (0=No, 1=Yes)", options=[0, 1], key='hypertension_in')
    st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1, key='bmi_in')
    st.sidebar.number_input("Level HbA1c (%)", min_value=3.0, max_value=20.0, value=6.0, step=0.1, key='hba1c_in')
    st.sidebar.number_input("Gula Darah (mg/dL)", min_value=50, max_value=500, value=100, step=1, key='blood_glucose_level_in')
    
    # Tombol Prediksi
    if st.sidebar.button("Lakukan Prediksi", type="primary"):
        st.session_state['predict_button_pressed'] = True
    
    # --- Konten Utama Halaman Diagnosis ---
    st.header("🎯 Diagnosis Diabetes Melitus (FKNN)")
    st.markdown(f"Sistem ini menggunakan konfigurasi model **FKNN paling optimal (K={best_k})**.")
    
    # Tampilkan Hasil saat tombol ditekan
    if st.session_state.get('predict_button_pressed', False):
        
        # Ambil input data dari session state
        input_data = {
            'gender': st.session_state.get('gender_in'), 
            'age': st.session_state.get('age_in'),
            'hypertension': st.session_state.get('hypertension_in'), 
            'bmi': st.session_state.get('bmi_in'),
            'HbA1c_level': st.session_state.get('hba1c_in'), 
            'blood_glucose_level': st.session_state.get('blood_glucose_level_in'),
        }
        input_df = pd.DataFrame([input_data])
        
        # Prediksi menggunakan model FKNN terbaik
        input_array = input_df[feature_cols].values
        
        # Normalisasi Input
        scaled_input = scaler.transform(input_array.reshape(1, -1))
        
        st.subheader("📋 Hasil Diagnosis")
        
        with st.spinner(f'Model FKNN (k={best_k}) sedang menganalisis data...'):
            prediction, memberships = best_model_fknn.predict(scaled_input)
            prediction = prediction[0]
            membership_result = memberships[0]
        
        # 3. Tampilkan Hasil Prediksi
        if prediction == 1:
            st.error("🚨 **PREDIKSI: DIABETES**")
            st.markdown("Berdasarkan data yang dimasukkan, pasien **Memiliki Diabetes**.")
        else:
            st.success("🎉 **PREDIKSI: NON-DIABETES**")
            st.markdown("Berdasarkan data yang dimasukkan, pasien **Tidak Memiliki Diabetes**.")
            
        st.markdown("---")
        
        # 4. Tampilkan Derajat Keanggotaan
        st.subheader(f"📊 Derajat Keanggotaan Fuzzy (FKNN $k={best_k}$)")
        
        col_m1, col_m2 = st.columns(2)
        non_diabetes_score = membership_result.get(0, 0)
        diabetes_score = membership_result.get(1, 0)
        
        col_m1.metric(
            label="Kelas 0: Non-Diabetes", 
            value=f"{non_diabetes_score:.4f}",
            delta="Tertinggi" if non_diabetes_score > diabetes_score else None,
            delta_color="normal" if non_diabetes_score > diabetes_score else "off"
        )
        col_m2.metric(
            label="Kelas 1: Diabetes", 
            value=f"{diabetes_score:.4f}",
            delta="Tertinggi" if diabetes_score > non_diabetes_score else None,
            delta_color="normal" if diabetes_score > non_diabetes_score else "off"
        )
        
        st.markdown("---")
        st.caption("Data Input Pasien:")
        st.dataframe(input_df.T, use_container_width=True)
        st.session_state['predict_button_pressed'] = False


# ==============================================================================
# 4. Fungsi Utama untuk Navigasi
# ==============================================================================
def main():
    st.set_page_config(page_title="Diagnosis Diabetes FKNN", layout="wide")
    st.title("🩺 Sistem Diagnosis Diabetes Melitus (FKNN)")

    # Memuat Model, Data, Metrik, dan Akurasi
    with st.spinner(f'Memuat data dan melatih {len(K_VALUES)} model FKNN untuk mencari yang terbaik...'):
        best_model_fknn, scaler, feature_cols, original_df, df_normalized, metrics_df, best_k = load_data_and_train_model()

    if not best_model_fknn:
        return

    if 'predict_button_pressed' not in st.session_state:
        st.session_state['predict_button_pressed'] = False
        
    # --- Navigasi Menggunakan Tabs ---
    st.header("Navigasi Aplikasi")
    
    tab_diagnosa, tab_data_und, tab_prep, tab_modelling = st.tabs(
        ["🎯 Diagnosis Diabetes", "🔬 Data Understanding", "⚙️ Preprocessing", "🤖 Modelling"]
    )
    st.markdown("---")
    
    # --- Mengontrol Konten per Tab ---
    
    with tab_diagnosa:
        # Panggilan dengan 4 argumen
        prediction_page(best_model_fknn, scaler, feature_cols, best_k)
        
    with tab_data_und:
        st.sidebar.empty() # Kosongkan sidebar saat di menu non-diagnosis
        data_understanding_page(original_df)
        
    with tab_prep:
        st.sidebar.empty()
        preprocessing_page(df_normalized)
        
    with tab_modelling:
        st.sidebar.empty()
        modelling_page(metrics_df, best_k, metrics_df['Akurasi'].max())

if __name__ == "__main__":
    main()