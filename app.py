import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# ---- Global CSS for styling to follow Default Design Guidelines ----
st.markdown(
    """
    <style>
    /* Base and layout */
    body, .css-1v3fvcr, .block-container {
        background: #ffffff;
        color: #6b7280;
        max-width: 1200px;
        margin: auto;
        padding: 3rem 2rem 6rem 2rem;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen,
            Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif;
        font-size: 18px;
    }
    /* Headings */
    h1, .stTitle {
        font-weight: 700;
        font-size: 3rem;
        color: #111827;
        margin-bottom: 0.5rem;
    }
    h2 {
        font-weight: 600;
        font-size: 2rem;
        color: #111827;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    /* Cards */
    .card {
        background: #f9fafb;
        border-radius: 0.75rem;
        box-shadow: 0 1px 3px rgb(0 0 0 / 0.1);
        padding: 2rem;
        margin-bottom: 2.5rem;
        transition: box-shadow 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 4px 12px rgb(0 0 0 / 0.15);
    }
    /* Buttons */
    .stButton > button {
        background-color: #111827;
        color: white;
        font-weight: 600;
        font-size: 1.125rem;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        transition: background-color 0.3s ease;
        border: none;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #374151;
    }
    /* Separators and layout spacing */
    .section {
        padding-top: 4rem;
        padding-bottom: 4rem;
    }
    /* File uploader customization */
    .css-1h8ys4k {
        border-radius: 0.75rem !important;
        border: 2px dashed #cbd5e1 !important;
        padding: 1rem !important;
        background: #f9fafb !important;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 class='stTitle'>Praktikum Data Mining 2025</h1>", unsafe_allow_html=True)
st.write("Teknik Informatika - Universitas Pelita Bangsa")

# Section: Upload dataset
st.markdown("<div class='section card'>", unsafe_allow_html=True)
st.header("Upload Dataset (.csv)")
uploaded_file = st.file_uploader("Pilih file CSV Anda", type=["csv"])
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Dataset berhasil diupload!")
        st.dataframe(data.head(10))
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")
else:
    st.info("Silakan upload dataset terlebih dahulu.")
st.markdown("</div>", unsafe_allow_html=True)

# Initialization variables for inputs
algorithm = None
target_column = None

if uploaded_file:
    st.markdown("<div class='section card'>", unsafe_allow_html=True)
    st.header("Pilih Algoritma dan Target")

    # Select target column for supervised algorithms
    numeric_cols = data.select_dtypes(include=['float64','int64']).columns.tolist()
    all_cols = data.columns.tolist()

    target_column = st.selectbox("Pilih kolom target (label)", options=all_cols)

    # Algorithm options based on whether target is categorical or numerical
    # We'll infer if classification or regression based on dtype of target column
    if target_column:
        if data[target_column].dtype in [float, int]:
            # For numeric target, allow regression or clustering
            algorithm = st.selectbox(
                "Pilih algoritma",
                options=[
                    "Regresi Linier",
                    "K-Means Clustering"
                ]
            )
        else:
            # For non-numeric target (categorical), classification algorithms
            algorithm = st.selectbox(
                "Pilih algoritma",
                options=[
                    "Naive Bayes",
                    "SVM",
                    "K-NN",
                    "Decision Tree",
                    "Regresi Logistik"
                ]
            )
    st.markdown("</div>", unsafe_allow_html=True)

# Button to run the algorithm
if uploaded_file and algorithm and target_column:
    st.markdown("<div class='section card'>", unsafe_allow_html=True)
    st.header("Hasil dan Visualisasi")

    # Prepare data for modeling
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Encode categorical columns if any
    X_encoded = pd.get_dummies(X)

    # Check if target is numerical or categorical
    is_regression = algorithm in ["Regresi Linier", "K-Means Clustering"]

    # If classification, encode target if needed
    if algorithm != "K-Means Clustering" and y.dtype == "object":
        y = pd.factorize(y)[0]

    # Split dataset for supervised models (except K-Means)
    if algorithm != "K-Means Clustering":
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        except Exception as e:
            st.error(f"Terjadi error saat membagi data: {e}")
            st.stop()

    model = None

    if algorithm == "Regresi Linier":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Akurasi (R² score): {r2:.4f}")
        st.write(f"Mean Squared Error: {mse:.4f}")

        # Plotting
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, color="#111827", alpha=0.7)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Nilai Aktual")
        plt.ylabel("Nilai Prediksi")
        plt.title("Plot Prediksi vs Aktual - Regresi Linier")
        st.pyplot(fig)

    elif algorithm == "Regresi Logistik":
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Akurasi: {acc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix - Regresi Logistik")
        st.pyplot(fig)

    elif algorithm == "Naive Bayes":
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Akurasi: {acc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix - Naive Bayes")
        st.pyplot(fig)

    elif algorithm == "SVM":
        model = SVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Akurasi: {acc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix - SVM")
        st.pyplot(fig)

    elif algorithm == "K-NN":
        k = st.slider("Pilih nilai K untuk K-NN", min_value=1, max_value=20, value=5)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Akurasi: {acc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title(f"Confusion Matrix - K-NN (K={k})")
        st.pyplot(fig)

    elif algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"Akurasi: {acc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix - Decision Tree")
        st.pyplot(fig)

    elif algorithm == "K-Means Clustering":
        n_clusters = st.slider("Jumlah cluster K", min_value=2, max_value=10, value=3)
        model = KMeans(n_clusters=n_clusters, random_state=42)
        try:
            model.fit(X_encoded)
            labels = model.labels_
            st.write(f"Jumlah cluster: {n_clusters}")
            st.write(f"Cluster labels:\n{labels}")

            # If there are at least 2 features, plot first two principal components for visualization
            from sklearn.decomposition import PCA
            if X_encoded.shape[1] >= 2:
                pca = PCA(n_components=2)
                pcs = pca.fit_transform(X_encoded)
                fig, ax = plt.subplots()
                scatter = ax.scatter(pcs[:,0], pcs[:,1], c=labels, cmap='tab10', alpha=0.7)
                legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend1)
                ax.set_title("Visualisasi Cluster - K-Means")
                st.pyplot(fig)
            else:
                st.info("Dataset memiliki kurang dari 2 fitur untuk visualisasi cluster.")
        except Exception as e:
            st.error(f"Error saat menjalankan K-Means: {e}")

    else:
        st.warning("Algoritma belum tersedia atau belum didukung.")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown(
    """
    <footer style='text-align:center; padding: 2rem 0; color:#9ca3af; border-top:1px solid #e5e7eb; margin-top:4rem;'>
    &copy; 2025 Universitas Pelita Bangsa - Teknik Informatika
    </footer>
    """,
    unsafe_allow_html=True,
)

