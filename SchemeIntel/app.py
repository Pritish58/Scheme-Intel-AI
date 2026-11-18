import os
import glob
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SchemeIntel AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(
    BASE_DIR,
    "SchemeIntel_KMeans_Model.pkl"
)

PREPROCESSOR_FILE = os.path.join(
    BASE_DIR,
    "SchemeIntel_Preprocessor.pkl"
)

PROFILE_FILE = os.path.join(
    BASE_DIR,
    "SchemeMatch_Cluster_Profiles.csv"
)


# ============================================================
# SAME 20 FEATURES USED FOR YOUR MODEL
# ============================================================

NUMERIC_FEATURES = [
    "economic_activities",
    "years_of_operation",
    "months_operated",
    "working_hours",
]

CATEGORICAL_FEATURES = [
    "mixed_activity",
    "major_activity_nic",
    "ownership_type",
    "education_level",
    "technical_training",
    "social_group",
    "bank_account",
    "establishment_location",
    "establishment_type",
    "nature_of_operation",
    "accounts_maintained",
    "computer_used",
    "internet_used",
    "operational_problem",
    "registered",
    "franchise",
]

FEATURES = [
    "mixed_activity",
    "major_activity_nic",
    "ownership_type",
    "education_level",
    "technical_training",
    "social_group",
    "economic_activities",
    "bank_account",
    "establishment_location",
    "establishment_type",
    "years_of_operation",
    "nature_of_operation",
    "months_operated",
    "working_hours",
    "accounts_maintained",
    "computer_used",
    "internet_used",
    "operational_problem",
    "registered",
    "franchise",
]


# ============================================================
# CUSTOM STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #071426;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #061326 0%,
            #0b2346 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border-radius: 16px;
        padding: 14px;
        border: 1px solid #dce3ee;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #172033 !important;
    }

    .hero {
        background: linear-gradient(
            110deg,
            #182f65,
            #4b32a3
        );
        border-radius: 0 0 22px 22px;
        padding: 28px 30px;
        margin-bottom: 28px;
    }

    .hero h1 {
        color: white;
        margin: 0;
    }

    .hero p {
        color: #d9e6ff;
        font-size: 18px;
        margin: 8px 0 0;
    }

    .cluster-card {
        background: #0d223a;
        border: 1px solid #294966;
        border-radius: 18px;
        padding: 20px;
    }

    .cluster-blue {
        color: #4c8bf5;
    }

    .cluster-orange {
        color: #ff7a18;
    }

    .map-note {
        background: #102b47;
        border: 1px solid #294f72;
        border-radius: 16px;
        padding: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_data_file():
    preferred = [
        "SchemeMatch_Clustered_Data.csv",
        "SchemeMatch_Clustered.csv",
        "SchemeMatch_Results.csv",
    ]

    for name in preferred:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return path

    csv_files = glob.glob(
        os.path.join(BASE_DIR, "*.csv")
    )

    # Prefer files whose names suggest clustered data.
    for path in csv_files:
        name = os.path.basename(path).lower()
        if "cluster" in name:
            return path

    return csv_files[0] if csv_files else None


DATA_FILE = find_data_file()


# ============================================================
# LOAD FILES
# ============================================================

@st.cache_resource
def load_artifact(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as file:
            return pickle.load(file)


@st.cache_data
def load_dataframe(path):
    if path and os.path.exists(path):
        return pd.read_csv(
            path,
            low_memory=False
        )
    return pd.DataFrame()


try:
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "SchemeIntel_KMeans_Model.pkl was not found."
        )

    if not os.path.exists(PREPROCESSOR_FILE):
        raise FileNotFoundError(
            "SchemeIntel_Preprocessor.pkl was not found."
        )

    kmeans = load_artifact(MODEL_FILE)
    preprocessor = load_artifact(
        PREPROCESSOR_FILE
    )

    data_df = load_dataframe(DATA_FILE)
    profile_df = load_dataframe(
        PROFILE_FILE
    )

except Exception as error:
    st.error("Could not load the SchemeIntel model files.")
    st.code(str(error))
    st.stop()


# Prevent sklearn from trying to wrap transformed output
# back into a pandas object.
try:
    preprocessor.set_output(
        transform="default"
    )
except Exception:
    pass


# ============================================================
# BASIC VALIDATION
# ============================================================

missing_features = [
    col
    for col in FEATURES
    if col not in data_df.columns
]

if missing_features:
    st.warning(
        "Some model features are not present in the clustered "
        "CSV. Prediction may not be available."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def cluster_column(df):
    if "cluster" in df.columns:
        return "cluster"

    for col in df.columns:
        if str(col).lower() == "cluster":
            return col

    return None


def get_cluster_counts():
    col = cluster_column(data_df)

    if col is None:
        return pd.Series(dtype=int)

    values = pd.to_numeric(
        data_df[col],
        errors="coerce"
    ).dropna()

    return values.astype(int).value_counts().sort_index()


def get_encoder_categories(column):
    try:
        encoder = (
            preprocessor
            .named_transformers_["categorical"]
        )

        index = CATEGORICAL_FEATURES.index(
            column
        )

        categories = encoder.categories_[index]

        return list(categories)

    except Exception:
        if column in data_df.columns:
            values = (
                data_df[column]
                .dropna()
                .unique()
                .tolist()
            )

            return sorted(
                values,
                key=lambda x: str(x)
            )

        return ["Missing"]


def convert_category_value(
    column,
    value
):
    try:
        encoder = (
            preprocessor
            .named_transformers_["categorical"]
        )

        index = CATEGORICAL_FEATURES.index(
            column
        )

        categories = np.asarray(
            encoder.categories_[index]
        )

        if len(categories) == 0:
            return value

        dtype = categories.dtype

        if np.issubdtype(
            dtype,
            np.number
        ):
            return int(float(value))

        return str(value)

    except Exception:
        return value


def prepare_input(values):
    frame = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    for col in NUMERIC_FEATURES:
        frame[col] = pd.to_numeric(
            frame[col],
            errors="coerce"
        )

        frame[col] = frame[col].fillna(0)

    for col in CATEGORICAL_FEATURES:
        frame[col] = frame[col].map(
            lambda value:
            convert_category_value(
                col,
                value
            )
        )

    return frame


def predict_enterprise(values):
    frame = prepare_input(values)

    transformed = preprocessor.transform(
        frame
    )

    cluster = int(
        kmeans.predict(
            transformed
        )[0]
    )

    distances = kmeans.transform(
        transformed
    )[0]

    return cluster, distances


def get_profile_value(
    profile,
    names,
    default="—"
):
    if profile is None:
        return default

    for name in names:
        if name in profile.index:
            return profile[name]

    return default


def get_cluster_profile(cluster):
    if profile_df.empty:
        return None

    col = cluster_column(profile_df)

    if col is None:
        return None

    values = pd.to_numeric(
        profile_df[col],
        errors="coerce"
    )

    result = profile_df[
        values == int(cluster)
    ]

    if result.empty:
        return None

    return result.iloc[0]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h2>🧠 SchemeIntel AI</h2>
        <p>Enterprise Intelligence Platform</p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Analyze Enterprise",
            "Cluster Explorer",
            "Enterprise Map",
            "Scheme Intelligence",
            "Model Insights",
            "Data Information",
            "About",
        ]
    )

    st.divider()

    st.success("🟢 Model Online")

    st.caption(
        f"MiniBatch K-Means • "
        f"{kmeans.n_clusters} clusters"
    )

    st.caption(
        "ASUSE 2023–24 • Level 02"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>SchemeIntel AI</h1>
        <p>
            AI-powered enterprise profiling &
            government scheme intelligence
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    counts = get_cluster_counts()
    total_records = len(data_df)

    try:
        processed_features = (
            preprocessor
            .transform(
                data_df[FEATURES].head(1)
            )
            .shape[1]
        )
    except Exception:
        processed_features = "—"

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Enterprise Records",
        f"{total_records:,}"
    )

    c2.metric(
        "ML Clusters",
        kmeans.n_clusters
    )

    c3.metric(
        "Original Features",
        20
    )

    c4.metric(
        "Processed Features",
        processed_features
    )

    c5.metric(
        "Model",
        "MiniBatch K-Means"
    )

    st.header(
        "🚀 Enterprise Intelligence Overview"
    )

    left, right = st.columns(
        [2.2, 1]
    )

    with left:

        st.subheader(
            "Enterprise Distribution by Cluster"
        )

        if not counts.empty:

            chart = pd.DataFrame({
                "Cluster": [
                    f"Cluster {int(x)}"
                    for x in counts.index
                ],
                "Enterprise Count":
                    counts.values
            })

            fig = px.bar(
                chart,
                x="Cluster",
                y="Enterprise Count",
                color="Cluster",
                text="Enterprise Count",
                color_discrete_map={
                    "Cluster 0": "#4c8bf5",
                    "Cluster 1": "#ff7a18"
                }
            )

            fig.update_traces(
                texttemplate="%{text:,}",
                textposition="outside"
            )

            fig.update_layout(
                template="plotly_dark",
                height=450,
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with right:

        st.subheader(
            "⚡ Quick Actions"
        )

        st.info(
            """
            Use the navigation to:

            • Analyze a new enterprise

            • Understand Cluster 0 and Cluster 1

            • Explore the enterprise map

            • Inspect the ML pipeline

            • Review scheme-screening information
            """
        )

    st.divider()

    st.subheader(
        "📊 Cluster Summary"
    )

    if not counts.empty:

        summary = []

        total = counts.sum()

        for cluster, count in counts.items():

            summary.append({
                "Cluster":
                    f"Cluster {int(cluster)}",
                "Enterprises":
                    int(count),
                "Share (%)":
                    round(
                        count / total * 100,
                        2
                    )
            })

        st.dataframe(
            pd.DataFrame(summary),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# ANALYZE ENTERPRISE
# ============================================================

elif page == "Analyze Enterprise":

    st.header(
        "🔍 Analyze a New Enterprise"
    )

    st.caption(
        "Enter the same 20 enterprise characteristics "
        "used during model training."
    )

    values = {}

    with st.form(
        "enterprise_analysis_form"
    ):

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Basic Information",
                "Enterprise Characteristics",
                "Operations",
                "Digital & Registration",
            ]
        )

        with tab1:

            a, b, c = st.columns(3)

            with a:
                values[
                    "years_of_operation"
                ] = st.number_input(
                    "Years of operation",
                    min_value=0,
                    max_value=100,
                    value=3
                )

            with b:
                values[
                    "months_operated"
                ] = st.number_input(
                    "Months operated",
                    min_value=1,
                    max_value=12,
                    value=12
                )

            with c:
                values[
                    "working_hours"
                ] = st.number_input(
                    "Working hours per day",
                    min_value=0.0,
                    max_value=24.0,
                    value=8.0,
                    step=0.5
                )

            values[
                "economic_activities"
            ] = st.number_input(
                "Number of economic activities",
                min_value=0,
                max_value=50,
                value=1
            )

            values[
                "mixed_activity"
            ] = st.selectbox(
                "Mixed activity",
                get_encoder_categories(
                    "mixed_activity"
                )
            )

            values[
                "major_activity_nic"
            ] = st.selectbox(
                "Major activity NIC",
                get_encoder_categories(
                    "major_activity_nic"
                )
            )

        with tab2:

            a, b, c = st.columns(3)

            with a:

                values[
                    "ownership_type"
                ] = st.selectbox(
                    "Ownership type",
                    get_encoder_categories(
                        "ownership_type"
                    )
                )

                values[
                    "education_level"
                ] = st.selectbox(
                    "Education level",
                    get_encoder_categories(
                        "education_level"
                    )
                )

                values[
                    "technical_training"
                ] = st.selectbox(
                    "Technical / vocational training",
                    get_encoder_categories(
                        "technical_training"
                    )
                )

            with b:

                values[
                    "social_group"
                ] = st.selectbox(
                    "Social group",
                    get_encoder_categories(
                        "social_group"
                    )
                )

                values[
                    "bank_account"
                ] = st.selectbox(
                    "Bank account",
                    get_encoder_categories(
                        "bank_account"
                    )
                )

                values[
                    "establishment_location"
                ] = st.selectbox(
                    "Establishment location",
                    get_encoder_categories(
                        "establishment_location"
                    )
                )

            with c:

                values[
                    "establishment_type"
                ] = st.selectbox(
                    "Establishment type",
                    get_encoder_categories(
                        "establishment_type"
                    )
                )

                values[
                    "nature_of_operation"
                ] = st.selectbox(
                    "Nature of operation",
                    get_encoder_categories(
                        "nature_of_operation"
                    )
                )

                values[
                    "accounts_maintained"
                ] = st.selectbox(
                    "Accounts maintained",
                    get_encoder_categories(
                        "accounts_maintained"
                    )
                )

        with tab3:

            a, b = st.columns(2)

            with a:
                values[
                    "operational_problem"
                ] = st.selectbox(
                    "Operational problem",
                    get_encoder_categories(
                        "operational_problem"
                    )
                )

                values[
                    "computer_used"
                ] = st.selectbox(
                    "Computer used",
                    get_encoder_categories(
                        "computer_used"
                    )
                )

            with b:
                values[
                    "internet_used"
                ] = st.selectbox(
                    "Internet used",
                    get_encoder_categories(
                        "internet_used"
                    )
                )

        with tab4:

            a, b = st.columns(2)

            with a:
                values[
                    "registered"
                ] = st.selectbox(
                    "Registered under an Act / authority",
                    get_encoder_categories(
                        "registered"
                    )
                )

            with b:
                values[
                    "franchise"
                ] = st.selectbox(
                    "Franchise outlet",
                    get_encoder_categories(
                        "franchise"
                    )
                )

        submitted = st.form_submit_button(
            "🚀 Analyze Enterprise",
            type="primary",
            use_container_width=True
        )

    if submitted:

        try:

            cluster, distances = predict_enterprise(
                values
            )

            st.session_state[
                "enterprise_prediction"
            ] = {
                "cluster": cluster,
                "distances": distances,
                "values": values,
            }

        except Exception as error:

            st.error(
                "Prediction failed."
            )

            st.code(
                str(error)
            )

    if (
        "enterprise_prediction"
        in st.session_state
    ):

        result = st.session_state[
            "enterprise_prediction"
        ]

        cluster = result["cluster"]
        distances = result["distances"]

        st.divider()

        st.header(
            "🧠 AI Analysis Result"
        )

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "Predicted Cluster",
                f"Cluster {cluster}"
            )

        with b:
            st.metric(
                "Distance to Assigned Cluster",
                f"{distances[cluster]:.4f}"
            )

        with c:

            counts = get_cluster_counts()

            count = int(
                counts.get(
                    cluster,
                    0
                )
            )

            st.metric(
                "Enterprises in Cluster",
                f"{count:,}"
            )

        distance_df = pd.DataFrame({
            "Cluster": [
                f"Cluster {i}"
                for i in range(
                    len(distances)
                )
            ],
            "Distance": distances
        })

        st.subheader(
            "📏 Distance to Every Cluster"
        )

        fig = px.bar(
            distance_df,
            x="Cluster",
            y="Distance",
            text_auto=".4f",
            color="Cluster",
            color_discrete_map={
                "Cluster 0": "#4c8bf5",
                "Cluster 1": "#ff7a18"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            height=350,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.warning(
            "Cluster assignment measures similarity to "
            "learned enterprise groups. It does not establish "
            "legal eligibility for any government scheme."
        )


# ============================================================
# CLUSTER EXPLORER
# ============================================================

elif page == "Cluster Explorer":

    st.header(
        "🧩 Cluster Explorer"
    )

    st.caption(
        "Understand the actual enterprise groups discovered "
        "by MiniBatch K-Means."
    )

    counts = get_cluster_counts()

    if counts.empty:

        st.warning(
            "Cluster information is unavailable."
        )

    else:

        selected_cluster = st.selectbox(
            "Select cluster",
            [
                int(x)
                for x in counts.index
            ],
            format_func=lambda x:
                f"Cluster {x}"
        )

        count = int(
            counts.get(
                selected_cluster,
                0
            )
        )

        share = (
            count / counts.sum() * 100
        )

        st.metric(
            "Enterprise Count",
            f"{count:,}"
        )

        st.metric(
            "Cluster Share",
            f"{share:.2f}%"
        )

        if "cluster" in data_df.columns:

            working = data_df.copy()

            working["cluster"] = pd.to_numeric(
                working["cluster"],
                errors="coerce"
            )

            numeric_profile = (
                working
                .groupby("cluster")[
                    [
                        x for x in NUMERIC_FEATURES
                        if x in working.columns
                    ]
                ]
                .mean()
                .reset_index()
            )

            long_df = numeric_profile.melt(
                id_vars="cluster",
                var_name="Feature",
                value_name="Average"
            )

            long_df["Cluster"] = (
                "Cluster "
                + long_df["cluster"]
                .astype(int)
                .astype(str)
            )

            long_df["Feature"] = (
                long_df["Feature"]
                .str.replace(
                    "_",
                    " "
                )
                .str.title()
            )

            st.subheader(
                "📊 Numeric Characteristics"
            )

            fig = px.bar(
                long_df,
                x="Feature",
                y="Average",
                color="Cluster",
                barmode="group",
                text_auto=".2f",
                color_discrete_map={
                    "Cluster 0": "#4c8bf5",
                    "Cluster 1": "#ff7a18"
                }
            )

            fig.update_layout(
                template="plotly_dark",
                height=450,
                xaxis_tickangle=-25
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader(
                "🏷️ Categorical Characteristics"
            )

            category = st.selectbox(
                "Select a characteristic",
                CATEGORICAL_FEATURES,
                format_func=lambda x:
                    x.replace(
                        "_",
                        " "
                    ).title()
            )

            temp = data_df[
                [
                    "cluster",
                    category
                ]
            ].copy()

            temp["cluster"] = pd.to_numeric(
                temp["cluster"],
                errors="coerce"
            )

            temp[category] = (
                temp[category]
                .astype(str)
            )

            distribution = (
                temp
                .groupby(
                    [
                        "cluster",
                        category
                    ]
                )
                .size()
                .reset_index(
                    name="Count"
                )
            )

            distribution["Percentage"] = (
                distribution
                .groupby("cluster")[
                    "Count"
                ]
                .transform(
                    lambda x:
                    x / x.sum() * 100
                )
            )

            distribution["Cluster"] = (
                "Cluster "
                + distribution["cluster"]
                .astype(int)
                .astype(str)
            )

            fig = px.bar(
                distribution,
                x=category,
                y="Percentage",
                color="Cluster",
                barmode="group",
                color_discrete_map={
                    "Cluster 0": "#4c8bf5",
                    "Cluster 1": "#ff7a18"
                }
            )

            fig.update_layout(
                template="plotly_dark",
                height=450,
                yaxis_title="Percentage (%)",
                xaxis_title=category.replace(
                    "_",
                    " "
                ).title()
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# ENTERPRISE MAP
# ============================================================

elif page == "Enterprise Map":

    st.header("🗺️ Enterprise Cluster Map")

    st.caption(
        "2D projection of sampled ASUSE enterprises. "
        "Each point represents one enterprise."
    )

    if data_df.empty or "cluster" not in data_df.columns:
        st.error("The clustered dataset is required for the map.")
        st.stop()

    # --------------------------------------------------------
    # Map settings
    # --------------------------------------------------------
    max_sample = min(3000, len(data_df))
    min_sample = min(500, max_sample)
    default_sample = min(2500, max_sample)

    if max_sample <= min_sample:
        sample_size = max_sample
    else:
        sample_size = st.slider(
            "Map sample size",
            min_value=min_sample,
            max_value=max_sample,
            value=default_sample,
            step=500 if max_sample >= 1000 else 100
        )

    try:
        # ====================================================
        # 1. SAMPLE FROM THE REAL DATA
        # ====================================================
        working = data_df.copy()

        working["cluster"] = pd.to_numeric(
            working["cluster"],
            errors="coerce"
        )

        working = working.dropna(subset=["cluster"])
        working["cluster"] = working["cluster"].astype(int)

        # Project only the two actual learned clusters.
        working = working[working["cluster"].isin([0, 1])].copy()

        if working.empty:
            st.error("Cluster 0 and Cluster 1 were not found in the data.")
            st.stop()

        map_df = working.sample(
            n=min(sample_size, len(working)),
            random_state=42
        ).copy()

        # ====================================================
        # 2. PREPARE THE SAME 20 FEATURES USED BY THE MODEL
        # ====================================================
        model_input = map_df[FEATURES].copy()

        # Make the dataframe safe for the saved preprocessor.
        for col in NUMERIC_FEATURES:
            model_input[col] = pd.to_numeric(
                model_input[col],
                errors="coerce"
            )

            model_input[col] = model_input[col].fillna(0)

        for col in CATEGORICAL_FEATURES:
            model_input[col] = model_input[col].fillna("Missing")

            # Convert values to strings first only for object/category
            # columns. Numeric categorical columns are restored below
            # to the encoder's expected dtype.
            try:
                encoder = preprocessor.named_transformers_["categorical"]
                idx = CATEGORICAL_FEATURES.index(col)
                categories = np.asarray(encoder.categories_[idx])

                if np.issubdtype(categories.dtype, np.number):
                    numeric_values = pd.to_numeric(
                        model_input[col],
                        errors="coerce"
                    )
                    fallback = (
                        float(categories[0])
                        if len(categories)
                        else 0
                    )
                    model_input[col] = numeric_values.fillna(
                        fallback
                    ).astype(categories.dtype)
                else:
                    model_input[col] = model_input[col].astype(str)

            except Exception:
                model_input[col] = model_input[col].astype(str)

        # ====================================================
        # 3. TRANSFORM THE FULL 20-FEATURE SPACE
        # ====================================================
        transformed = preprocessor.transform(model_input)

        # PCA directly on the full transformed feature space.
        # This is the important difference from the old map:
        # the old map used only 4 numeric columns.
        #
        # The K-Means model was trained on the transformed
        # 20-feature representation, so this visualization
        # starts from that same representation.
        if hasattr(transformed, "toarray"):
            dense_for_pca = transformed.toarray()
        else:
            dense_for_pca = np.asarray(transformed)

        # PCA is safe because the map is only a sample.
        pca = PCA(
            n_components=2,
            random_state=42
        )

        coords = pca.fit_transform(
            dense_for_pca
        )

        map_df["Component 1"] = coords[:, 0]
        map_df["Component 2"] = coords[:, 1]

        # ====================================================
        # 4. PUT CLUSTER 0 ON LEFT AND CLUSTER 1 ON RIGHT
        #
        # This only rotates the visualization horizontally.
        # It does NOT change the cluster labels.
        # ====================================================
        mean_0 = map_df.loc[
            map_df["cluster"] == 0,
            "Component 1"
        ].mean()

        mean_1 = map_df.loc[
            map_df["cluster"] == 1,
            "Component 1"
        ].mean()

        if mean_0 > mean_1:
            map_df["Component 1"] *= -1

        # ====================================================
        # 5. COUNTS FROM THE FULL CLUSTERED DATASET
        # ====================================================
        full_clusters = pd.to_numeric(
            data_df["cluster"],
            errors="coerce"
        ).dropna().astype(int)

        counts = full_clusters.value_counts().sort_index()

        total = int(counts.sum())

        count_0 = int(counts.get(0, 0))
        count_1 = int(counts.get(1, 0))

        share_0 = count_0 / total * 100 if total else 0
        share_1 = count_1 / total * 100 if total else 0

        # ====================================================
        # 6. GET THE TWO POINT CLOUDS
        # ====================================================
        points_0 = map_df[map_df["cluster"] == 0].copy()
        points_1 = map_df[map_df["cluster"] == 1].copy()

        # ====================================================
        # 7. PLOT
        # ====================================================
        fig = go.Figure()

        # Cluster 0
        fig.add_trace(
            go.Scatter(
                x=points_0["Component 1"],
                y=points_0["Component 2"],
                mode="markers",
                name="Cluster 0",
                marker=dict(
                    size=7,
                    color="#4C8BF5",
                    opacity=0.72,
                    line=dict(
                        width=0.35,
                        color="#BFD9FF"
                    )
                ),
                hovertemplate=(
                    "<b>Cluster 0</b><br>"
                    "Component 1: %{x:.2f}<br>"
                    "Component 2: %{y:.2f}<br>"
                    "<extra></extra>"
                )
            )
        )

        # Cluster 1
        fig.add_trace(
            go.Scatter(
                x=points_1["Component 1"],
                y=points_1["Component 2"],
                mode="markers",
                name="Cluster 1",
                marker=dict(
                    size=7,
                    color="#FF7A18",
                    opacity=0.72,
                    line=dict(
                        width=0.35,
                        color="#FFD0AA"
                    )
                ),
                hovertemplate=(
                    "<b>Cluster 1</b><br>"
                    "Component 1: %{x:.2f}<br>"
                    "Component 2: %{y:.2f}<br>"
                    "<extra></extra>"
                )
            )
        )

        # ====================================================
        # 8. DRAW APPROXIMATE CIRCULAR CLUSTER BOUNDARIES
        # ====================================================
        def add_cluster_circle(
            figure,
            points,
            line_color,
            fill_color
        ):
            if len(points) < 3:
                return

            x = points["Component 1"].to_numpy()
            y = points["Component 2"].to_numpy()

            cx = np.mean(x)
            cy = np.mean(y)

            radius_values = np.sqrt(
                (x - cx) ** 2 +
                (y - cy) ** 2
            )

            radius = np.percentile(
                radius_values,
                97
            )

            radius = max(radius, 0.5)

            theta = np.linspace(
                0,
                2 * np.pi,
                300
            )

            circle_x = (
                cx + radius * np.cos(theta)
            )
            circle_y = (
                cy + radius * np.sin(theta)
            )

            figure.add_trace(
                go.Scatter(
                    x=circle_x,
                    y=circle_y,
                    mode="lines",
                    line=dict(
                        color=line_color,
                        width=2.5,
                        dash="dash"
                    ),
                    fill="toself",
                    fillcolor=fill_color,
                    hoverinfo="skip",
                    showlegend=False
                )
            )

        add_cluster_circle(
            fig,
            points_0,
            "#4C8BF5",
            "rgba(76,139,245,0.08)"
        )

        add_cluster_circle(
            fig,
            points_1,
            "#FF7A18",
            "rgba(255,122,24,0.08)"
        )

        # ====================================================
        # 9. CLUSTER CENTERS
        # ====================================================
        center_0_x = points_0["Component 1"].mean()
        center_0_y = points_0["Component 2"].mean()

        center_1_x = points_1["Component 1"].mean()
        center_1_y = points_1["Component 2"].mean()

        fig.add_trace(
            go.Scatter(
                x=[center_0_x],
                y=[center_0_y],
                mode="markers",
                name="Cluster 0 Center",
                marker=dict(
                    symbol="x",
                    size=18,
                    color="white",
                    line=dict(
                        width=3,
                        color="#071426"
                    )
                ),
                hovertemplate=(
                    "<b>Cluster 0 Center</b>"
                    "<extra></extra>"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[center_1_x],
                y=[center_1_y],
                mode="markers",
                name="Cluster 1 Center",
                marker=dict(
                    symbol="x",
                    size=18,
                    color="white",
                    line=dict(
                        width=3,
                        color="#071426"
                    )
                ),
                hovertemplate=(
                    "<b>Cluster 1 Center</b>"
                    "<extra></extra>"
                )
            )
        )

        # ====================================================
        # 10. LABEL BOXES
        # ====================================================
        fig.add_annotation(
            x=center_0_x,
            y=center_0_y,
            text=(
                "<b>Cluster 0</b><br>"
                f"{count_0:,} enterprises<br>"
                f"{share_0:.2f}%"
            ),
            showarrow=True,
            arrowhead=2,
            arrowcolor="#4C8BF5",
            ax=0,
            ay=-135,
            bgcolor="rgba(7,24,48,0.95)",
            bordercolor="#4C8BF5",
            borderwidth=2,
            borderpad=10,
            font=dict(
                color="white",
                size=14
            )
        )

        fig.add_annotation(
            x=center_1_x,
            y=center_1_y,
            text=(
                "<b>Cluster 1</b><br>"
                f"{count_1:,} enterprises<br>"
                f"{share_1:.2f}%"
            ),
            showarrow=True,
            arrowhead=2,
            arrowcolor="#FF7A18",
            ax=0,
            ay=-135,
            bgcolor="rgba(48,25,10,0.95)",
            bordercolor="#FF7A18",
            borderwidth=2,
            borderpad=10,
            font=dict(
                color="white",
                size=14
            )
        )

        # ====================================================
        # 11. FINAL FIGURE STYLE
        # ====================================================
        fig.update_layout(
            template="plotly_dark",
            height=720,
            paper_bgcolor="#071426",
            plot_bgcolor="#0B1B2D",
            title=dict(
                text="<b>Enterprise Cluster Visualization</b>",
                x=0.5,
                font=dict(size=23)
            ),
            xaxis=dict(
                title="Component 1",
                gridcolor="rgba(150,180,210,0.20)",
                zeroline=False
            ),
            yaxis=dict(
                title="Component 2",
                gridcolor="rgba(150,180,210,0.20)",
                zeroline=False
            ),
            legend=dict(
                title="Enterprise Cluster",
                bgcolor="rgba(8,24,44,0.94)",
                bordercolor="#52677F",
                borderwidth=1
            ),
            margin=dict(
                l=70,
                r=30,
                t=90,
                b=70
            ),
            hovermode="closest"
        )

        # ====================================================
        # 12. MAP + SUMMARY
        # ====================================================
        left, right = st.columns([3.4, 1])

        with left:
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "responsive": True
                }
            )

        with right:

            st.markdown(
                f"""
                <div class="cluster-card">

                <h2>📊 Cluster Summary</h2>

                <h3 class="cluster-blue">
                    🔵 Cluster 0
                </h3>

                <p>
                    <b>Enterprises:</b> {count_0:,}
                </p>

                <p>
                    <b>Share:</b> {share_0:.2f}%
                </p>

                <hr>

                <h3 class="cluster-orange">
                    🟠 Cluster 1
                </h3>

                <p>
                    <b>Enterprises:</b> {count_1:,}
                </p>

                <p>
                    <b>Share:</b> {share_1:.2f}%
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### 📖 What do the axes mean?")

            st.info(
                """
                **Component 1** and **Component 2** are the
                two PCA directions used only to visualize the
                high-dimensional enterprise data.

                They are NOT K values.

                **K = 2**

                🔵 Cluster 0

                🟠 Cluster 1
                """
            )

        # ====================================================
        # 13. EXPLANATION CARDS
        # ====================================================
        st.divider()

        a, b, c = st.columns(3)

        with a:
            st.markdown(
                """
                ### 🔵 Map Interpretation

                Each point represents one sampled ASUSE
                enterprise.

                Blue points belong to **Cluster 0**.
                Orange points belong to **Cluster 1**.

                The **X** marks the visual center.
                """
            )

        with b:
            st.markdown(
                f"""
                ### ⚙️ Model Information

                **Clustering Algorithm:** MiniBatch K-Means

                **Number of Clusters (K):**
                {kmeans.n_clusters}

                **Features Used:** 20 enterprise features

                **Map Sample:** {len(map_df):,} enterprises
                """
            )

        with c:
            st.markdown(
                """
                ### 📌 Important Note

                The circles are visual boundaries showing the
                approximate spread of each sampled cluster.

                The K-Means model itself works in the full
                transformed feature space, not in this 2D map.
                """
            )

        st.info(
            "This map is now generated from the same 20-feature "
            "preprocessed representation used by the model, "
            "rather than only four numeric variables."
        )


    except Exception as error:

        st.error(
            "The enterprise cluster map could not be generated."
        )
        st.code(str(error))

        st.info(
            "The rest of the dashboard remains available. "
            "Check the model/preprocessor files if the map cannot be created."
        )


# ============================================================
# SCHEME INTELLIGENCE
# ============================================================

elif page == "Scheme Intelligence":

    st.header(
        "🏛️ Scheme Intelligence"
    )

    st.warning(
        "This is a scheme-screening layer, not a legal "
        "eligibility decision. Official scheme criteria "
        "should always be verified with the relevant "
        "government authority."
    )

    schemes = pd.DataFrame({
        "Scheme / Pathway": [
            "MUDRA",
            "PMEGP",
            "CGTMSE",
            "Stand-Up India"
        ],
        "Purpose": [
            "Potential micro-enterprise finance pathway",
            "Potential new-enterprise support pathway",
            "Potential credit-guarantee pathway",
            "Potential targeted finance pathway"
        ]
    })

    st.dataframe(
        schemes,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":

    st.header(
        "📈 Model Insights"
    )

    try:
        processed_features = (
            preprocessor
            .transform(
                data_df[FEATURES].head(1)
            )
            .shape[1]
        )
    except Exception:
        processed_features = "—"

    a, b, c, d = st.columns(4)

    a.metric(
        "Algorithm",
        "MiniBatch K-Means"
    )

    b.metric(
        "Clusters",
        kmeans.n_clusters
    )

    c.metric(
        "Processed Features",
        processed_features
    )

    d.metric(
        "Original Features",
        20
    )

    st.subheader(
        "🔄 Complete ML Pipeline"
    )

    st.code(
        """
ASUSE Level-02 enterprise data
            ↓
20 selected enterprise features
            ↓
Missing-value handling
            ↓
StandardScaler + OneHotEncoder
            ↓
Transformed feature space
            ↓
MiniBatch K-Means
            ↓
Learned enterprise clusters
            ↓
Cluster profiles
            ↓
New enterprise
            ↓
Same saved preprocessing pipeline
            ↓
Closest learned cluster
        """
    )

    st.info(
        "The dashboard uses the saved model. "
        "It does not retrain K-Means when a new enterprise "
        "is analyzed."
    )


# ============================================================
# DATA INFORMATION
# ============================================================

elif page == "Data Information":

    st.header(
        "📚 Data Information"
    )

    counts = get_cluster_counts()

    a, b, c, d = st.columns(4)

    a.metric(
        "Rows",
        f"{len(data_df):,}"
    )

    b.metric(
        "Model Features",
        20
    )

    c.metric(
        "Clusters",
        kmeans.n_clusters
    )

    d.metric(
        "Dataset Level",
        "02"
    )

    st.write(
        """
        The project uses the Level-02 ASUSE 2023–24
        enterprise profile dataset. The saved clustered
        dataset is used for exploration, while the saved
        preprocessor and K-Means model are used for new
        enterprise prediction.
        """
    )

    st.subheader(
        "Model Feature Dictionary"
    )

    dictionary = pd.DataFrame({
        "Feature": [
            x.replace(
                "_",
                " "
            ).title()
            for x in FEATURES
        ],
        "Column": FEATURES,
        "Type": [
            "Numeric"
            if x in NUMERIC_FEATURES
            else "Categorical"
            for x in FEATURES
        ]
    })

    st.dataframe(
        dictionary,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        data_df.head(100),
        use_container_width=True,
        height=450
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.header(
        "ℹ️ About SchemeIntel AI"
    )

    st.markdown(
        """
        ## SchemeIntel AI

        SchemeIntel AI is an enterprise intelligence
        platform that uses unsupervised machine learning
        to discover groups of similar enterprises.

        ### Project workflow

        **ASUSE 2023–24**

        ↓

        **Feature Selection**

        ↓

        **Missing Value Handling**

        ↓

        **Scaling + One-Hot Encoding**

        ↓

        **MiniBatch K-Means**

        ↓

        **Enterprise Cluster Profiles**

        ↓

        **New Enterprise Input**

        ↓

        **Saved Preprocessor**

        ↓

        **Predicted Cluster**

        ↓

        **Separate Scheme Screening**

        ### Important distinction

        K-Means predicts a cluster based on similarity.

        It does not determine legal eligibility for a
        government scheme.
        """
    )

    st.success(
        "SchemeIntel AI • ASUSE 2023–24 • Level 02"
    )
