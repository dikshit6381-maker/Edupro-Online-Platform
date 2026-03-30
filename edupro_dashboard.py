"""
EduPro – Student Segmentation & Personalized Course Recommendation Dashboard
============================================================================
Data Analyst Presentation | All findings from the EduPro Online Platform notebook.

Run with:
    streamlit run edupro_dashboard.py

Requires:
    pip install streamlit plotly pandas numpy scikit-learn scipy openpyxl
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPro – Student Segmentation Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── THEME ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#0d0d1a}
[data-testid="stSidebar"]{background:#16162a;border-right:1px solid #252540}
[data-testid="stSidebar"] *{color:#8080a8 !important}
[data-testid="metric-container"]{background:#16162a;border:1px solid #252540;
  border-radius:10px;padding:.85rem 1rem}
[data-testid="metric-container"] [data-testid="stMetricValue"]
  {color:#7c73ff;font-size:1.4rem;font-weight:700}
[data-testid="metric-container"] [data-testid="stMetricLabel"]
  {color:#8080a8;font-size:.72rem}
h1{color:#7c73ff !important;border-bottom:2px solid #7c73ff22;padding-bottom:6px}
h2,h3{color:#e0e0ff !important}
.stTabs [data-baseweb="tab"]{background:#16162a;border:1px solid #252540;
  color:#8080a8;border-radius:18px;font-size:.83rem;padding:.3rem .9rem}
.stTabs [aria-selected="true"]{background:#7c73ff !important;
  color:#fff !important;border-color:#7c73ff !important}
.stTabs [data-baseweb="tab-list"]{gap:.4rem}
.stSelectbox [data-baseweb="select"] > div{background:#16162a;border-color:#252540}
.stSelectbox [data-baseweb="select"] > div > div{color:#e0e0ff}
footer{display:none}
div[data-testid="stDataFrame"]{background:#16162a;border-radius:10px}
div[data-testid="stDataFrame"] th{background:#252540;color:#7c73ff}
div[data-testid="stDataFrame"] td{color:#e0e0ff}
div.stButton > button{
  background:#7c73ff !important;color:#fff !important;
  border:none !important;border-radius:8px !important;
  font-weight:600 !important;font-size:.88rem !important;
  padding:.45rem 1rem !important;
}
div.stButton > button:hover{background:#6b62f0 !important}
div[data-baseweb="input"] > div{
  background:#16162a !important;border:1px solid #252540 !important;
  border-radius:8px !important;color:#e0e0ff !important;
}
div[data-baseweb="input"] input{color:#e0e0ff !important;font-size:.9rem !important;}
</style>
""", unsafe_allow_html=True)

# ─── PALETTE ──────────────────────────────────────────────────────────────────
CLUSTER_COLORS  = ["#f7971e", "#6c63ff", "#ff6584", "#43e97b"]
CLUSTER_ICONS   = ["💰", "⚡", "🌐", "💼"]
CLUSTER_NAMES   = ["Budget Learner", "Power Learner", "Casual Explorer", "Career Focuser"]
CLUSTER_ACTIONS = [
    "Discount bundles & limited-time offers",
    "Subscription tier & advanced content paths",
    "Free-to-paid nudges after first completion",
    "Premium certification bundle upsell",
]
CLUSTER_STRATEGIES = [
    "Mix of free and paid courses. Moderate spender — target with bundles & discounts.",
    "Highly engaged multi-category learner. Prime candidate for subscription plan upsell.",
    "Single free course only. High churn risk — needs conversion nudge.",
    "100% paid learner. Career-focused — upsell premium certifications.",
]


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def dark_fig(fig, height=320, t=36, b=20, l=20, r=20):
    fig.update_layout(
        height=height, paper_bgcolor="#16162a", plot_bgcolor="#16162a",
        font=dict(color="#e0e0ff", size=11),
        margin=dict(t=t, b=b, l=l, r=r),
        legend=dict(font=dict(color="#e0e0ff", size=10), bgcolor="#16162a",
                    bordercolor="#252540"),
        xaxis=dict(gridcolor="#252540", color="#8080a8", linecolor="#252540",
                   zerolinecolor="#252540"),
        yaxis=dict(gridcolor="#252540", color="#8080a8", linecolor="#252540",
                   zerolinecolor="#252540"),
    )
    return fig


def info_box(text, color="#7c73ff"):
    st.markdown(
        f"<div style='background:{color}12;border:1px solid {color}44;border-radius:8px;"
        f"padding:.65rem .9rem;font-size:.82rem;color:#a0a0c0;margin:.5rem 0;'>{text}</div>",
        unsafe_allow_html=True)


def cluster_badge(ci):
    return (f"<span style='background:{CLUSTER_COLORS[ci]};color:#fff;"
            f"padding:.18rem .55rem;border-radius:12px;font-size:.8rem;"
            f"font-weight:700;'>{CLUSTER_ICONS[ci]} C{ci}: {CLUSTER_NAMES[ci]}</span>")


# ─── DATA LOADING & PROCESSING ────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & processing EduPro data…")
def load_and_process():
    """
    Loads all three sheets from the Excel file, engineers features,
    runs KMeans (K=4) and Hierarchical clustering, and returns
    everything needed for every dashboard page.
    """
    FILE = "EduPro Online Platform.xlsx"

    # ── Load sheets ──────────────────────────────────────────────────────────
    users_df   = pd.read_excel(FILE, sheet_name="Users")
    courses_df = pd.read_excel(FILE, sheet_name="Courses")
    trans_df   = pd.read_excel(FILE, sheet_name="Transactions")

    trans_df["TransactionDate"] = pd.to_datetime(trans_df["TransactionDate"])

    # ── Merge for full transaction detail ────────────────────────────────────
    full_df = trans_df.merge(courses_df, on="CourseID", how="left")

    # ── Feature engineering per user ─────────────────────────────────────────
    # 1. total_enrollments
    total_enroll = full_df.groupby("UserID").size().rename("total_enrollments")

    # 2. avg_spending
    avg_spend = full_df.groupby("UserID")["Amount"].mean().rename("avg_spending")

    # 3. total_spending
    total_spend = full_df.groupby("UserID")["Amount"].sum().rename("total_spending")

    # 4. avg_course_rating
    avg_rating = full_df.groupby("UserID")["CourseRating"].mean().rename("avg_course_rating")

    # 5. diversity_score (unique categories per user)
    diversity = full_df.groupby("UserID")["CourseCategory"].nunique().rename("diversity_score")

    # 6. enrollment_frequency (enrollments / active days span)
    trans_span = full_df.groupby("UserID")["TransactionDate"].agg(
        lambda x: (x.max() - x.min()).days + 1
    ).rename("active_days")
    enroll_freq = (total_enroll / trans_span).rename("enrollment_frequency")
    enroll_freq = enroll_freq.replace([np.inf, -np.inf], 1.0)

    # 7. paid_ratio
    full_df["is_paid"] = (full_df["Amount"] > 0).astype(int)
    paid_ratio = full_df.groupby("UserID")["is_paid"].mean().rename("paid_ratio")

    # 8. Learning Depth Index (fraction of Advanced + Intermediate courses)
    full_df["is_adv_inter"] = full_df["CourseLevel"].isin(["Advanced", "Intermediate"]).astype(int)
    ldi = full_df.groupby("UserID")["is_adv_inter"].mean().rename("ldi")

    # 9. avg_courses_per_cat
    courses_per_cat = full_df.groupby(["UserID", "CourseCategory"]).size().reset_index()
    avg_per_cat = courses_per_cat.groupby("UserID")[0].mean().rename("avg_courses_per_cat")

    # 10. avg_course_duration
    avg_duration = full_df.groupby("UserID")["CourseDuration"].mean().rename("avg_course_duration")

    # ── Preferred category / level ────────────────────────────────────────────
    pref_cat = (
        full_df.groupby(["UserID", "CourseCategory"]).size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("UserID")
        .set_index("UserID")["CourseCategory"]
        .rename("preferred_category")
    )
    pref_lvl = (
        full_df.groupby(["UserID", "CourseLevel"]).size()
        .reset_index(name="cnt")
        .sort_values("cnt", ascending=False)
        .drop_duplicates("UserID")
        .set_index("UserID")["CourseLevel"]
        .rename("preferred_level")
    )

    # ── Combine features ──────────────────────────────────────────────────────
    feat_df = pd.concat([
        total_enroll, avg_spend, total_spend, avg_rating, diversity,
        enroll_freq, paid_ratio, ldi, avg_per_cat, avg_duration,
        pref_cat, pref_lvl,
    ], axis=1).reset_index()
    feat_df = feat_df.fillna(0)

    # Merge user demographics
    feat_df = feat_df.merge(users_df[["UserID", "Age", "Gender"]], on="UserID", how="left")

    # ── Scale 7 clustering features ───────────────────────────────────────────
    FEAT_COLS = [
        "total_enrollments", "avg_spending", "avg_course_rating",
        "diversity_score", "enrollment_frequency", "paid_ratio", "ldi"
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df[FEAT_COLS].fillna(0))

    # ── KMeans K=4 ────────────────────────────────────────────────────────────
    km = KMeans(n_clusters=4, init="k-means++", n_init=20, random_state=42)
    km.fit(X_scaled)
    feat_df["Cluster"] = km.labels_

    # Compute silhouette
    sil_score = silhouette_score(X_scaled, km.labels_)
    per_cluster_sil_raw = silhouette_samples(X_scaled, km.labels_)
    per_cluster_sil = [
        float(per_cluster_sil_raw[km.labels_ == c].mean()) for c in range(4)
    ]

    # ── Remap clusters to match canonical names ────────────────────────────────
    # Match clusters by dominant feature signatures:
    #   C0=Budget (moderate enroll, moderate spend, some paid)
    #   C1=Power  (high enroll, high diversity)
    #   C2=Casual (low enroll, near-zero spend, ~0 paid_ratio)
    #   C3=Career (low enroll, high spend, 100% paid)
    cl_stats = feat_df.groupby("Cluster")[FEAT_COLS].mean()
    # Identify canonical mapping
    # Casual: lowest paid_ratio
    casual_raw = int(cl_stats["paid_ratio"].idxmin())
    # Power: highest diversity_score
    power_raw  = int(cl_stats["diversity_score"].idxmax())
    # Career: highest avg_spending
    career_raw = int(cl_stats["avg_spending"].idxmax())
    # Budget: the remaining one
    others = [c for c in range(4) if c not in [casual_raw, power_raw, career_raw]]
    budget_raw = others[0] if others else 0

    remap = {budget_raw: 0, power_raw: 1, casual_raw: 2, career_raw: 3}
    feat_df["Cluster"] = feat_df["Cluster"].map(remap)

    # ── Hierarchical clustering ────────────────────────────────────────────────
    hc = AgglomerativeClustering(n_clusters=4, linkage="ward")
    hc_labels = hc.fit_predict(X_scaled)
    hc_sil = silhouette_score(X_scaled, hc_labels)

    # ── Elbow data ────────────────────────────────────────────────────────────
    inertias, silhouettes = [], []
    for k in range(2, 11):
        km_k = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        km_k.fit(X_scaled)
        inertias.append(float(km_k.inertia_))
        silhouettes.append(float(silhouette_score(X_scaled, km_k.labels_)))

    # ── PCA 2D ────────────────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    pca_var = pca.explained_variance_ratio_.tolist()

    pca_df = pd.DataFrame({
        "x": X_pca[:, 0],
        "y": X_pca[:, 1],
        "c": feat_df["Cluster"].values,
        "uid": feat_df["UserID"].values,
    })

    # Centroids in PCA space
    centroids_pca = []
    for ci in range(4):
        mask = feat_df["Cluster"].values == ci
        centroids_pca.append([float(X_pca[mask, 0].mean()), float(X_pca[mask, 1].mean())])

    # ── Cluster summary (mean feature values) ─────────────────────────────────
    ALL_FEAT_COLS = FEAT_COLS + ["avg_courses_per_cat", "avg_course_duration"]
    cl_summary = []
    for ci in range(4):
        sub = feat_df[feat_df["Cluster"] == ci]
        row = {f: float(sub[f].mean()) for f in ALL_FEAT_COLS}
        row["total_spending"] = float(sub["total_spending"].mean())
        cl_summary.append(row)

    # ── Cluster stats ─────────────────────────────────────────────────────────
    cluster_stats = []
    for ci in range(4):
        sub = feat_df[feat_df["Cluster"] == ci]
        n   = len(sub)
        cluster_stats.append({
            "size":   n,
            "pct":    round(n / len(feat_df) * 100, 1),
            "enroll": round(float(sub["total_enrollments"].mean()), 1),
            "spend":  round(float(sub["avg_spending"].mean())),
            "cats":   round(float(sub["diversity_score"].mean()), 1),
            "paid":   f"{float(sub['paid_ratio'].mean()):.0%}",
        })

    # ── Intra-cluster cosine similarity ───────────────────────────────────────
    from sklearn.preprocessing import normalize
    X_norm = normalize(X_scaled, norm="l2")
    intra_sim = []
    for ci in range(4):
        mask = feat_df["Cluster"].values == ci
        Xc = X_norm[mask]
        if len(Xc) < 2:
            intra_sim.append(1.0)
        else:
            sims = (Xc @ Xc.T)
            # Mean of off-diagonal
            n_c = len(Xc)
            total = sims.sum() - n_c  # subtract diagonal (all 1s)
            intra_sim.append(float(total / (n_c * (n_c - 1))) if n_c > 1 else 1.0)

    # ── Monthly trends ────────────────────────────────────────────────────────
    trans_df["YM"] = trans_df["TransactionDate"].dt.to_period("M").astype(str)
    monthly_enroll = trans_df.groupby("YM").size().rename("Enrollments")
    monthly_rev    = trans_df.groupby("YM")["Amount"].sum().rename("Revenue")
    monthly_df     = pd.concat([monthly_enroll, monthly_rev], axis=1).reset_index()
    monthly_df     = monthly_df.sort_values("YM").reset_index(drop=True)

    # ── Age distribution ──────────────────────────────────────────────────────
    age_counts, age_bins = np.histogram(users_df["Age"], bins=15)

    # ── Gender ───────────────────────────────────────────────────────────────
    gender_counts = users_df["Gender"].value_counts().to_dict()

    # ── Courses EDA ───────────────────────────────────────────────────────────
    cat_counts  = courses_df["CourseCategory"].value_counts().to_dict()
    lvl_counts  = courses_df["CourseLevel"].value_counts().to_dict()
    type_counts = courses_df["CourseType"].value_counts().to_dict()
    ratings     = courses_df["CourseRating"].tolist()

    # ── Payment method breakdown ──────────────────────────────────────────────
    pay_counts = trans_df["PaymentMethod"].value_counts().to_dict()

    # ── Recommendation engine ─────────────────────────────────────────────────
    # For each cluster, rank courses by popularity_score = enrollments * avg_rating
    enroll_per_course = full_df.groupby("CourseID").size().rename("enrollments")
    avg_rating_course = full_df.groupby("CourseID")["CourseRating"].mean().rename("CourseRating")
    courses_ext = courses_df.copy()
    courses_ext = courses_ext.merge(enroll_per_course.reset_index(), on="CourseID", how="left")
    courses_ext["enrollments"] = courses_ext["enrollments"].fillna(0)
    courses_ext["popularity_score"] = courses_ext["enrollments"] * courses_ext["CourseRating"]

    # Cluster-course affinity: for each cluster, count how many users enrolled in each course
    cluster_user = feat_df[["UserID", "Cluster"]].copy()
    full_with_cl = full_df.merge(cluster_user, on="UserID", how="left")
    cluster_course_count = (
        full_with_cl.groupby(["Cluster", "CourseID"]).size()
        .reset_index(name="cluster_enroll")
    )

    cca_list = []
    for ci in range(4):
        cc = cluster_course_count[cluster_course_count["Cluster"] == ci]
        merged = courses_ext.merge(cc[["CourseID", "cluster_enroll"]], on="CourseID", how="left")
        merged["cluster_enroll"] = merged["cluster_enroll"].fillna(0)
        merged["pop_score"] = merged["cluster_enroll"] * merged["CourseRating"]
        merged["Cluster"] = ci
        merged["popularity_score"] = merged["pop_score"]
        cca_list.append(merged)
    cca_df = pd.concat(cca_list, ignore_index=True)

    # ── Learner-level data (for Learner Lookup) ───────────────────────────────
    learner_df = feat_df.copy()

    # ── Transaction history (enriched) ────────────────────────────────────────
    user_trans_df = full_df[["UserID", "CourseID", "CourseName", "CourseCategory",
                              "CourseLevel", "CourseType", "Amount", "TransactionDate"]].copy()
    user_trans_df["TransactionDate"] = user_trans_df["TransactionDate"].dt.strftime("%Y-%m-%d")

    return {
        "users_df":       users_df,
        "courses_df":     courses_df,
        "trans_df":       trans_df,
        "full_df":        full_df,
        "feat_df":        feat_df,
        "pca_df":         pca_df,
        "pca_var":        pca_var,
        "centroids_pca":  centroids_pca,
        "sil":            sil_score,
        "per_cluster_sil": per_cluster_sil,
        "hc_sil":         hc_sil,
        "intra_sim":      intra_sim,
        "inertias":       inertias,
        "silhouettes":    silhouettes,
        "km_inertia":     float(km.inertia_),
        "cluster_stats":  cluster_stats,
        "cl_summary":     cl_summary,
        "monthly_df":     monthly_df,
        "age_counts":     age_counts.tolist(),
        "age_bins":       age_bins.tolist(),
        "gender_counts":  gender_counts,
        "cat_counts":     cat_counts,
        "lvl_counts":     lvl_counts,
        "type_counts":    type_counts,
        "ratings":        ratings,
        "pay_counts":     pay_counts,
        "cca_df":         cca_df,
        "learner_df":     learner_df,
        "user_trans_df":  user_trans_df,
        "total_rev":      float(trans_df["Amount"].sum()),
        "n_users":        len(users_df),
        "n_courses":      len(courses_df),
        "n_trans":        len(trans_df),
    }


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
try:
    D = load_and_process()
    data_ok = True
except FileNotFoundError:
    data_ok = False
except Exception as exc:
    data_ok = False
    load_error = str(exc)

if not data_ok:
    st.error(
        "⚠️ **EduPro Online Platform.xlsx** not found in the working directory.\n\n"
        "Place the Excel file next to this script and re-run:\n"
        "```\n"
        "streamlit run edupro_dashboard.py\n"
        "```"
    )
    st.stop()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style='padding:.5rem 0 .8rem;'>
  <div style='font-size:1.15rem;font-weight:700;color:#7c73ff;'>🎓 EduPro Analytics</div>
  <div style='font-size:.72rem;color:#555575;margin-top:.15rem;'>
    Student Segmentation Platform
  </div>
</div>
<div style='border-top:1px solid #252540;margin:.2rem 0 .8rem;'></div>
""", unsafe_allow_html=True)

    page = st.radio("nav", [
        "🏠 Overview Dashboard",
        "🔍 Exploratory Analysis",
        "⚙️ Feature Engineering",
        "🔵 Cluster Explorer",
        "🎯 Recommendation Engine",
        "👤 Learner Lookup",
        "📐 Model Evaluation",
    ], label_visibility="collapsed")

    sil_str  = f"{D['sil']:.4f}"
    hcsil_str = f"{D['hc_sil']:.4f}"
    st.markdown(f"""
<div style='border-top:1px solid #252540;margin-top:.8rem;padding-top:.8rem;
     font-size:.73rem;color:#444460;line-height:2;'>
  Method: K-Means + Hierarchical<br>
  K = 4 &nbsp;|&nbsp; Features: 7<br>
  Silhouette: {sil_str}<br>
  HC Silhouette: {hcsil_str}<br>
  March 2026
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div style='border-top:1px solid #252540;margin-top:.8rem;padding-top:.8rem;'>
  <div style='font-size:.7rem;color:#444460;line-height:2;'>
    <span style='color:#555575;font-weight:600;'>👤 Author</span><br>
    <span style='color:#7c73ff;font-weight:700;font-size:.78rem;'>Dikshit</span>
  </div>
  <div style='font-size:.7rem;color:#444460;margin-top:.4rem;line-height:2;'>
    <span style='color:#555575;font-weight:600;'>🏢 Organization</span><br>
    <span style='color:#7c73ff;font-weight:700;font-size:.78rem;'>Unified Mentor</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview Dashboard":
    st.title("🎓 EduPro Student Segmentation Dashboard")
    st.caption("Personalized Learning Intelligence — Data-Driven Learner Profiles & Course Recommendations")

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("👥 Total Users",    f"{D['n_users']:,}")
    c2.metric("📚 Courses",        str(D["n_courses"]))
    c3.metric("🔄 Transactions",   f"{D['n_trans']:,}")
    c4.metric("💰 Revenue",        f"${D['total_rev']:,.0f}")
    c5.metric("🔵 Segments",       "4")
    c6.metric("📊 Silhouette",     f"{D['sil']:.4f}")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Learner Segment Distribution")
        sizes = [D["cluster_stats"][i]["size"] for i in range(4)]
        fig = go.Figure(go.Pie(
            labels=[f"C{i}: {CLUSTER_ICONS[i]} {CLUSTER_NAMES[i]}" for i in range(4)],
            values=sizes, hole=0.52,
            marker=dict(colors=CLUSTER_COLORS, line=dict(color="#0d0d1a", width=2)),
            textinfo="label+percent", textfont_size=10,
        ))
        dark_fig(fig, 310)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("#### Monthly Enrollments & Revenue")
        mdf = D["monthly_df"]
        fig = go.Figure()
        fig.add_bar(x=mdf["YM"], y=mdf["Enrollments"], name="Enrollments",
                    marker_color="#7c73ff", opacity=0.85, yaxis="y")
        fig.add_scatter(x=mdf["YM"], y=mdf["Revenue"], name="Revenue ($)",
                        mode="lines+markers", line=dict(color="#43e97b", width=2.5),
                        marker=dict(size=6), yaxis="y2")
        fig.update_layout(
            yaxis=dict(title="Enrollments", gridcolor="#252540", color="#8080a8"),
            yaxis2=dict(title="Revenue ($)", overlaying="y", side="right",
                        gridcolor="#252540", color="#8080a8"),
            legend=dict(orientation="h", y=1.08),
            xaxis=dict(tickangle=-45),
        )
        dark_fig(fig, 310)
        st.plotly_chart(fig, use_container_width=True)

    # Persona cards
    st.markdown("### Learner Personas")
    cols = st.columns(4)
    for i in range(4):
        stat = D["cluster_stats"][i]
        cs   = D["cl_summary"][i]
        with cols[i]:
            st.markdown(f"""
<div style='background:#16162a;border-left:4px solid {CLUSTER_COLORS[i]};
     border:1px solid #252540;border-radius:10px;padding:.9rem 1rem;'>
  <div style='font-size:.95rem;font-weight:700;color:#e0e0ff;'>
    {CLUSTER_ICONS[i]} {CLUSTER_NAMES[i]}
  </div>
  <div style='font-size:.78rem;color:{CLUSTER_COLORS[i]};margin:.18rem 0;'>
    {stat["size"]:,} learners · {stat["pct"]}%
  </div>
  <hr style='border-color:#252540;margin:.5rem 0;'>
  <div style='font-size:.78rem;color:#8080a8;line-height:1.9;'>
    📚 Avg enroll: <b style='color:#e0e0ff;'>{stat["enroll"]}</b><br>
    💵 Avg spend: <b style='color:#e0e0ff;'>${stat["spend"]}</b><br>
    🗂 Diversity: <b style='color:#e0e0ff;'>{stat["cats"]} cats</b><br>
    💳 Paid: <b style='color:#e0e0ff;'>{stat["paid"]}</b>
  </div>
  <div style='margin-top:.6rem;font-size:.78rem;color:#43e97b;font-weight:600;'>
    ▶ {CLUSTER_ACTIONS[i]}
  </div>
</div>""", unsafe_allow_html=True)

    # Payment method breakdown
    st.divider()
    st.markdown("### Payment Method Distribution")
    pay = D["pay_counts"]
    fig = go.Figure(go.Pie(
        labels=list(pay.keys()), values=list(pay.values()), hole=0.45,
        marker=dict(colors=["#6c63ff", "#43e97b", "#f7971e", "#ff6584"],
                    line=dict(color="#0d0d1a", width=2)),
        textinfo="label+percent", textfont_size=11,
    ))
    dark_fig(fig, 270)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploratory Analysis":
    st.title("🔍 Exploratory Data Analysis")
    st.caption("Deep dive into users, courses, and transaction patterns from all three dataset sheets")

    t_users, t_courses, t_txn, t_raw = st.tabs(
        ["👤 Users", "📚 Courses", "🔄 Transactions", "📄 Raw Data Preview"]
    )

    # ── Users tab ─────────────────────────────────────────────────────────────
    with t_users:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### User Age Distribution")
            counts = D["age_counts"]
            bins   = D["age_bins"]
            mids   = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
            fig = go.Figure(go.Bar(
                x=[f"{m:.0f}" for m in mids], y=counts,
                marker_color="#7c73ff", opacity=0.85))
            fig.update_layout(xaxis_title="Age", yaxis_title="Count", bargap=0.08)
            dark_fig(fig, 280)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Gender Distribution")
            gd = D["gender_counts"]
            fig = go.Figure(go.Pie(
                labels=list(gd.keys()), values=list(gd.values()),
                hole=0.50,
                marker=dict(colors=["#ff6584","#7c73ff"],
                            line=dict(color="#0d0d1a", width=2)),
                textinfo="label+percent", textfont_size=12,
            ))
            dark_fig(fig, 280)
            st.plotly_chart(fig, use_container_width=True)

        total_g = sum(gd.values())
        female  = gd.get("Female", 0)
        male    = gd.get("Male", 0)
        info_box(
            f"<b>{D['n_users']:,} users</b> analysed · Age range: "
            f"{int(D['users_df']['Age'].min())}–{int(D['users_df']['Age'].max())} · "
            f"Female: {female} ({female/total_g*100:.1f}%) "
            f"· Male: {male} ({male/total_g*100:.1f}%)"
        )

    # ── Courses tab ───────────────────────────────────────────────────────────
    with t_courses:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("#### Courses per Category")
            cats = D["cat_counts"]
            fig = go.Figure(go.Bar(
                x=list(cats.values()), y=list(cats.keys()),
                orientation="h", marker_color="#7c73ff", opacity=0.85))
            fig.update_layout(xaxis_title="Number of Courses",
                              yaxis=dict(autorange="reversed"))
            dark_fig(fig, 370, t=30)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Course Level Split")
            lvls = D["lvl_counts"]
            fig = go.Figure(go.Pie(
                labels=list(lvls.keys()), values=list(lvls.values()), hole=0.45,
                marker=dict(colors=["#f7971e","#6c63ff","#43e97b"],
                            line=dict(color="#0d0d1a", width=2)),
                textinfo="label+percent", textfont_size=11,
            ))
            dark_fig(fig, 200)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Free vs Paid")
            typs = D["type_counts"]
            fig = go.Figure(go.Bar(
                x=list(typs.keys()), y=list(typs.values()),
                marker_color=["#43e97b","#ff6584"]))
            fig.update_layout(yaxis_title="Count")
            dark_fig(fig, 180, t=20)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Course Rating Distribution")
        fig = go.Figure(go.Histogram(
            x=D["ratings"], nbinsx=15,
            marker_color="#f7971e", opacity=0.85))
        fig.update_layout(xaxis_title="Rating", yaxis_title="Count", bargap=0.05)
        dark_fig(fig, 220, t=30)
        st.plotly_chart(fig, use_container_width=True)

        info_box(
            f"<b>{D['n_courses']} courses</b> across "
            f"{len(D['cat_counts'])} categories · "
            f"Price range: Free – ${D['courses_df']['CoursePrice'].max():,.2f} · "
            f"Avg duration: {D['courses_df']['CourseDuration'].mean():.1f} hrs · "
            f"Avg rating: {D['courses_df']['CourseRating'].mean():.2f}"
        )

    # ── Transactions tab ───────────────────────────────────────────────────────
    with t_txn:
        st.markdown("#### Monthly Transaction Trends")
        mdf = D["monthly_df"]
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Scatter(
                x=mdf["YM"], y=mdf["Enrollments"],
                mode="lines+markers", line=dict(color="#7c73ff", width=2.5),
                marker=dict(size=7), fill="tozeroy", fillcolor="rgba(124,115,255,0.13)",
                name="Enrollments"))
            fig.update_layout(xaxis_title="Month", yaxis_title="Enrollments",
                              xaxis=dict(tickangle=-45))
            dark_fig(fig, 270)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = go.Figure(go.Scatter(
                x=mdf["YM"], y=mdf["Revenue"],
                mode="lines+markers", line=dict(color="#43e97b", width=2.5),
                marker=dict(size=7, symbol="square"),
                fill="tozeroy", fillcolor="rgba(67,233,123,0.13)", name="Revenue"))
            fig.update_layout(xaxis_title="Month", yaxis_title="Revenue ($)",
                              xaxis=dict(tickangle=-45))
            dark_fig(fig, 270)
            st.plotly_chart(fig, use_container_width=True)

        avg_enroll = mdf["Enrollments"].mean()
        info_box(
            f"Total Revenue: <b>${D['total_rev']:,.2f}</b> over "
            f"{len(mdf)} months · "
            f"Average Monthly Enrollments: <b>{avg_enroll:.0f}</b> · "
            f"Peak Month: <b>{mdf.loc[mdf['Revenue'].idxmax(), 'YM']}</b>"
        )

        # Payment methods
        st.markdown("#### Payment Method Breakdown")
        pay = D["pay_counts"]
        fig = go.Figure(go.Bar(
            x=list(pay.keys()), y=list(pay.values()),
            marker_color=["#6c63ff","#43e97b","#f7971e","#ff6584"]))
        fig.update_layout(xaxis_title="Payment Method", yaxis_title="Transactions")
        dark_fig(fig, 250, t=30)
        st.plotly_chart(fig, use_container_width=True)

    # ── Raw Data Preview tab ───────────────────────────────────────────────────
    with t_raw:
        st.markdown("#### Users Sheet — First 20 rows")
        st.dataframe(D["users_df"].head(20), use_container_width=True)

        st.markdown("#### Courses Sheet — All rows")
        st.dataframe(D["courses_df"], use_container_width=True)

        st.markdown("#### Transactions Sheet — First 50 rows")
        show_trans = D["trans_df"].copy()
        show_trans["TransactionDate"] = show_trans["TransactionDate"].dt.strftime("%Y-%m-%d")
        st.dataframe(show_trans.head(50), use_container_width=True)

        info_box(
            f"<b>Users</b>: {D['users_df'].shape[0]:,} rows × {D['users_df'].shape[1]} cols &nbsp;|&nbsp; "
            f"<b>Courses</b>: {D['courses_df'].shape[0]:,} rows × {D['courses_df'].shape[1]} cols &nbsp;|&nbsp; "
            f"<b>Transactions</b>: {D['trans_df'].shape[0]:,} rows × {D['trans_df'].shape[1]} cols"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Feature Engineering":
    st.title("⚙️ Feature Engineering")
    st.caption("9 learner-level features engineered from raw transactions for clustering & profiling")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Feature Definitions")
        feat_defs = [
            ("total_enrollments",    "Engagement",  "Total courses enrolled"),
            ("avg_spending",         "Behavioral",  "Mean transaction amount ($)"),
            ("avg_course_rating",    "Preference",  "Mean rating of enrolled courses"),
            ("diversity_score",      "Behavioral",  "Unique categories explored"),
            ("enrollment_frequency", "Engagement",  "Enrollments per active day"),
            ("paid_ratio",           "Behavioral",  "Fraction of paid enrollments"),
            ("ldi",                  "Preference",  "Learning Depth Index (Adv+Inter ratio)"),
            ("avg_courses_per_cat",  "Behavioral",  "Avg courses per category"),
            ("avg_course_duration",  "Engagement",  "Avg course duration (hours)"),
        ]
        type_color = {"Engagement":"#f7971e","Behavioral":"#ff6584","Preference":"#6c63ff"}
        rows_html = "".join(
            f"<tr>"
            f"<td style='color:#e0e0ff;padding:.42rem .75rem;border-top:1px solid #252540;"
            f"font-size:.82rem;'>{f}</td>"
            f"<td style='color:{type_color[t]};padding:.42rem .75rem;border-top:1px solid #252540;"
            f"font-size:.82rem;font-weight:600;'>{t}</td>"
            f"<td style='color:#8080a8;padding:.42rem .75rem;border-top:1px solid #252540;"
            f"font-size:.8rem;'>{d}</td></tr>"
            for f, t, d in feat_defs
        )
        st.markdown(f"""
<div style='background:#16162a;border:1px solid #252540;border-radius:11px;overflow:hidden;'>
<table style='width:100%;border-collapse:collapse;'>
  <thead><tr>
    <th style='background:#ffffff08;color:#7c73ff;padding:.5rem .75rem;text-align:left;
        font-size:.82rem;font-weight:700;'>Feature</th>
    <th style='background:#ffffff08;color:#7c73ff;padding:.5rem .75rem;text-align:left;
        font-size:.82rem;font-weight:700;'>Type</th>
    <th style='background:#ffffff08;color:#7c73ff;padding:.5rem .75rem;text-align:left;
        font-size:.82rem;font-weight:700;'>Description</th>
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table></div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("#### Cluster Mean Feature Values")
        ALL_FEATS = [
            "total_enrollments","avg_spending","avg_course_rating",
            "diversity_score","enrollment_frequency","paid_ratio","ldi",
            "avg_courses_per_cat","avg_course_duration"
        ]
        hdr_html = "".join(
            f"<th style='background:#ffffff08;color:{CLUSTER_COLORS[i]};padding:.48rem .65rem;"
            f"text-align:right;font-size:.82rem;font-weight:700;'>{CLUSTER_ICONS[i]} C{i}</th>"
            for i in range(4)
        )
        body_html = ""
        for key in ALL_FEATS:
            vals = [D["cl_summary"][ci].get(key, 0) for ci in range(4)]
            cells = "".join(
                f"<td style='color:#e0e0ff;padding:.42rem .65rem;border-top:1px solid #252540;"
                f"text-align:right;font-size:.82rem;'>{v:.3f}</td>"
                for v in vals
            )
            body_html += (
                f"<tr><td style='color:#8080a8;padding:.42rem .75rem;"
                f"border-top:1px solid #252540;font-size:.82rem;'>{key}</td>{cells}</tr>"
            )
        st.markdown(f"""
<div style='background:#16162a;border:1px solid #252540;border-radius:11px;overflow:hidden;'>
<table style='width:100%;border-collapse:collapse;'>
  <thead><tr>
    <th style='background:#ffffff08;color:#7c73ff;padding:.48rem .75rem;text-align:left;
        font-size:.82rem;font-weight:700;'>Feature</th>
    {hdr_html}
  </tr></thead>
  <tbody>{body_html}</tbody>
</table></div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Feature Comparison by Cluster")
    feat_opts = {
        "Total Enrollments":    "total_enrollments",
        "Avg Spending ($)":     "avg_spending",
        "Avg Course Rating":    "avg_course_rating",
        "Diversity Score":      "diversity_score",
        "Enrollment Frequency": "enrollment_frequency",
        "Paid Ratio":           "paid_ratio",
        "Learning Depth Index": "ldi",
        "Avg Courses per Cat":  "avg_courses_per_cat",
        "Avg Course Duration":  "avg_course_duration",
    }
    sel_label = st.selectbox("Select Feature to Compare", list(feat_opts.keys()))
    sel_feat  = feat_opts[sel_label]
    vals = [D["cl_summary"][ci].get(sel_feat, 0) for ci in range(4)]
    fig = go.Figure(go.Bar(
        x=[f"C{i}: {CLUSTER_NAMES[i]}" for i in range(4)], y=vals,
        marker_color=CLUSTER_COLORS,
        text=[f"{v:.3f}" for v in vals], textposition="outside",
    ))
    fig.update_layout(yaxis_title=sel_label, yaxis_range=[0, max(vals) * 1.3 + 0.001])
    dark_fig(fig, 300)
    st.plotly_chart(fig, use_container_width=True)

    # Feature distribution heatmap
    st.divider()
    st.markdown("#### Cluster Feature Heatmap (Normalised)")
    norm_vals = []
    for feat in ALL_FEATS:
        col_vals = [D["cl_summary"][ci].get(feat, 0) for ci in range(4)]
        col_max = max(col_vals) or 1
        norm_vals.append([v / col_max for v in col_vals])
    z_matrix = [list(row) for row in zip(*norm_vals)]  # shape: 4 x 9
    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=ALL_FEATS,
        y=[f"C{i}: {CLUSTER_NAMES[i]}" for i in range(4)],
        colorscale="Viridis",
        text=[[f"{v:.2f}" for v in row] for row in z_matrix],
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        showscale=True,
    ))
    fig.update_layout(xaxis=dict(tickangle=-30))
    dark_fig(fig, 280, t=30)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CLUSTER EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Cluster Explorer":
    st.title("🔵 Learner Cluster Explorer")
    st.caption("Visualise and compare the four learner behavioural segments")

    t_pca, t_cmp, t_radar, t_detail = st.tabs([
        "🗺️ PCA Map", "📊 Comparison", "🧭 Radar Profiles", "📋 Segment Detail"
    ])

    # ── PCA Map ───────────────────────────────────────────────────────────────
    with t_pca:
        st.markdown("### Learner Behavioral Space — PCA 2D Projection")
        pca_df = D["pca_df"]
        fig = go.Figure()
        for ci in range(4):
            sub = pca_df[pca_df["c"] == ci]
            fig.add_scatter(
                x=sub["x"], y=sub["y"], mode="markers",
                name=f"C{ci}: {CLUSTER_ICONS[ci]} {CLUSTER_NAMES[ci]}",
                marker=dict(color=CLUSTER_COLORS[ci], size=4, opacity=0.55,
                            line=dict(width=0)),
                text=sub["uid"],
                hovertemplate="<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
            )
        for ci, cp in enumerate(D["centroids_pca"]):
            fig.add_scatter(
                x=[cp[0]], y=[cp[1]], mode="markers+text",
                marker=dict(color=CLUSTER_COLORS[ci], size=22, symbol="star",
                            line=dict(color="#0d0d1a", width=1.5)),
                text=[f"C{ci}"], textposition="top center",
                textfont=dict(color="#e0e0ff", size=12, family="Arial Black"),
                showlegend=False,
            )
        fig.update_layout(
            xaxis_title="PC1", yaxis_title="PC2",
            legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11)),
            plot_bgcolor="#0d0d1a", paper_bgcolor="#0d0d1a",
        )
        dark_fig(fig, 520, t=50, b=30, l=40, r=20)
        fig.update_xaxes(showgrid=True, gridcolor="#1a1a30", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#1a1a30", zeroline=False)
        st.plotly_chart(fig, use_container_width=True)

        pv = D["pca_var"]
        tot_var = (pv[0] + pv[1]) * 100
        st.markdown(f"""
<div style='background:#1a1a2e;border:1px solid #252540;border-radius:8px;
     padding:.7rem 1.2rem;font-size:.84rem;color:#8080a8;margin-top:-.5rem;'>
  PC1: <b style='color:#e0e0ff;'>{pv[0]*100:.1f}%</b> variance &nbsp;|&nbsp;
  PC2: <b style='color:#e0e0ff;'>{pv[1]*100:.1f}%</b> variance &nbsp;|&nbsp;
  Total: <b style='color:#e0e0ff;'>{tot_var:.1f}%</b> &nbsp;|&nbsp;
  Silhouette: <b style='color:#7c73ff;'>{D['sil']:.4f}</b>
</div>""", unsafe_allow_html=True)

    # ── Feature Comparison ────────────────────────────────────────────────────
    with t_cmp:
        st.markdown("#### Feature Comparison Across Clusters")
        feat_cmp_opts = {
            "Total Enrollments":    "total_enrollments",
            "Avg Spending ($)":     "avg_spending",
            "Avg Course Rating":    "avg_course_rating",
            "Diversity Score":      "diversity_score",
            "Paid Ratio":           "paid_ratio",
            "Learning Depth Index": "ldi",
            "Enrollment Frequency": "enrollment_frequency",
        }
        sel = st.selectbox("Feature", list(feat_cmp_opts.keys()), key="cmp_feat")
        c1, c2 = st.columns(2)
        with c1:
            vals = [D["cl_summary"][ci].get(feat_cmp_opts[sel], 0) for ci in range(4)]
            fig = go.Figure(go.Bar(
                x=[f"C{i}: {CLUSTER_NAMES[i]}" for i in range(4)], y=vals,
                marker_color=CLUSTER_COLORS,
                text=[f"{v:.3f}" for v in vals], textposition="outside",
            ))
            fig.update_layout(yaxis_title=sel, yaxis_range=[0, max(vals) * 1.3 + 0.001])
            dark_fig(fig, 320)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Cluster Summary Table")
            rows = [{
                "Cluster":    f"{CLUSTER_ICONS[i]} {CLUSTER_NAMES[i]}",
                "Size":       f"{D['cluster_stats'][i]['size']:,}",
                "Share":      f"{D['cluster_stats'][i]['pct']}%",
                "Avg Enroll": round(D["cl_summary"][i]["total_enrollments"], 2),
                "Avg Spend":  f"${D['cl_summary'][i]['avg_spending']:.0f}",
                "Diversity":  round(D["cl_summary"][i]["diversity_score"], 2),
                "Paid Ratio": f"{D['cl_summary'][i]['paid_ratio']:.0%}",
            } for i in range(4)]
            st.dataframe(pd.DataFrame(rows).set_index("Cluster"),
                         use_container_width=True, height=260)

    # ── Radar ─────────────────────────────────────────────────────────────────
    with t_radar:
        st.markdown("#### Radar Profiles — Normalised Cluster Characteristics")
        rf = ["total_enrollments","avg_spending","diversity_score",
              "paid_ratio","ldi","enrollment_frequency","avg_course_rating"]
        rl = ["Enrollments","Spending","Diversity","Paid Ratio","LDI","Enroll Freq","Rating"]
        raw_vals = [[D["cl_summary"][ci].get(f, 0) for f in rf] for ci in range(4)]
        col_max  = [max(raw_vals[ci][j] for ci in range(4)) or 1 for j in range(len(rf))]
        fig = go.Figure()
        for ci in range(4):
            norm = [raw_vals[ci][j] / col_max[j] for j in range(len(rf))]
            norm_closed = norm + [norm[0]]
            lbls_closed = rl + [rl[0]]
            fig.add_scatterpolar(
                r=norm_closed, theta=lbls_closed, fill="toself",
                name=f"C{ci}: {CLUSTER_ICONS[ci]} {CLUSTER_NAMES[ci]}",
                line=dict(color=CLUSTER_COLORS[ci], width=2),
                fillcolor=CLUSTER_COLORS[ci], opacity=0.25,
            )
        fig.update_layout(
            polar=dict(
                bgcolor="#16162a",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#252540", color="#666680"),
                angularaxis=dict(gridcolor="#252540", color="#e0e0ff"),
            ),
            legend=dict(orientation="h", y=-0.1),
        )
        dark_fig(fig, 480, t=30, b=60)
        fig.update_layout(paper_bgcolor="#16162a")
        st.plotly_chart(fig, use_container_width=True)

    # ── Segment Detail Cards ──────────────────────────────────────────────────
    with t_detail:
        c_left, c_right = st.columns(2)
        for i in range(4):
            cs   = D["cl_summary"][i]
            stat = D["cluster_stats"][i]
            col  = c_left if i % 2 == 0 else c_right
            with col:
                st.markdown(f"""
<div style='background:#16162a;border:1px solid #252540;
     border-left:4px solid {CLUSTER_COLORS[i]};
     border-radius:11px;padding:1.1rem;margin-bottom:1rem;'>
  <div style='font-size:1rem;font-weight:700;color:#e0e0ff;margin-bottom:.7rem;'>
    {CLUSTER_ICONS[i]} C{i}: {CLUSTER_NAMES[i]}
  </div>
  <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:.4rem;'>
    <div style='background:#0d0d22;border-radius:8px;padding:.55rem;text-align:center;'>
      <div style='font-size:1rem;font-weight:700;color:{CLUSTER_COLORS[i]};'>{stat["size"]:,}</div>
      <div style='font-size:.7rem;color:#8080a8;'>Learners ({stat["pct"]}%)</div>
    </div>
    <div style='background:#0d0d22;border-radius:8px;padding:.55rem;text-align:center;'>
      <div style='font-size:1rem;font-weight:700;color:{CLUSTER_COLORS[i]};'>{cs["total_enrollments"]:.1f}</div>
      <div style='font-size:.7rem;color:#8080a8;'>Avg Enrollments</div>
    </div>
    <div style='background:#0d0d22;border-radius:8px;padding:.55rem;text-align:center;'>
      <div style='font-size:1rem;font-weight:700;color:{CLUSTER_COLORS[i]};'>${cs["avg_spending"]:.0f}</div>
      <div style='font-size:.7rem;color:#8080a8;'>Avg Spending</div>
    </div>
    <div style='background:#0d0d22;border-radius:8px;padding:.55rem;text-align:center;'>
      <div style='font-size:1rem;font-weight:700;color:{CLUSTER_COLORS[i]};'>{cs["diversity_score"]:.2f}</div>
      <div style='font-size:.7rem;color:#8080a8;'>Diversity Score</div>
    </div>
    <div style='background:#0d0d22;border-radius:8px;padding:.55rem;text-align:center;'>
      <div style='font-size:1rem;font-weight:700;color:{CLUSTER_COLORS[i]};'>{cs["paid_ratio"]:.0%}</div>
      <div style='font-size:.7rem;color:#8080a8;'>Paid Ratio</div>
    </div>
    <div style='background:#0d0d22;border-radius:8px;padding:.55rem;text-align:center;'>
      <div style='font-size:1rem;font-weight:700;color:{CLUSTER_COLORS[i]};'>{cs["ldi"]:.2f}</div>
      <div style='font-size:.7rem;color:#8080a8;'>Depth Index</div>
    </div>
  </div>
  <div style='margin-top:.8rem;font-size:.78rem;color:#43e97b;font-weight:600;'>
    ▶ Strategy: {CLUSTER_ACTIONS[i]}
  </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Recommendation Engine":
    st.title("🎯 Personalized Course Recommendation Engine")
    st.caption("Cluster-based content recommendations ranked by cluster popularity score")

    cca_df = D["cca_df"]

    # Filters
    col_cl, col_lv, col_cat, col_n = st.columns([2, 2, 3, 2])
    with col_cl:
        sel_cl = st.selectbox(
            "Learner Cluster",
            options=list(range(4)),
            format_func=lambda i: f"{CLUSTER_ICONS[i]} {CLUSTER_NAMES[i]}",
            key="rec_cl",
        )
    with col_lv:
        all_levels = ["All Levels"] + sorted(D["courses_df"]["CourseLevel"].dropna().unique().tolist())
        sel_lv = st.selectbox("Level", all_levels, key="rec_lv")
    with col_cat:
        all_cats = ["All Categories"] + sorted(D["courses_df"]["CourseCategory"].dropna().unique().tolist())
        sel_cat = st.selectbox("Category", all_cats, key="rec_cat")
    with col_n:
        n_recs = st.slider("Max Results", 3, 15, 5)

    # Filter
    recs_df = cca_df[cca_df["Cluster"] == sel_cl].copy()
    if sel_lv  != "All Levels":     recs_df = recs_df[recs_df["CourseLevel"] == sel_lv]
    if sel_cat != "All Categories": recs_df = recs_df[recs_df["CourseCategory"] == sel_cat]
    recs_df = recs_df.sort_values("popularity_score", ascending=False).head(n_recs)

    st.divider()
    c_list, c_chart = st.columns(2)

    with c_list:
        st.markdown(
            f"### {CLUSTER_ICONS[sel_cl]} Top Picks for "
            f"<span style='color:{CLUSTER_COLORS[sel_cl]};'>{CLUSTER_NAMES[sel_cl]}</span>",
            unsafe_allow_html=True)
        if not recs_df.empty:
            for rank, (_, r) in enumerate(recs_df.iterrows(), 1):
                price_str  = f"${r['CoursePrice']:.0f}" if r.get("CoursePrice", 0) > 0 else "🆓 Free"
                ctype_badge = (
                    "<span style='background:#43e97b22;color:#43e97b;padding:.1rem .4rem;"
                    "border-radius:6px;font-size:.75rem;'>Free</span>"
                    if r.get("CourseType") == "Free" else
                    "<span style='background:#ff658422;color:#ff6584;padding:.1rem .4rem;"
                    "border-radius:6px;font-size:.75rem;'>Paid</span>"
                )
                st.markdown(f"""
<div style='background:#16162a;border:1px solid #252540;border-radius:9px;
     padding:.75rem 1rem;margin-bottom:.45rem;
     border-left:3px solid {CLUSTER_COLORS[sel_cl]};'>
  <div style='display:flex;justify-content:space-between;align-items:center;'>
    <span style='font-weight:700;font-size:.9rem;color:#e0e0ff;'>
      #{rank} {r["CourseName"]}
    </span>
    <span style='color:#43e97b;font-weight:700;font-size:.9rem;'>
      ★ {r["popularity_score"]:.1f}
    </span>
  </div>
  <div style='font-size:.78rem;color:#8080a8;margin-top:.3rem;'>
    📂 {r["CourseCategory"]} &nbsp;·&nbsp;
    📊 {r.get("CourseLevel","?")} &nbsp;·&nbsp;
    {ctype_badge} &nbsp;·&nbsp;
    ⭐ {r["CourseRating"]:.2f} &nbsp;·&nbsp;
    {price_str} &nbsp;·&nbsp;
    👥 {int(r.get("enrollments", 0))} enrolled
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.info("No courses match the selected filters. Try relaxing the criteria.")

    with c_chart:
        if not recs_df.empty:
            st.markdown("#### Popularity Score Ranking")
            fig = go.Figure(go.Bar(
                x=recs_df["popularity_score"].tolist(),
                y=recs_df["CourseName"].tolist(),
                orientation="h",
                marker=dict(color=CLUSTER_COLORS[sel_cl], opacity=0.85),
                text=[f"{v:.1f}" for v in recs_df["popularity_score"]],
                textposition="outside",
            ))
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                xaxis_title="Popularity Score",
            )
            dark_fig(fig, max(280, 60 + 38 * len(recs_df)), t=30, b=30)
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 🔀 Top-5 Courses Across All Clusters")
    top_cols = st.columns(4)
    for ci in range(4):
        top_recs = (
            cca_df[cca_df["Cluster"] == ci]
            .sort_values("popularity_score", ascending=False)
            .head(5)
        )
        with top_cols[ci]:
            st.markdown(
                f"<div style='color:{CLUSTER_COLORS[ci]};font-weight:700;"
                f"font-size:.88rem;margin-bottom:.4rem;'>"
                f"{CLUSTER_ICONS[ci]} {CLUSTER_NAMES[ci]}</div>",
                unsafe_allow_html=True)
            for rank, (_, r) in enumerate(top_recs.iterrows(), 1):
                price_str = f"${r['CoursePrice']:.0f}" if r.get("CoursePrice", 0) > 0 else "Free"
                st.markdown(f"""
<div style='background:#16162a;border:1px solid #252540;border-radius:8px;
     padding:.5rem .7rem;margin-bottom:.3rem;'>
  <div style='font-size:.8rem;font-weight:600;color:#e0e0ff;'>
    {rank}. {r["CourseName"]}
  </div>
  <div style='font-size:.73rem;color:#8080a8;'>
    {r["CourseCategory"]} · {price_str}
  </div>
  <div style='font-size:.75rem;'>
    <span style='color:#43e97b;font-weight:700;'>★ {r["popularity_score"]:.1f}</span>
    &nbsp;·&nbsp;
    <span style='color:#8080a8;'>⭐ {r["CourseRating"]:.2f}</span>
  </div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — LEARNER LOOKUP
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Learner Lookup":
    st.title("👤 Individual Learner Profile Lookup")
    st.caption(f"Select any of {D['n_users']:,} learners to view their segment, behavior metrics, and personalized recommendations")

    learner_df   = D["learner_df"]
    user_trans   = D["user_trans_df"]
    cca_df       = D["cca_df"]
    learner_map  = learner_df.set_index("UserID").to_dict("index")

    c_lbl, c_inp, c_btn = st.columns([1, 6, 2])
    with c_lbl:
        st.markdown(
            "<div style='padding-top:1.75rem;color:#8080a8;font-size:.85rem;"
            "font-weight:600;'>User ID</div>",
            unsafe_allow_html=True)
    with c_inp:
        uid_raw = st.text_input("uid", value="U00059",
                                label_visibility="collapsed",
                                placeholder="e.g. U00059")
    with c_btn:
        st.write("")
        st.button("⚡ Load Profile", use_container_width=True)

    # Normalize
    uid = uid_raw.strip().upper()
    if uid and not uid.startswith("U"):
        uid = "U" + uid.zfill(5)
    elif uid.startswith("U") and len(uid) < 6:
        uid = "U" + uid[1:].zfill(5)

    learner = learner_map.get(uid)

    if learner:
        ci    = int(learner["Cluster"])
        color = CLUSTER_COLORS[ci]

        st.divider()
        c_profile, c_recs = st.columns([1, 1], gap="medium")

        with c_profile:
            st.markdown(f"""
<div style='background:#16162a;border:1px solid #252540;border-radius:12px;padding:1.2rem;'>
  <div style='font-size:1rem;font-weight:700;color:#e0e0ff;margin-bottom:.65rem;'>
    Profile — {uid}
  </div>
  <div style='margin-bottom:.65rem;'>
    <span style='background:{color};color:#fff;padding:.22rem .75rem;
         border-radius:20px;font-size:.8rem;font-weight:700;'>
      {CLUSTER_ICONS[ci]} C{ci}: {CLUSTER_NAMES[ci]}
    </span>
  </div>
  <div style='background:#0d0d22;border-radius:8px;padding:.6rem .8rem;
       margin-bottom:.85rem;color:#8080a8;font-size:.82rem;font-style:italic;'>
    {CLUSTER_STRATEGIES[ci]}
  </div>
  <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:.45rem;'>
    <div style='background:#1a1a35;border-radius:8px;padding:.65rem .4rem;text-align:center;'>
      <div style='font-size:1.15rem;font-weight:700;color:#7c73ff;'>{int(learner["total_enrollments"])}</div>
      <div style='font-size:.67rem;color:#8080a8;margin-top:.1rem;'>📚 Enrolled</div>
    </div>
    <div style='background:#1a1a35;border-radius:8px;padding:.65rem .4rem;text-align:center;'>
      <div style='font-size:1.15rem;font-weight:700;color:#7c73ff;'>${int(learner.get("total_spending", 0))}</div>
      <div style='font-size:.67rem;color:#8080a8;margin-top:.1rem;'>💰 Total Spend</div>
    </div>
    <div style='background:#1a1a35;border-radius:8px;padding:.65rem .4rem;text-align:center;'>
      <div style='font-size:1.15rem;font-weight:700;color:#7c73ff;'>${int(learner["avg_spending"])}</div>
      <div style='font-size:.67rem;color:#8080a8;margin-top:.1rem;'>💵 Avg Spend</div>
    </div>
    <div style='background:#1a1a35;border-radius:8px;padding:.65rem .4rem;text-align:center;'>
      <div style='font-size:1.15rem;font-weight:700;color:#7c73ff;'>{int(learner["diversity_score"])}</div>
      <div style='font-size:.67rem;color:#8080a8;margin-top:.1rem;'>🌐 Categories</div>
    </div>
    <div style='background:#1a1a35;border-radius:8px;padding:.65rem .4rem;text-align:center;'>
      <div style='font-size:1.15rem;font-weight:700;color:#7c73ff;'>{learner["avg_course_rating"]:.2f}</div>
      <div style='font-size:.67rem;color:#8080a8;margin-top:.1rem;'>⭐ Avg Rating</div>
    </div>
    <div style='background:#1a1a35;border-radius:8px;padding:.65rem .4rem;text-align:center;'>
      <div style='font-size:1.15rem;font-weight:700;color:#7c73ff;'>{learner["ldi"]:.2f}</div>
      <div style='font-size:.67rem;color:#8080a8;margin-top:.1rem;'>📊 Depth Index</div>
    </div>
    <div style='background:#1a1a35;border-radius:8px;padding:.65rem .4rem;text-align:center;'>
      <div style='font-size:.95rem;font-weight:700;color:#7c73ff;'>{learner.get("preferred_level", "—")}</div>
      <div style='font-size:.67rem;color:#8080a8;margin-top:.1rem;'>🎓 Pref Level</div>
    </div>
    <div style='background:#1a1a35;border-radius:8px;padding:.65rem .4rem;text-align:center;'>
      <div style='font-size:.75rem;font-weight:700;color:#7c73ff;word-break:break-word;'>{learner.get("preferred_category", "—")}</div>
      <div style='font-size:.67rem;color:#8080a8;margin-top:.1rem;'>📂 Pref Cat</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        with c_recs:
            top5 = (
                cca_df[cca_df["Cluster"] == ci]
                .sort_values("popularity_score", ascending=False)
                .head(5)
            )
            recs_html = ""
            for _, r in top5.iterrows():
                type_icon = "🆓" if r.get("CourseType") == "Free" else "💳"
                recs_html += f"""
<div style='background:#0d0d22;border-radius:9px;padding:.7rem .9rem;margin-bottom:.4rem;'>
  <div style='font-weight:700;font-size:.9rem;color:#e0e0ff;margin-bottom:.28rem;'>
    {r["CourseName"]}
  </div>
  <div style='font-size:.77rem;color:#8080a8;'>
    📁 {r["CourseCategory"]} &nbsp;|&nbsp;
    📊 {r.get("CourseLevel","?")} &nbsp;|&nbsp;
    {type_icon} &nbsp;|&nbsp;
    ⭐ {r["CourseRating"]:.1f} &nbsp;|&nbsp;
    <span style='color:#43e97b;font-weight:700;'>Score: {r["popularity_score"]:.1f}</span>
  </div>
</div>"""
            st.markdown(f"""
<div style='background:#16162a;border:1px solid #252540;border-radius:12px;padding:1.2rem;'>
  <div style='font-size:1rem;font-weight:700;color:#e0e0ff;margin-bottom:.75rem;'>
    🎯 Personalized Top-5 Recommendations
  </div>
  {recs_html}
</div>""", unsafe_allow_html=True)

        # Enrollment history
        st.divider()
        hist = user_trans[user_trans["UserID"] == uid]
        if not hist.empty:
            st.markdown("#### 📋 Enrollment History")
            header = (
                "<tr>"
                "<th style='color:#7c73ff;padding:.5rem .75rem;text-align:left;"
                "background:#ffffff08;font-size:.82rem;'>Course</th>"
                "<th style='color:#7c73ff;padding:.5rem .75rem;text-align:left;"
                "background:#ffffff08;font-size:.82rem;'>Category</th>"
                "<th style='color:#7c73ff;padding:.5rem .75rem;text-align:left;"
                "background:#ffffff08;font-size:.82rem;'>Level</th>"
                "<th style='color:#7c73ff;padding:.5rem .75rem;text-align:left;"
                "background:#ffffff08;font-size:.82rem;'>Type</th>"
                "<th style='color:#7c73ff;padding:.5rem .75rem;text-align:right;"
                "background:#ffffff08;font-size:.82rem;'>Paid ($)</th>"
                "<th style='color:#7c73ff;padding:.5rem .75rem;text-align:left;"
                "background:#ffffff08;font-size:.82rem;'>Date</th>"
                "</tr>"
            )
            body = ""
            for _, t in hist.iterrows():
                amt = t["Amount"]
                amt_str = f"{int(amt)}" if amt == int(amt) else f"{amt:.2f}"
                body += (
                    f"<tr>"
                    f"<td style='color:#e0e0ff;padding:.42rem .75rem;"
                    f"border-top:1px solid #252540;font-size:.82rem;'>{t['CourseName']}</td>"
                    f"<td style='color:#8080a8;padding:.42rem .75rem;"
                    f"border-top:1px solid #252540;font-size:.82rem;'>{t['CourseCategory']}</td>"
                    f"<td style='color:#8080a8;padding:.42rem .75rem;"
                    f"border-top:1px solid #252540;font-size:.82rem;'>{t['CourseLevel']}</td>"
                    f"<td style='color:#8080a8;padding:.42rem .75rem;"
                    f"border-top:1px solid #252540;font-size:.82rem;'>{t['CourseType']}</td>"
                    f"<td style='color:#8080a8;padding:.42rem .75rem;"
                    f"border-top:1px solid #252540;font-size:.82rem;text-align:right;'>{amt_str}</td>"
                    f"<td style='color:#8080a8;padding:.42rem .75rem;"
                    f"border-top:1px solid #252540;font-size:.82rem;'>{t['TransactionDate']}</td>"
                    f"</tr>"
                )
            st.markdown(f"""
<div style='background:#16162a;border:1px solid #252540;border-radius:11px;overflow:hidden;'>
<table style='width:100%;border-collapse:collapse;'>
  <thead>{header}</thead>
  <tbody>{body}</tbody>
</table></div>""", unsafe_allow_html=True)
        else:
            st.info(f"No transaction records found for {uid}.")

    elif uid.strip():
        st.warning(f"⚠️ Learner **{uid}** not found. Try IDs like U00001, U00059, U01000.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Model Evaluation":
    st.title("📐 Model Evaluation & Validation")
    st.caption("Cluster quality metrics · K-selection analysis · Hierarchical validation")

    t_sil, t_k, t_metrics = st.tabs([
        "📊 Silhouette Analysis", "📈 K Selection", "📋 Metrics Summary"
    ])

    # ── Silhouette ────────────────────────────────────────────────────────────
    with t_sil:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Avg Silhouette Score per Cluster")
            sil_vals = D["per_cluster_sil"]
            fig = go.Figure(go.Bar(
                x=[f"C{i}: {CLUSTER_NAMES[i]}" for i in range(4)],
                y=sil_vals, marker_color=CLUSTER_COLORS,
                text=[f"{v:.4f}" for v in sil_vals], textposition="outside",
            ))
            fig.update_layout(
                yaxis_title="Avg Silhouette Score",
                yaxis_range=[0, max(sil_vals) * 1.3])
            dark_fig(fig, 320)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Per-Cluster Silhouette Breakdown")
            for ci, sil in enumerate(sil_vals):
                st.markdown(
                    f"<div style='margin-bottom:.5rem;'>"
                    f"<span style='color:{CLUSTER_COLORS[ci]};font-weight:700;font-size:.88rem;'>"
                    f"{CLUSTER_ICONS[ci]} C{ci}: {CLUSTER_NAMES[ci]}</span>"
                    f"<span style='color:#7c73ff;font-weight:700;float:right;'>{sil:.4f}</span>"
                    f"</div>",
                    unsafe_allow_html=True)
                st.progress(min(sil / max(sil_vals), 1.0))
                st.write("")

            info_box(
                f"Overall K-Means Silhouette: <b style='color:#7c73ff;'>{D['sil']:.4f}</b> · "
                f"HC Silhouette: <b style='color:#7c73ff;'>{D['hc_sil']:.4f}</b><br>"
                f"Score 0.30–0.50 = reasonable behavioural structure."
            )

        st.markdown("#### Intra-Cluster Cosine Similarity")
        sim_vals = D["intra_sim"]
        fig2 = go.Figure(go.Bar(
            x=[f"C{i}: {CLUSTER_NAMES[i]}" for i in range(4)],
            y=sim_vals, marker_color=CLUSTER_COLORS,
            text=[f"{v:.4f}" for v in sim_vals], textposition="outside",
        ))
        fig2.update_layout(
            yaxis_title="Avg Intra-Cluster Cosine Similarity",
            yaxis_range=[0, max(sim_vals) * 1.2])
        dark_fig(fig2, 260)
        st.plotly_chart(fig2, use_container_width=True)

    # ── K Selection ───────────────────────────────────────────────────────────
    with t_k:
        ks = list(range(2, 2 + len(D["inertias"])))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Elbow Method — Inertia vs K")
            fig = go.Figure()
            fig.add_scatter(
                x=ks, y=D["inertias"], mode="lines+markers",
                line=dict(color="#7c73ff", width=2.5), marker_size=8, name="Inertia")
            idx4 = ks.index(4) if 4 in ks else 2
            fig.add_scatter(
                x=[4], y=[D["inertias"][idx4]], mode="markers",
                marker=dict(color="#ff6584", size=16, symbol="star"),
                name="K=4 (selected)")
            fig.update_layout(
                xaxis_title="Number of Clusters (K)",
                yaxis_title="Inertia (WCSS)",
                xaxis=dict(tickvals=ks))
            dark_fig(fig, 310)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Silhouette Score vs K")
            fig2 = go.Figure()
            fig2.add_scatter(
                x=ks, y=D["silhouettes"], mode="lines+markers",
                line=dict(color="#43e97b", width=2.5), marker_size=8, name="Silhouette")
            fig2.add_scatter(
                x=[4], y=[D["silhouettes"][idx4]], mode="markers",
                marker=dict(color="#f7971e", size=16, symbol="diamond"),
                name="K=4 (selected)")
            fig2.update_layout(
                xaxis_title="Number of Clusters (K)",
                yaxis_title="Silhouette Score",
                xaxis=dict(tickvals=ks))
            dark_fig(fig2, 310)
            st.plotly_chart(fig2, use_container_width=True)

        info_box(
            f"✅ <b>K=4 selected</b> based on Elbow inflection point & domain knowledge. "
            f"Final model (n_init=20) achieved Silhouette = <b>{D['sil']:.4f}</b>, "
            f"Final Inertia = <b>{D['km_inertia']:,.2f}</b>."
        )

    # ── Metrics Summary ───────────────────────────────────────────────────────
    with t_metrics:
        st.markdown("#### Complete Model Evaluation Summary")
        pca_var     = D["pca_var"]
        tot_pca_var = sum(pca_var) * 100
        cl_sizes    = [D["cluster_stats"][i]["size"] for i in range(4)]
        largest_ci  = int(np.argmax(cl_sizes))
        smallest_ci = int(np.argmin(cl_sizes))
        best_sil_ci = int(np.argmax(D["per_cluster_sil"]))
        best_cos_ci = int(np.argmax(D["intra_sim"]))

        metrics = [
            ("Algorithm",                          "K-Means (k-means++, n_init=20, random_state=42)"),
            ("Hierarchical Method",                "Agglomerative Clustering (Ward linkage)"),
            ("Optimal Clusters (K)",               "4"),
            ("Clustering Features Used",           "7 (total_enrollments, avg_spending, avg_course_rating, diversity_score, enrollment_frequency, paid_ratio, ldi)"),
            ("Total Learners Segmented",           f"{D['n_users']:,}"),
            ("K-Means Silhouette Score",           f"{D['sil']:.4f}"),
            ("K-Means Final Inertia",              f"{D['km_inertia']:,.2f}"),
            ("Hierarchical Clustering Silhouette", f"{D['hc_sil']:.4f}"),
            ("PCA PC1 Variance Explained",         f"{pca_var[0]*100:.2f}%"),
            ("PCA PC2 Variance Explained",         f"{pca_var[1]*100:.2f}%"),
            ("PCA Total Variance (2 components)",  f"{tot_pca_var:.1f}%"),
            ("Largest Cluster",
             f"C{largest_ci}: {CLUSTER_ICONS[largest_ci]} {CLUSTER_NAMES[largest_ci]} "
             f"({cl_sizes[largest_ci]:,} learners, {D['cluster_stats'][largest_ci]['pct']}%)"),
            ("Smallest Cluster",
             f"C{smallest_ci}: {CLUSTER_ICONS[smallest_ci]} {CLUSTER_NAMES[smallest_ci]} "
             f"({cl_sizes[smallest_ci]:,} learners, {D['cluster_stats'][smallest_ci]['pct']}%)"),
            ("Best Per-Cluster Silhouette",
             f"C{best_sil_ci}: {CLUSTER_ICONS[best_sil_ci]} {CLUSTER_NAMES[best_sil_ci]} "
             f"({D['per_cluster_sil'][best_sil_ci]:.4f})"),
            ("Best Intra-Cluster Cosine Sim",
             f"C{best_cos_ci}: {CLUSTER_ICONS[best_cos_ci]} {CLUSTER_NAMES[best_cos_ci]} "
             f"({D['intra_sim'][best_cos_ci]:.4f})"),
        ]

        st.markdown(
            "<div style='background:#16162a;border:1px solid #252540;border-radius:11px;"
            "padding:1rem 1.2rem;'>", unsafe_allow_html=True)
        for label, val in metrics:
            cl, cv = st.columns([2, 3])
            cl.markdown(f"<span style='color:#8080a8;font-size:.83rem;'>{label}</span>",
                        unsafe_allow_html=True)
            cv.markdown(f"<span style='color:#7c73ff;font-weight:700;font-size:.83rem;'>{val}</span>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        info_box(
            "📌 A Silhouette Score of 0.30–0.50 is considered <b>reasonable structure</b> for "
            "behavioural clustering data. Agreement between K-Means and Hierarchical clustering "
            "confirms cluster stability. The <b>Power Learner</b> cluster typically achieves "
            "exceptional internal cohesion."
        )
