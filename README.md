# 🎓 EduPro – Student Segmentation & Personalized Recommendation System

A data-driven analytics project that segments learners and generates personalized course recommendations using machine learning, built as an interactive **Streamlit dashboard**.

---

## 📌 Problem Statement

Online learning platforms often struggle with:
- Low user engagement  
- Poor conversion from free to paid users  
- Lack of personalized learning paths  

This project solves that by:
> Segmenting learners based on behavior and delivering targeted recommendations to improve engagement and revenue.

---

## 🚀 Key Outcomes

- 👥 Segmented **3,000 learners into 4 meaningful personas**
- 📊 Identified **behavioral patterns driving revenue**
- 🎯 Built a **recommendation engine** based on cluster preferences
- 📈 Enabled **data-driven marketing & product strategies**

---

## 🧠 Learner Segments & Business Insights

| Cluster | Persona | Insight | Business Action |
|---|---|---|---|
| 💰 C0 | Budget Learner | Price-sensitive, mixed usage | Offer discounts & bundles |
| ⚡ C1 | Power Learner | Highly engaged, explores widely | Introduce subscriptions |
| 🌐 C2 | Casual Explorer | Low engagement, free users | Push conversion campaigns |
| 💼 C3 | Career Focuser | High spend, focused learning | Upsell premium certifications |

---

## 📊 Dashboard Features

### 1. Overview Dashboard
- KPI metrics (Revenue, Users, Enrollments)
- Segment distribution
- Monthly trends

### 2. Exploratory Data Analysis
- User demographics
- Course categories
- Transaction patterns

### 3. Feature Engineering
- Behavioral feature creation
- Cluster comparison visuals

### 4. Cluster Explorer
- PCA visualization
- Segment profiling
- Radar comparisons

### 5. Recommendation Engine
- Cluster-based course recommendations
- Popularity scoring system

### 6. Learner Lookup
- Individual user analysis
- Personalized recommendations

### 7. Model Evaluation
- Silhouette score
- Elbow method
- Cluster validation

---

## ⚙️ Machine Learning Approach

- **Clustering Algorithms:**
  - K-Means (Primary)
  - Agglomerative Clustering (Validation)

- **Optimal Clusters:** 4  
- **Silhouette Score:** ~0.38–0.39  

- **Features Used:**
  - Engagement, spending, diversity, preferences

- **Dimensionality Reduction:**
  - PCA (for visualization)

---

## 🧪 Feature Engineering

Key behavioral features:

- Total enrollments  
- Average spending  
- Course rating preference  
- Category diversity  
- Paid ratio  
- Learning depth index (LDI)  

👉 These features capture **user intent, engagement, and monetization behavior**

---

## 🎯 Recommendation Logic

Courses are ranked using:

> **Popularity Score = Cluster Enrollments × Average Rating**

This ensures:
- Relevance (cluster-specific)
- Quality (high-rated courses)
- Engagement optimization

---

## 🏗️ Tech Stack

- **Frontend:** Streamlit  
- **Visualization:** Plotly  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn, SciPy  
- **Data Source:** Excel (.xlsx)

---

## 👨‍💻 Author

**Dikshit**  
Data Analyst | Machine Learning Enthusiast  

🏢 Organization: Unified Mentor 

🔗 LinkedIn: [Dikshit](https://www.linkedin.com/in/dikshit-05a9a9244/)
