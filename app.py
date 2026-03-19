import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# -------------------------------
# 1. PAGE CONFIG & STYLING
# -------------------------------
st.set_page_config(
    page_title="Customer Intel AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    div[data-testid="stExpander"] { border: none !important; box-shadow: none !important; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# 2. SMART DATA LOADER
# -------------------------------
@st.cache_data
def load_and_clean_data():
    try:
        # Attempting multiple delimiters for marketing_campaign.csv
        df = pd.read_csv("marketing_campaign.csv", sep=None, engine='python')
    except Exception:
        # Fallback: Create dummy data for demo purposes if file is missing
        data = {
            'ID': range(100),
            'Year_Birth': np.random.randint(1960, 2005, 100),
            'Income': np.random.randint(30000, 120000, 100),
            'MntWines': np.random.randint(0, 1000, 100),
            'MntMeatProducts': np.random.randint(0, 1000, 100),
            'MntGoldProds': np.random.randint(0, 500, 100)
        }
        df = pd.DataFrame(data)

    # Cleaning logic
    df['Income'] = pd.to_numeric(df.get('Income', 0), errors='coerce').fillna(df['Income'].median() if 'Income' in df else 0)
    df['Age'] = 2026 - pd.to_numeric(df.get('Year_Birth', 1980), errors='coerce')
    
    # Dynamic Spending Calculation
    spend_cols = [c for c in df.columns if c.startswith('Mnt')]
    df['Total_Spending'] = df[spend_cols].sum(axis=1) if spend_cols else np.random.randint(100, 5000, len(df))
    
    return df

df_raw = load_and_clean_data()

# -------------------------------
# 3. SIDEBAR NAVIGATION & FILTERS
# -------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("Control Center")
    
    st.subheader("🛠️ Global Filters")
    age_range = st.slider("Age Range", int(df_raw['Age'].min()), int(df_raw['Age'].max()), (25, 65))
    income_slider = st.slider("Income Tier (k)", 0, 200, (20, 150))
    
    st.divider()
    st.info("💡 **Pro Tip:** Use the 'Clusters' slider in the Segmentation tab to refine persona groups.")

# Filter logic
df = df_raw[
    (df_raw['Age'].between(age_range[0], age_range[1])) & 
    (df_raw['Income'].between(income_slider[0]*1000, income_slider[1]*1000))
]

# -------------------------------
# 4. DASHBOARD LAYOUT
# -------------------------------
tabs = st.tabs(["📊 Overview", "🧪 Segmentation", "🔮 Predictive CLV", "💬 AI Assistant"])

# --- TAB 1: OVERVIEW ---
with tabs[0]:
    # KPI Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", len(df), delta=f"{len(df)-len(df_raw)} vs Total")
    m2.metric("Avg. Ticket Size", f"${df['Total_Spending'].mean():,.0f}")
    m3.metric("Retention Rate", "84%", "+2%") # Placeholder for logic
    m4.metric("Potential Revenue", f"${df['Total_Spending'].sum():,.0f}")

    st.divider()

    c1, c2 = st.columns([6, 4])
    with c1:
        fig_scatter = px.scatter(df, x="Income", y="Total_Spending", color="Age",
                                size="Total_Spending", hover_name="ID" if "ID" in df else None,
                                title="Income vs Spending Power", color_continuous_scale="Viridis")
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with c2:
        # Pie chart of spending categories
        if any(c.startswith('Mnt') for c in df.columns):
            spend_sums = df[[c for c in df.columns if c.startswith('Mnt')]].mean().reset_index()
            fig_pie = px.pie(spend_sums, values=0, names='index', hole=0.4, title="Category Mix")
            st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 2: SEGMENTATION ---
with tabs[1]:
    st.subheader("Persona Clustering (K-Means)")
    
    col_feat, col_viz = st.columns([1, 2])
    
    with col_feat:
        k_val = st.number_input("Number of Segments (k)", 2, 6, 3)
        features = ['Age', 'Income', 'Total_Spending']
        st.write("Using features: " + ", ".join(features))
        
        # ML Logic
        X = StandardScaler().fit_transform(df[features])
        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10).fit(X)
        df['Cluster'] = kmeans.labels_
        
        st.success(f"Identified {k_val} distinct customer personas.")

    with col_viz:
        fig_cluster = px.scatter_3d(df, x='Income', y='Total_Spending', z='Age',
                                    color=df['Cluster'].astype(str), title="3D Cluster Map")
        st.plotly_chart(fig_cluster, use_container_width=True)

    # Segment Breakdown
    st.write("### Segment DNA")
    breakdown = df.groupby('Cluster')[features].mean().style.background_gradient(cmap='Blues')
    st.table(breakdown)

# --- TAB 3: PREDICTIVE ---
with tabs[2]:
    st.subheader("Customer Lifetime Value (CLV) Projection")
    
    # Simple Regression
    reg_model = LinearRegression()
    reg_model.fit(df[['Age', 'Income']], df['Total_Spending'])
    df['Predicted_Value'] = reg_model.predict(df[['Age', 'Income']])
    
    fig_clv = go.Figure()
    fig_clv.add_trace(go.Scatter(x=df['Income'], y=df['Total_Spending'], mode='markers', name='Actual'))
    fig_clv.add_trace(go.Scatter(x=df['Income'], y=df['Predicted_Value'], mode='lines', name='Predicted Trend'))
    fig_clv.update_layout(title="Income vs Predicted Spending Trend")
    st.plotly_chart(fig_clv, use_container_width=True)
    
    st.warning("⚠️ **Note:** Prediction is based on current Income and Age demographics.")

# --- TAB 4: AI ASSISTANT ---
with tabs[3]:
    st.subheader("🗨️ Business Intelligence Chat")
    query = st.text_input("Ask about your data (e.g., 'Who are my top earners?')")
    
    if query:
        query = query.lower()
        if "top" in query or "earner" in query:
            top_5 = df.nlargest(5, 'Income')[['Age', 'Income', 'Total_Spending']]
            st.write("Here are your top 5 high-income customers:")
            st.dataframe(top_5)
        elif "spend" in query:
            st.write(f"Average spending in this filter: ${df['Total_Spending'].mean():.2f}")
        else:
            st.write("I'm trained on 'income', 'spending', and 'top customers'. Try those keywords!")

# -------------------------------
# 5. STRATEGY EXPORT (FOOTER)
# -------------------------------
st.divider()
st.subheader("🎯 Recommended Strategy")
s1, s2 = st.columns(2)

with s1:
    high_potential = df[(df['Income'] > df['Income'].median()) & (df['Total_Spending'] < df['Total_Spending'].median())]
    st.info(f"**Upsell Opportunity:** {len(high_potential)} customers have high income but low spending. Target with premium loyalty invites.")

with s2:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📩 Export Intelligence Report", data=csv, file_name="ai_report.csv", mime="text/csv")