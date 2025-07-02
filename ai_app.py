import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import time
st.set_page_config(page_title="AI Adoption EDA", layout="wide")
st.title("🤖 AI Tool Adoption EDA Dashboard")

# Create a placeholder
message_placeholder = st.empty()


# Show loading spinner while loading data
with st.spinner("Loading dataset..."):
    csv_url = "https://raw.githubusercontent.com/ANALYZE-IRONTHUNDER/ANALYZE-IRONTHUNDER/refs/heads/main/AI_ADOPTATION/ai_adoption_dataset.csv"
    df = pd.read_csv(csv_url)
    
# Show success message temporarily
message_placeholder.success("Data loaded successfully!")

# Optional: Wait and then clear message
time.sleep(2)
message_placeholder.empty()

# Optional filters
with st.sidebar:
    st.header("🔍 Filters")
    year_filter = st.multiselect("Year", sorted(df['year'].unique()), default=sorted(df['year'].unique()))
    country_filter = st.multiselect("Country", sorted(df['country'].unique()), default=sorted(df['country'].unique()))
    age_filter = st.multiselect("Age Group", sorted(df['age_group'].unique()), default=sorted(df['age_group'].unique()))

# Apply filters
df = df[df['year'].isin(year_filter) & df['country'].isin(country_filter) & df['age_group'].isin(age_filter)]

tab1, tab2, tab3, tab4 = st.tabs(["📌 Overview", "📈 Numeric Analysis", "📊 Categoricals", "📉 Trends & Comparisons"])

with tab1:
    st.subheader("🧾 Dataset Snapshot")
    st.dataframe(df.head())
    st.write("Shape:", df.shape)
    

with tab2:
    st.subheader("📈 Distribution of Numeric Features")
    num_cols = ['adoption_rate', 'daily_active_users']
    for col in num_cols:
        fig = px.histogram(df, x=col, nbins=30, marginal="box", color_discrete_sequence=["steelblue"])
        fig.update_layout(title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔗 Correlation Heatmap")
    fig2, ax = plt.subplots()
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig2)

with tab3:
    st.subheader("📊 AI Tool Usage")
    fig3 = px.pie(df, names="ai_tool", title="AI Tool Distribution")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("📊 Industry Spread")
    industry_counts = df['industry'].value_counts().reset_index()
    industry_counts.columns = ['industry', 'count']
    fig4 = px.bar(industry_counts, x='industry', y='count',
                labels={"industry": "Industry", "count": "Count"},
                title="Industry Representation", color='industry')
    st.plotly_chart(fig4, use_container_width=True)


with tab4:
    st.subheader("📉 Adoption Rate by Industry")
    fig5 = px.box(df, x="industry", y="adoption_rate", color="industry")
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("📈 Average Adoption Over Time by Tool")
    df_grouped = df.groupby(["year", "ai_tool"])["adoption_rate"].mean().reset_index()
    fig6 = px.line(df_grouped, x="year", y="adoption_rate", color="ai_tool", markers=True)
    st.plotly_chart(fig6, use_container_width=True)

    st.subheader("📊 Daily Active Users by Company Size")
    fig7 = px.box(df, x="company_size", y="daily_active_users", color="company_size")
    st.plotly_chart(fig7, use_container_width=True)

