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
        st.markdown(f"""
**Interpretation:**

The histogram above visualizes the distribution of `{col}` across the filtered dataset. This allows you to quickly assess the central tendency, spread, and shape of the data for this numeric feature. Look for patterns such as skewness (whether the data leans left or right), the presence of outliers, and whether the distribution is normal, bimodal, or otherwise irregular. Understanding these characteristics can help inform further statistical analysis and guide data preprocessing or modeling decisions.
""")

    st.subheader("🔗 Correlation Heatmap")
    fig2, ax = plt.subplots()
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig2)
    st.markdown("""
**Interpretation:**

The heatmap above displays the pairwise correlation coefficients between the selected numeric features. Correlation values range from -1 (perfect negative) to +1 (perfect positive), with 0 indicating no linear relationship. Strong positive or negative correlations suggest that two features move together or in opposite directions, respectively. Identifying these relationships can help detect multicollinearity, inform feature selection, and reveal underlying patterns in the data that may be relevant for predictive modeling or business insights.
""")

with tab3:

    st.subheader("📊 AI Tool Usage")
    fig3 = px.pie(df, names="ai_tool", title="AI Tool Distribution")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""
**Interpretation:**

The pie chart above illustrates the proportion of each AI tool represented in the dataset. This visualization helps you quickly identify which tools are most widely adopted and which are less common among respondents. A larger slice indicates greater popularity or usage. Understanding the distribution of tool usage can inform market trends, highlight dominant technologies, and guide strategic decisions for stakeholders interested in AI adoption.
""")


    st.subheader("📊 Industry Spread")
    industry_counts = df['industry'].value_counts().reset_index()
    industry_counts.columns = ['industry', 'count']
    fig4 = px.bar(industry_counts, x='industry', y='count',
                labels={"industry": "Industry", "count": "Count"},
                title="Industry Representation", color='industry')
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown("""
**Interpretation:**

The bar chart above shows the number of records for each industry present in the dataset. This provides insight into the representation and diversity of industries surveyed or analyzed. Industries with higher counts are more prevalent in the data, which may influence the generalizability of findings. Recognizing the industry spread is important for contextualizing results and understanding which sectors are leading or lagging in AI adoption.
""")


with tab4:

    st.subheader("📉 Adoption Rate by Industry")
    fig5 = px.box(df, x="industry", y="adoption_rate", color="industry")
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("""
**Interpretation:**

The box plot above compares the distribution of AI adoption rates across different industries. Each box summarizes the median, quartiles, and potential outliers for each industry, allowing you to assess variability and central tendency. Wide boxes or long whiskers indicate greater variability, while outliers may point to exceptional cases. This visualization helps identify which industries are leading or lagging in AI adoption and whether adoption is consistent or highly variable within each sector.
""")


    st.subheader("📈 Average Adoption Over Time by Tool")
    df_grouped = df.groupby(["year", "ai_tool"])["adoption_rate"].mean().reset_index()
    fig6 = px.line(df_grouped, x="year", y="adoption_rate", color="ai_tool", markers=True)
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("""
**Interpretation:**

The line chart above tracks the average adoption rate of each AI tool over time, providing a temporal perspective on how usage patterns have evolved. Each line represents a different tool, and the trajectory shows whether its adoption is increasing, decreasing, or remaining stable. This helps reveal trends, seasonality, and shifts in popularity, which can be valuable for forecasting, strategic planning, and understanding the dynamics of AI adoption in the market.
""")


    st.subheader("📊 Daily Active Users by Company Size")
    fig7 = px.box(df, x="company_size", y="daily_active_users", color="company_size")
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown("""
**Interpretation:**

The box plot above shows the distribution of daily active users segmented by company size. This allows for comparison of user engagement across organizations of different scales. The plot highlights the median, spread, and outliers for each company size group, helping to identify whether larger or smaller companies tend to have more active users, and whether there is significant variability within each group. Such insights can inform product targeting, resource allocation, and further segmentation analysis.
""")

