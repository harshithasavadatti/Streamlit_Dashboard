import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils.load_data import load_data

# Load data
df = load_data()

# Page configuration
st.set_page_config(page_title="🏦 Home Credit Risk Dashboard", layout="wide", page_icon="🏦")

# Calculate additional columns
df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).round(1)

# ===================== HEADER =====================
st.title("🏦 Home Credit Default Risk Dashboard")
st.markdown("""
This dashboard provides an overview of the Home Credit dataset, including data quality assessment, 
applicant demographics, and portfolio risk metrics.
""")

# ===================== KPI CALCULATIONS =====================
total_applicants = df['SK_ID_CURR'].nunique()
default_rate = df['TARGET'].mean() * 100
repaid_rate = (1 - df['TARGET'].mean()) * 100
total_features = df.shape[1]
avg_missing = (df.isnull().mean().mean()) * 100
num_features = df.select_dtypes(include='number').shape[1]
cat_features = df.select_dtypes(exclude='number').shape[1]
median_age = df['AGE_YEARS'].median()
median_income = df['AMT_INCOME_TOTAL'].median()
avg_credit = df['AMT_CREDIT'].mean()

# ===================== KPI DISPLAY =====================
st.subheader("📊 Portfolio Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Applicants", f"{total_applicants:,.0f}")
col2.metric("Default Rate", f"{default_rate:.2f}%", delta=f"{-repaid_rate:.2f}%", delta_color="inverse")
col3.metric("Repaid Rate", f"{repaid_rate:.2f}%", delta=f"{repaid_rate-default_rate:.2f}%")
col4.metric("Median Age", f"{median_age:.1f} years")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Median Income", f"${median_income:,.0f}")
col6.metric("Avg Credit Amount", f"${avg_credit:,.0f}")
col7.metric("Data Quality", f"{100-avg_missing:.1f}%", delta=f"{-avg_missing:.1f}%", delta_color="inverse")
col8.metric("Features", f"{total_features}", help=f"Numerical: {num_features}, Categorical: {cat_features}")
st.markdown("---")

# ===================== DATA QUALITY SECTION =====================
st.subheader("📈 Data Quality Assessment")

# Calculate missing values by column
missing_data = df.isnull().sum().reset_index()
missing_data.columns = ['Feature', 'Missing_Count']
missing_data['Missing_Percentage'] = (missing_data['Missing_Count'] / len(df) * 100).round(2)
missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)

col1, col2 = st.columns([3, 2])

with col1:
    st.write("**Top 20 Features by Missing Percentage**")
    if len(missing_data) > 0:
        # Create bar chart for missing values (using only the top 20)
        missing_chart = alt.Chart(missing_data.head(20)).mark_bar(color='#FF6B6B').encode(
            x=alt.X('Missing_Percentage:Q', title='Missing Values (%)'),
            y=alt.Y('Feature:N', sort='-x', title='Feature'),
            tooltip=['Feature', 'Missing_Percentage']
        ).properties(height=400)
        st.altair_chart(missing_chart, use_container_width=True)
    else:
        st.success("No missing values found in the dataset!")
# :N → Nominal → categories with no intrinsic order (like names, genders, feature labels).
# :Q → Quantitative → numerical, continuous values (like age, income, percentages).
# :O → Ordinal → ordered categories (like low/medium/high, ranks).
# :T → Temporal → time/date values.
with col2:
    st.write("**Data Completeness Summary**")
    completeness_data = {
        'Metric': ['Complete Features', 'Features with Missing Data', 'Average Missing %'],
        'Value': [f"{total_features - len(missing_data)}", f"{len(missing_data)}", f"{avg_missing:.2f}%"]
    }
    completeness_df = pd.DataFrame(completeness_data)
    st.table(completeness_df)
    
    st.write("**Data Types Summary**")
    dtype_data = {
        'Type': ['Numerical', 'Categorical'],
        'Count': [num_features, cat_features]
    }
    dtype_df = pd.DataFrame(dtype_data)
    st.table(dtype_df)

st.markdown("---")

# ===================== DISTRIBUTIONS SECTION =====================
st.subheader("📊 Data Distributions")

# Create a sample of the data for visualization to reduce data transfer
sample_size = min(5000, len(df))
df_sample = df.sample(sample_size, random_state=42) if len(df) > 5000 else df

# Create tabs for different types of distributions
tab1, tab2, tab3 = st.tabs(["Target Variable", "Demographics", "Financials"])

with tab1:
    st.write("**Loan Repayment Status**")
    
    # Target distribution - using value counts instead of full data
    target_counts = df['TARGET'].value_counts()
    target_data = pd.DataFrame({
        'TARGET': ['Repaid', 'Default'],
        'count': [target_counts.get(0, 0), target_counts.get(1, 0)]
    })
    target_data['Percentage'] = (target_data['count'] / target_data['count'].sum() * 100).round(1)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(target_data.set_index('TARGET')[['count', 'Percentage']])
    
    with col2:
        # Create donut chart
        donut_chart = alt.Chart(target_data).mark_arc(innerRadius=60).encode(
            theta=alt.Theta(field="count", type="quantitative"),
            color=alt.Color(field="TARGET", type="nominal", 
                          scale=alt.Scale(domain=['Repaid', 'Default'], 
                                         range=['#2E8B57', '#DC143C']),
                          legend=alt.Legend(title="Loan Status")),
            tooltip=['TARGET', 'count', 'Percentage']
        ).properties(width=300, height=300)
        st.altair_chart(donut_chart, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution - using sampled data
        st.write("**Age Distribution**")
        age_chart = alt.Chart(df_sample).mark_bar(color="#4682B4", opacity=0.7).encode(
            alt.X("AGE_YEARS:Q", bin=alt.Bin(maxbins=30), title="Age (Years)"),
            alt.Y("count()", title="Number of Applicants"),
            tooltip=[alt.Tooltip('AGE_YEARS:Q', bin=True), 'count()']
        ).properties(height=300)
        st.altair_chart(age_chart, use_container_width=True)
        
        # Gender distribution - using value counts
        if 'CODE_GENDER' in df.columns:
            st.write("**Gender Distribution**")
            gender_counts = df['CODE_GENDER'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            gender_chart = alt.Chart(gender_counts).mark_bar().encode(
                x=alt.X('Gender', axis=alt.Axis(labelAngle=360)),
                y='Count',
                color='Gender',
                tooltip=['Gender', 'Count']
            ).properties(height=300)
            st.altair_chart(gender_chart, use_container_width=True)
    
    with col2:
        # Family status distribution - using value counts
        if 'NAME_FAMILY_STATUS' in df.columns:
            st.write("**Family Status Distribution**")
            family_counts = df['NAME_FAMILY_STATUS'].value_counts().reset_index()
            family_counts.columns = ['Family_Status', 'Count']
            family_chart = alt.Chart(family_counts).mark_bar().encode(
                x=alt.X('Family_Status', sort='-y'),
                y='Count',
                color=alt.Color('Family_Status', legend=None),
                tooltip=['Family_Status', 'Count']
            ).properties(height=300)
            st.altair_chart(family_chart, use_container_width=True)
        
        # Education distribution - using value counts
        if 'NAME_EDUCATION_TYPE' in df.columns:
            st.write("**Education Levels**")
            education_counts = df['NAME_EDUCATION_TYPE'].value_counts().reset_index()
            education_counts.columns = ['Education', 'Count']
            education_chart = alt.Chart(education_counts).mark_bar().encode(
                x=alt.X('Education', sort='-y'),
                y='Count',
                color=alt.Color('Education', legend=None),
                tooltip=['Education', 'Count']
            ).properties(height=300)
            st.altair_chart(education_chart, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Income distribution - using sampled data
        st.write("**Annual Income Distribution**")
        income_q99 = df['AMT_INCOME_TOTAL'].quantile(0.99)
        income_sample = df_sample[df_sample['AMT_INCOME_TOTAL'] <= income_q99]
        
        income_chart = alt.Chart(income_sample).mark_bar(color="#228B22", opacity=0.7).encode(
            alt.X("AMT_INCOME_TOTAL:Q", bin=alt.Bin(maxbins=50), title="Annual Income"),
            alt.Y("count()", title="Number of Applicants"),
            tooltip=[alt.Tooltip('AMT_INCOME_TOTAL:Q', bin=True), 'count()']
        ).properties(height=300)
        st.altair_chart(income_chart, use_container_width=True)
        
        # Income boxplot - using sampled data
        st.write("**Income Distribution (Box Plot)**")
        income_box = alt.Chart(income_sample).mark_boxplot().encode(
            y='AMT_INCOME_TOTAL:Q'
        ).properties(height=200)
        st.altair_chart(income_box, use_container_width=True)
    
    with col2:
        # Credit amount distribution - using sampled data
        st.write("**Credit Amount Distribution**")
        credit_q99 = df['AMT_CREDIT'].quantile(0.99)
        credit_sample = df_sample[df_sample['AMT_CREDIT'] <= credit_q99]
        
        credit_chart = alt.Chart(credit_sample).mark_bar(color="#FF6347", opacity=0.7).encode(
            alt.X("AMT_CREDIT:Q", bin=alt.Bin(maxbins=50), title="Credit Amount"),
            alt.Y("count()", title="Number of Applicants"),
            tooltip=[alt.Tooltip('AMT_CREDIT:Q', bin=True), 'count()']
        ).properties(height=300)
        st.altair_chart(credit_chart, use_container_width=True)
        
        # Credit amount boxplot - using sampled data
        st.write("**Credit Amount Distribution (Box Plot)**")
        credit_box = alt.Chart(credit_sample).mark_boxplot(color="#B847FF").encode(
            y='AMT_CREDIT:Q'
        ).properties(height=200)
        st.altair_chart(credit_box, use_container_width=True)

st.markdown("---")

# ===================== INSIGHTS SECTION =====================
st.subheader("🔍 Key Insights")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown("""
    **Data Quality Observations:**
    - Average missing value percentage is **{:.2f}%** across all features
    - **{} features** contain missing data requiring attention
    - The dataset contains **{} numerical** and **{} categorical** features
    """.format(avg_missing, len(missing_data), num_features, cat_features))

with insight_col2:
    st.markdown("""
    **Portfolio Risk Insights:**
    - Default rate of **{:.2f}%** indicates significant credit risk
    - Median applicant age is **{:.1f} years**
    - Income distribution is typically right-skewed with a long tail of high earners
    """.format(default_rate, median_age))

gender_counts = df['CODE_GENDER'].value_counts().to_dict() if 'CODE_GENDER' in df.columns else "varied"
st.markdown("""
**Distribution Characteristics:**
- Age distribution follows a roughly normal pattern, centered around {:.1f} years
- Income and credit amount distributions are highly right-skewed with extreme outliers
- Gender distribution shows {} applicants
- Family status and education level provide important segmentation variables for risk assessment
""".format(median_age, gender_counts))

# ===================== FOOTER =====================
st.markdown("---")
