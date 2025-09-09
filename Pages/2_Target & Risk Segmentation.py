import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils.load_data import load_data

# Load data
df = load_data()

# Page configuration
st.set_page_config(page_title="🏦 Risk Segmentation", layout="wide", page_icon="📊")

# Calculate additional columns
df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).round(1)
df['EMPLOYMENT_YEARS'] = (-df['DAYS_EMPLOYED'] / 365).round(1)
# Fix invalid employment values (those positive indicate unemployed)
df.loc[df['DAYS_EMPLOYED'] > 0, 'EMPLOYMENT_YEARS'] = 0

# Create a sample of the data for visualization to reduce data transfer
sample_size = min(5000, len(df))
df_sample = df.sample(sample_size, random_state=42) if len(df) > 5000 else df

# ===================== HEADER =====================
st.title("📊 Risk Segmentation Analysis")
st.markdown("""
This dashboard analyzes how default rates vary across different applicant segments,
helping identify high-risk and low-risk customer profiles.
""")

# ===================== KPI CALCULATIONS =====================
# ===================== KPI CALCULATIONS =====================
total_defaults = df['TARGET'].sum()
default_rate = df['TARGET'].mean() * 100

# Segment default rates
default_rate_gender = df.groupby('CODE_GENDER')['TARGET'].mean() * 100 if 'CODE_GENDER' in df.columns else pd.Series()
default_rate_education = df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean() * 100 if 'NAME_EDUCATION_TYPE' in df.columns else pd.Series()
default_rate_family = df.groupby('NAME_FAMILY_STATUS')['TARGET'].mean() * 100 if 'NAME_FAMILY_STATUS' in df.columns else pd.Series()
default_rate_housing = df.groupby('NAME_HOUSING_TYPE')['TARGET'].mean() * 100 if 'NAME_HOUSING_TYPE' in df.columns else pd.Series()

# Averages for defaulters
defaulters = df[df['TARGET'] == 1]
avg_income_default = defaulters['AMT_INCOME_TOTAL'].mean()
avg_credit_default = defaulters['AMT_CREDIT'].mean()
avg_annuity_default = defaulters['AMT_ANNUITY'].mean() if 'AMT_ANNUITY' in df.columns else 0
avg_employment_default = defaulters['EMPLOYMENT_YEARS'].mean()

# ===================== KPI DISPLAY =====================
st.subheader("📈 Default Risk Metrics")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Defaults", f"{total_defaults:,.0f}")
col2.metric("Overall Default Rate", f"{default_rate:.2f}%")
col3.metric("Default Rate - Gender", f"{default_rate_gender.mean():.2f}%")
col4.metric("Default Rate - Education", f"{default_rate_education.mean():.2f}%")
col5.metric("Default Rate - Family", f"{default_rate_family.mean():.2f}%")

col6, col7, col8, col9, col10 = st.columns(5)
col6.metric("Avg Income (Defaulters)", f"${avg_income_default:,.0f}")
col7.metric("Avg Credit (Defaulters)", f"${avg_credit_default:,.0f}")
col8.metric("Avg Annuity (Defaulters)", f"${avg_annuity_default:,.0f}")
col9.metric("Avg Employment (Years)", f"{avg_employment_default:.1f}")
col10.metric("Default Rate - Housing", f"{default_rate_housing.mean():.2f}%")

st.markdown("---")

# ===================== SEGMENTATION ANALYSIS =====================
st.subheader("🔍 Default Rates by Segment")

# Create tabs for different segmentation categories
tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Default Rates", "Income & Credit by Target", "Contract Types"])

with tab1:
    col1, col2= st.columns(2)
    
    with col1:
        # Default vs Repaid counts
        st.write("**Default vs Repaid Counts**")
        target_counts = df['TARGET'].value_counts().reset_index()
        target_counts.columns = ['TARGET', 'count']
        target_counts['TARGET'] = target_counts['TARGET'].map({0: 'Repaid', 1: 'Default'})
        target_chart = alt.Chart(target_counts).mark_bar().encode(
            x=alt.X('TARGET:N', title='Repayment Status'),
            y=alt.Y('count:Q', title='Count'),
            color=alt.Color('TARGET:N', scale=alt.Scale(domain=['Repaid', 'Default'], range=['#2E8B57', '#DC143C']),
                          legend=alt.Legend(title="Repayment Status")),
            tooltip=['TARGET', 'count']
        ).properties(height=300, width=400)
        st.altair_chart(target_chart, use_container_width=True)
    with col2:
        # Default rate by gender
        if not default_rate_gender.empty:
            st.write("**Default Rate by Gender**")
            gender_df = default_rate_gender.reset_index()
            gender_df.columns = ['Gender', 'Default_Rate']
            gender_chart = alt.Chart(gender_df).mark_bar().encode(
                x=alt.X('Gender:N', title='Gender'),
                y=alt.Y('Default_Rate:Q', title='Default Rate (%)'),
                color=alt.Color('Gender:N', legend=alt.Legend(title="Gender")),
                tooltip=['Gender', 'Default_Rate']
            ).properties(height=300, width=400)
            st.altair_chart(gender_chart, use_container_width=True)
    
    


with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Default rate by education
        if not default_rate_education.empty:
            st.write("**Default Rate by Education Level**")
            education_df = default_rate_education.reset_index()
            education_df.columns = ['Education_Level', 'Default_Rate']
            education_chart = alt.Chart(education_df).mark_bar().encode(
                x=alt.X('Education_Level:N', sort='-y', title='Education Level'),
                y=alt.Y('Default_Rate:Q', title='Default Rate (%)'),
                color=alt.Color('Education_Level:N', legend=None),
                tooltip=['Education_Level', 'Default_Rate']
            ).properties(height=300, width=400)
            st.altair_chart(education_chart, use_container_width=True)
    
        # Default rate by family status
        if not default_rate_family.empty:
            st.write("**Default Rate by Family Status**")
            family_df = default_rate_family.reset_index()
            family_df.columns = ['Family_Status', 'Default_Rate']
            family_chart = alt.Chart(family_df).mark_bar().encode(
                x=alt.X('Family_Status:N', sort='-y', title='Family Status'),
                y=alt.Y('Default_Rate:Q', title='Default Rate (%)'),
                color=alt.Color('Family_Status:N', legend=None),
                tooltip=['Family_Status', 'Default_Rate']
            ).properties(height=300, width=400)
            st.altair_chart(family_chart, use_container_width=True)
    with col2:
           
        # Default rate by family status
        if not default_rate_family.empty:
            st.write("**Default Rate by Family Status**")
            family_df = default_rate_family.reset_index()
            family_df.columns = ['Family_Status', 'Default_Rate']
            family_chart = alt.Chart(family_df).mark_bar().encode(
                x=alt.X('Family_Status:N', sort='-y', title='Family Status'),
                y=alt.Y('Default_Rate:Q', title='Default Rate (%)'),
                color=alt.Color('Family_Status:N', legend=None),
                tooltip=['Family_Status', 'Default_Rate']
            ).properties(height=300, width=400)
            st.altair_chart(family_chart, use_container_width=True)
    
        # Default rate by housing type
        if not default_rate_housing.empty:
            st.write("**Default Rate by Housing Type**")
            housing_df = default_rate_housing.reset_index()
            housing_df.columns = ['Housing_Type', 'Default_Rate']
            housing_chart = alt.Chart(housing_df).mark_bar().encode(
                x=alt.X('Housing_Type:N', sort='-y', title='Housing Type'),
                y=alt.Y('Default_Rate:Q', title='Default Rate (%)'),
                color=alt.Color('Housing_Type:N', legend=None),
                tooltip=['Housing_Type', 'Default_Rate']
            ).properties(height=300, width=400)
            st.altair_chart(housing_chart, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        # Income by target
        st.write("**Income Distribution by Target**")
        # Filter outliers for better visualization
        income_q99 = df['AMT_INCOME_TOTAL'].quantile(0.99)
        income_filtered = df_sample[df_sample['AMT_INCOME_TOTAL'] <= income_q99]
        
        # Convert target to categorical for proper labeling
        income_filtered = income_filtered.copy()
        income_filtered['Default_Status'] = income_filtered['TARGET'].map({0: 'No', 1: 'Yes'})
        
        income_chart = alt.Chart(income_filtered).mark_boxplot().encode(
            x=alt.X('Default_Status:N', title='Defaulted'),
            y=alt.Y('AMT_INCOME_TOTAL:Q', title='Income', scale=alt.Scale(zero=False)),
            color=alt.Color('Default_Status:N', scale=alt.Scale(domain=['No', 'Yes'], range=['#2E8B57', '#DC143C']),
                          legend=alt.Legend(title="Defaulted")),
            tooltip=['Default_Status', 'AMT_INCOME_TOTAL']
        ).properties(height=300, width=400)
        st.altair_chart(income_chart, use_container_width=True)
    with col2:   
        # Credit by target
        st.write("**Credit Amount by Target**")
        credit_q99 = df['AMT_CREDIT'].quantile(0.99)
        credit_filtered = df_sample[df_sample['AMT_CREDIT'] <= credit_q99]
        credit_filtered = credit_filtered.copy()
        credit_filtered['Default_Status'] = credit_filtered['TARGET'].map({0: 'No', 1: 'Yes'})
        
        credit_chart = alt.Chart(credit_filtered).mark_boxplot().encode(
            x=alt.X('Default_Status:N', title='Defaulted'),
            y=alt.Y('AMT_CREDIT:Q', title='Credit Amount', scale=alt.Scale(zero=False)),
            color=alt.Color('Default_Status:N', scale=alt.Scale(domain=['No', 'Yes'], range=['#2E8B57', '#DC143C']),
                          legend=alt.Legend(title="Defaulted")),
            tooltip=['Default_Status', 'AMT_CREDIT']
        ).properties(height=300, width=400)
        st.altair_chart(credit_chart, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
    # ================= HISTOGRAM: EMPLOYMENT YEARS =================
        st.write("**Employment Years Distribution by Repayment Status**")

        # Filter extreme employment values (e.g., cap at 50 years for clarity)
        employment_filtered = df_sample[df_sample['EMPLOYMENT_YEARS'] <= 50].copy()
        employment_filtered['Default_Status'] = employment_filtered['TARGET'].map({0: 'Repaid', 1: 'Defaulted'})

        employment_hist = alt.Chart(employment_filtered).mark_bar(opacity=0.7).encode(
            x=alt.X('EMPLOYMENT_YEARS:Q', bin=alt.Bin(maxbins=20), title="Employment Years"),
            y=alt.Y('count()', title="Count"),
            color=alt.Color('Default_Status:N',
                            scale=alt.Scale(domain=['Repaid', 'Defaulted'], range=['#2E8B57', '#DC143C']),
                            legend=alt.Legend(title="Repayment Status")),
            tooltip=['EMPLOYMENT_YEARS', 'count()', 'Default_Status']
        ).properties(height=300, width=400)

        st.altair_chart(employment_hist, use_container_width=True)

    with col2:
    # ================= STACKED BAR: CONTRACT TYPE =================
        if 'NAME_CONTRACT_TYPE' in df.columns:
            st.write("**Contract Type vs Repayment Status**")

            contract_data = df.groupby(['NAME_CONTRACT_TYPE', 'TARGET']).size().reset_index(name='Count')
            contract_data['Default_Status'] = contract_data['TARGET'].map({0: 'Repaid', 1: 'Defaulted'})

            contract_chart = alt.Chart(contract_data).mark_bar().encode(
                x=alt.X('NAME_CONTRACT_TYPE:N', title='Contract Type'),
                y=alt.Y('Count:Q', title='Count'),
                color=alt.Color('Default_Status:N',
                                scale=alt.Scale(domain=['Repaid', 'Defaulted'], range=['#2E8B57', '#DC143C']),
                                legend=alt.Legend(title="Repayment Status")),
                tooltip=['NAME_CONTRACT_TYPE', 'Default_Status', 'Count']
            ).properties(height=300, width=400)

            st.altair_chart(contract_chart, use_container_width=True)


    st.markdown("---")

# ===================== INSIGHTS SECTION =====================
st.subheader("🔍 Risk Segmentation Insights")

# Find segments with highest and lowest default rates
insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown("**Highest Risk Segments:**")
    
    # Education insights
    if not default_rate_education.empty:
        highest_edu = default_rate_education.idxmax()
        highest_edu_rate = default_rate_education.max()
        st.markdown(f"- **Education**: {highest_edu} ({highest_edu_rate:.1f}% default rate)")
    
    # Family status insights
    if not default_rate_family.empty:
        highest_family = default_rate_family.idxmax()
        highest_family_rate = default_rate_family.max()
        st.markdown(f"- **Family Status**: {highest_family} ({highest_family_rate:.1f}% default rate)")
    
    # Housing insights
    if not default_rate_housing.empty:
        highest_housing = default_rate_housing.idxmax()
        highest_housing_rate = default_rate_housing.max()
        st.markdown(f"- **Housing Type**: {highest_housing} ({highest_housing_rate:.1f}% default rate)")

with insight_col2:
    st.markdown("**Lowest Risk Segments:**")
    
    # Education insights
    if not default_rate_education.empty:
        lowest_edu = default_rate_education.idxmin()
        lowest_edu_rate = default_rate_education.min()
        st.markdown(f"- **Education**: {lowest_edu} ({lowest_edu_rate:.1f}% default rate)")
    
    # Family status insights
    if not default_rate_family.empty:
        lowest_family = default_rate_family.idxmin()
        lowest_family_rate = default_rate_family.min()
        st.markdown(f"- **Family Status**: {lowest_family} ({lowest_family_rate:.1f}% default rate)")
    
    # Housing insights
    if not default_rate_housing.empty:
        lowest_housing = default_rate_housing.idxmin()
        lowest_housing_rate = default_rate_housing.min()
        st.markdown(f"- **Housing Type**: {lowest_housing} ({lowest_housing_rate:.1f}% default rate)")

# Additional insights
st.markdown("""
**Key Risk Factors Hypothesis:**

1. **Income-to-Debt Mismatch**: Applicants with lower income but higher credit amounts show elevated default rates,
suggesting loan affordability issues.

2. **Employment Instability**: Those with shorter employment histories or frequent job changes demonstrate
higher default probabilities, indicating income instability.

3. **Life Stage Factors**: Certain family statuses and housing types correlate with financial stress points
(e.g., single parents, renters) which may increase default risk.
""")

# ===================== FOOTER =====================
st.markdown("---")

