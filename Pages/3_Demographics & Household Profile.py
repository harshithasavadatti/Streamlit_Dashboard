import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from utils.load_data import load_data

# Load data
df = load_data()

# Page configuration
st.set_page_config(page_title="👥 Demographics & Household", layout="wide", page_icon="👥")

# Calculate additional columns
df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).round(1)
df['EMPLOYMENT_YEARS'] = (-df['DAYS_EMPLOYED'] / 365).round(1)
# Fix invalid employment values (those positive indicate unemployed)
df.loc[df['DAYS_EMPLOYED'] > 0, 'EMPLOYMENT_YEARS'] = 0

# Create a sample of the data for visualization to reduce data transfer
sample_size = min(5000, len(df))
df_sample = df.sample(sample_size, random_state=42) if len(df) > 5000 else df

# ===================== HEADER =====================
st.title("👥 Applicant Demographics & Household Profile")
st.markdown("""
This dashboard provides insights into applicant demographics, household structure, 
and how these factors relate to credit risk.
""")

# ===================== KPI CALCULATIONS =====================
# Gender metrics
if 'CODE_GENDER' in df.columns:
    gender_counts = df['CODE_GENDER'].value_counts(normalize=True) * 100
    percent_male = gender_counts.get('M', 0)
    percent_female = gender_counts.get('F', 0)
else:
    percent_male, percent_female = 0, 0

# Age metrics
defaulters = df[df['TARGET'] == 1]
non_defaulters = df[df['TARGET'] == 0]
avg_age_defaulters = defaulters['AGE_YEARS'].mean() if 'AGE_YEARS' in df.columns else 0
avg_age_non_defaulters = non_defaulters['AGE_YEARS'].mean() if 'AGE_YEARS' in df.columns else 0

# Family metrics
if 'CNT_CHILDREN' in df.columns:
    percent_with_children = (df['CNT_CHILDREN'] > 0).mean() * 100
else:
    percent_with_children = 0

if 'CNT_FAM_MEMBERS' in df.columns:
    avg_family_size = df['CNT_FAM_MEMBERS'].mean()
else:
    avg_family_size = 0

# Marital status
if 'NAME_FAMILY_STATUS' in df.columns:
    married_mask = df['NAME_FAMILY_STATUS'].str.contains('Married', case=False, na=False)
    percent_married = married_mask.mean() * 100
    percent_single = 100 - percent_married
else:
    percent_married, percent_single = 0, 0

# Education
if 'NAME_EDUCATION_TYPE' in df.columns:
    higher_ed_mask = df['NAME_EDUCATION_TYPE'].str.contains('higher', case=False, na=False)
    percent_higher_ed = higher_ed_mask.mean() * 100
else:
    percent_higher_ed = 0

# Housing
if 'NAME_HOUSING_TYPE' in df.columns:
    with_parents_mask = df['NAME_HOUSING_TYPE'] == 'With parents'
    percent_with_parents = with_parents_mask.mean() * 100
else:
    percent_with_parents = 0

# Employment
if 'DAYS_EMPLOYED' in df.columns:
    working_mask = df['DAYS_EMPLOYED'] < 0  # Negative values indicate employed
    percent_working = working_mask.mean() * 100
else:
    percent_working = 0

# Employment years
avg_employment_years = df['EMPLOYMENT_YEARS'].mean() if 'EMPLOYMENT_YEARS' in df.columns else 0

# ===================== KPI DISPLAY =====================
st.subheader("📊 Demographic Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Male vs Female", f"{percent_male:.1f}% : {percent_female:.1f}%")
    st.metric("Avg Age (Defaulters)", f"{avg_age_defaulters:.1f} years")
    st.metric("Avg Age (Non-Defaulters)", f"{avg_age_non_defaulters:.1f} years")
    st.metric("With Children", f"{percent_with_children:.1f}%")

with col2:
    st.metric("Avg Family Size", f"{avg_family_size:.1f}")
    st.metric("Married vs Single", f"{percent_married:.1f}% : {percent_single:.1f}%")
    st.metric("Higher Education", f"{percent_higher_ed:.1f}%")

with col3:
    st.metric("Living With Parents", f"{percent_with_parents:.1f}%")
    st.metric("Currently Working", f"{percent_working:.1f}%")
    st.metric("Avg Employment Years", f"{avg_employment_years:.1f} years")

st.markdown("---")

# ===================== VISUALIZATIONS =====================
st.subheader("📈 Demographic Distributions")

# Create tabs for different visualization categories
tab1, tab2, tab3, tab4 = st.tabs(["Age Analysis", "Family Structure", "Education & Occupation", "Correlations"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution histogram
        st.write("**Age Distribution (All Applicants)**")
        if 'AGE_YEARS' in df.columns:
            age_chart = alt.Chart(df_sample).mark_bar(color='steelblue', opacity=0.7).encode(
                alt.X("AGE_YEARS:Q", bin=alt.Bin(maxbins=30), title="Age (Years)"),
                alt.Y("count()", title="Number of Applicants"),
                tooltip=[alt.Tooltip('AGE_YEARS:Q', bin=True), 'count()']
            ).properties(height=300)
            st.altair_chart(age_chart, use_container_width=True)
        
        # Boxplot - Age vs Target
        st.write("**Age Vs Target**")
        if 'AGE_YEARS' in df.columns:
            age_sample = df_sample.copy()
            age_sample['Default_Status'] = age_sample['TARGET'].map({0: 'No', 1: 'Yes'})
            
            age_box = alt.Chart(age_sample).mark_boxplot().encode(
                x=alt.X('Default_Status:N', title='Defaulted'),
                y=alt.Y('AGE_YEARS:Q', title='Age (Years)', scale=alt.Scale(zero=False)),
                color=alt.Color('Default_Status:N', scale=alt.Scale(domain=['No', 'Yes'], range=['#2E8B57', '#DC143C']),
                              legend=alt.Legend(title="Defaulted")),
                tooltip=['Default_Status', 'AGE_YEARS']
            ).properties(height=300)
            st.altair_chart(age_box, use_container_width=True)
    
    with col2:
        # Age by Target (overlay histogram)
        st.write("**Age Distribution by Target (Overlay)**")
        if 'AGE_YEARS' in df.columns:
            # Create separate data for defaulters and non-defaulters
            non_def = df_sample[df_sample['TARGET'] == 0]
            defs = df_sample[df_sample['TARGET'] == 1]
            
            # Create histograms
            non_def_hist = alt.Chart(non_def).mark_bar(opacity=0.6, color='#2E8B57').encode(
                alt.X('AGE_YEARS:Q', bin=alt.Bin(maxbins=30), title='Age (Years)'),
                alt.Y('count()', title='Count'),
                tooltip=[alt.Tooltip('AGE_YEARS:Q', bin=True), 'count()']
            )
            
            def_hist = alt.Chart(defs).mark_bar(opacity=0.6, color='#DC143C').encode(
                alt.X('AGE_YEARS:Q', bin=alt.Bin(maxbins=30), title='Age (Years)'),
                alt.Y('count()', title='Count'),
                tooltip=[alt.Tooltip('AGE_YEARS:Q', bin=True), 'count()']
            )
            
            overlay_chart = alt.layer(non_def_hist, def_hist).properties(height=300)
            st.altair_chart(overlay_chart, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Gender distribution
        if 'CODE_GENDER' in df.columns:
            st.write("**Gender Distribution**")
            gender_counts = df['CODE_GENDER'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            gender_chart = alt.Chart(gender_counts).mark_bar().encode(
                x='Gender',
                y='Count',
                color='Gender',
                tooltip=['Gender', 'Count']
            ).properties(height=300)
            st.altair_chart(gender_chart, use_container_width=True)
        
        # Family status distribution
        if 'NAME_FAMILY_STATUS' in df.columns:
            st.write("**Family Status Distribution**")
            family_counts = df['NAME_FAMILY_STATUS'].value_counts().reset_index()
            family_counts.columns = ['Family_Status', 'Count']
            family_chart = alt.Chart(family_counts).mark_bar().encode(
                x=alt.X('Family_Status:N', sort='-y', title='Family Status'),
                y=alt.Y('Count:Q', title='Count'),
                color=alt.Color('Family_Status:N', legend=None),
                tooltip=['Family_Status', 'Count']
            ).properties(height=300)
            st.altair_chart(family_chart, use_container_width=True)
    
    with col2:
        # Children count
        if 'CNT_CHILDREN' in df.columns:
            st.write("**Number of Children**")
            children_counts = df['CNT_CHILDREN'].value_counts().reset_index()
            children_counts.columns = ['Children_Count', 'Count']
            children_chart = alt.Chart(children_counts).mark_bar().encode(
                x=alt.X('Children_Count:N', title='Number of Children'),
                y=alt.Y('Count:Q', title='Count'),
                color=alt.Color('Children_Count:N', legend=None),
                tooltip=['Children_Count', 'Count']
            ).properties(height=300)
            st.altair_chart(children_chart, use_container_width=True)
        
        # Housing type distribution (pie chart)
        if 'NAME_HOUSING_TYPE' in df.columns:
            st.write("**Housing Type Distribution**")
            housing_counts = df['NAME_HOUSING_TYPE'].value_counts().reset_index()
            housing_counts.columns = ['Housing_Type', 'Count']
            
            housing_pie = alt.Chart(housing_counts).mark_arc().encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Housing_Type", type="nominal", legend=alt.Legend(title="Housing Type")),
                tooltip=['Housing_Type', 'Count']
            ).properties(height=300, width=400)
            st.altair_chart(housing_pie, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Education distribution
        if 'NAME_EDUCATION_TYPE' in df.columns:
            st.write("**Education Level Distribution**")
            education_counts = df['NAME_EDUCATION_TYPE'].value_counts().reset_index()
            education_counts.columns = ['Education_Level', 'Count']
            education_chart = alt.Chart(education_counts).mark_bar().encode(
                x=alt.X('Education_Level:N', sort='-y', title='Education Level'),
                y=alt.Y('Count:Q', title='Count'),
                color=alt.Color('Education_Level:N', legend=None),
                tooltip=['Education_Level', 'Count']
            ).properties(height=300)
            st.altair_chart(education_chart, use_container_width=True)
    
    with col2:
        # Occupation distribution (top 10)
        if 'OCCUPATION_TYPE' in df.columns:
            st.write("**Top 10 Occupations**")
            occupation_counts = df['OCCUPATION_TYPE'].value_counts().head(10).reset_index()
            occupation_counts.columns = ['Occupation', 'Count']
            occupation_chart = alt.Chart(occupation_counts).mark_bar().encode(
                x=alt.X('Occupation:N', sort='-y', title='Occupation'),
                y=alt.Y('Count:Q', title='Count'),
                color=alt.Color('Occupation:N', legend=None),
                tooltip=['Occupation', 'Count']
            ).properties(height=300)
            st.altair_chart(occupation_chart, use_container_width=True)

with tab4:
    # Heatmap of correlations
    st.write("**Correlation Heatmap: Demographic Factors**")
    
    # Select relevant columns for correlation
    corr_columns = []
    if 'AGE_YEARS' in df.columns:
        corr_columns.append('AGE_YEARS')
    if 'CNT_CHILDREN' in df.columns:
        corr_columns.append('CNT_CHILDREN')
    if 'CNT_FAM_MEMBERS' in df.columns:
        corr_columns.append('CNT_FAM_MEMBERS')
    if 'TARGET' in df.columns:
        corr_columns.append('TARGET')
    if 'EMPLOYMENT_YEARS' in df.columns:
        corr_columns.append('EMPLOYMENT_YEARS')
    
    if len(corr_columns) > 1:
        corr_data = df[corr_columns].corr().reset_index().melt('index')
        corr_data.columns = ['Variable1', 'Variable2', 'Correlation']
        
        # Create heatmap
        heatmap = alt.Chart(corr_data).mark_rect().encode(
            x='Variable1:N',
            y='Variable2:N',
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue')),
            tooltip=['Variable1', 'Variable2', 'Correlation']
        ).properties(width=500, height=400)
        
        # Add text labels
        text = heatmap.mark_text(baseline='middle').encode(
            text=alt.Text('Correlation:Q', format='.2f'),
            color=alt.condition(
                alt.datum.Correlation > 0.5,
                alt.value('white'),
                alt.value('black')
            )
        )
        
        st.altair_chart(heatmap + text, use_container_width=True)
    else:
        st.info("Insufficient data available for correlation analysis.")

st.markdown("---")

# ===================== INSIGHTS SECTION =====================
st.subheader("🔍 Demographic Insights")

st.markdown("""
**Key Demographic Patterns:**

1. **Age Impact**: {} difference in average age between defaulters ({:.1f} years) and non-defaulters ({:.1f} years)
2. **Family Structure**: {:.1f}% of applicants have children, with average family size of {:.1f}
3. **Education Factor**: {:.1f}% have higher education, which correlates with {} default risk
4. **Employment Stability**: {:.1f}% currently employed, with average {:.1f} years of experience
5. **Housing Situation**: {:.1f}% living with parents suggests {} financial independence
6. **Marital Status**: {:.1f}% married applicants show {} repayment behavior
""".format(
    "Significant" if abs(avg_age_defaulters - avg_age_non_defaulters) > 3 else "Minimal",
    avg_age_defaulters,
    avg_age_non_defaulters,
    percent_with_children,
    avg_family_size,
    percent_higher_ed,
    "reduced" if percent_higher_ed > 30 else "similar",
    percent_working,
    avg_employment_years,
    percent_with_parents,
    "delayed" if percent_with_parents > 10 else "early",
    percent_married,
    "more stable" if percent_married > 50 else "similar"
))

# ===================== FOOTER =====================
st.markdown("---")
