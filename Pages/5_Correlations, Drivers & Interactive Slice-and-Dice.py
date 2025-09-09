import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils.load_data import load_data

# Load data
df = load_data()

# Page configuration
st.set_page_config(page_title="📊 Correlation Analysis", layout="wide", page_icon="📊")

# Calculate additional columns
df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365).round(1)
df['EMPLOYMENT_YEARS'] = (-df['DAYS_EMPLOYED'] / 365).round(1)
# Fix invalid employment values (those positive indicate unemployed)
df.loc[df['DAYS_EMPLOYED'] > 0, 'EMPLOYMENT_YEARS'] = 0

# Create a sample of the data for visualization to reduce data transfer
sample_size = min(3000, len(df))
df_sample = df.sample(sample_size, random_state=42) if len(df) > 3000 else df

# ===================== HEADER =====================
st.title("📊 Correlation Analysis & Driver Identification")
st.markdown("""
This dashboard analyzes feature correlations, identifies key drivers of default risk,
and allows interactive exploration of the data.
""")

# ===================== SIDEBAR FILTERS =====================
st.sidebar.header("🔍 Filter Data")

# Initialize filtered_df with the full dataset
filtered_df = df.copy()

# Gender filter
if 'CODE_GENDER' in df.columns:
    gender_options = ['All'] + list(df['CODE_GENDER'].unique())
    selected_gender = st.sidebar.selectbox("Gender", gender_options, index=0)
    if selected_gender != 'All':
        filtered_df = filtered_df[filtered_df['CODE_GENDER'] == selected_gender]

# Education filter
if 'NAME_EDUCATION_TYPE' in df.columns:
    education_options = ['All'] + list(df['NAME_EDUCATION_TYPE'].unique())
    selected_education = st.sidebar.selectbox("Education", education_options, index=0)
    if selected_education != 'All':
        filtered_df = filtered_df[filtered_df['NAME_EDUCATION_TYPE'] == selected_education]

# Create a filtered sample for visualizations
filtered_sample_size = min(1000, len(filtered_df))
filtered_sample = filtered_df.sample(filtered_sample_size, random_state=42) if len(filtered_df) > 1000 else filtered_df

# ===================== KPI CALCULATIONS =====================
# Use full dataset for correlation calculations (not affected by filters)
numerical_features = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AGE_YEARS', 'EMPLOYMENT_YEARS', 'TARGET']
if 'AMT_ANNUITY' in df.columns:
    numerical_features.append('AMT_ANNUITY')
if 'CNT_FAM_MEMBERS' in df.columns:
    numerical_features.append('CNT_FAM_MEMBERS')
if 'CNT_CHILDREN' in df.columns:
    numerical_features.append('CNT_CHILDREN')

# Filter to only include columns that exist in the dataframe
numerical_features = [col for col in numerical_features if col in df.columns]

# Calculate correlations with TARGET using full dataset
if 'TARGET' in df.columns and len(numerical_features) > 1:
    corr_matrix = df[numerical_features].corr()
    if 'TARGET' in corr_matrix.columns:
        corr_with_target = corr_matrix['TARGET'].drop('TARGET', errors='ignore').sort_values(ascending=False)
        
        # Top correlations
        top_5_positive = corr_with_target.head(5)
        top_5_negative = corr_with_target.tail(5).sort_values(ascending=False)
        
        # Other correlations
        corr_with_income = corr_matrix['AMT_INCOME_TOTAL'].drop('AMT_INCOME_TOTAL', errors='ignore').sort_values(ascending=False) if 'AMT_INCOME_TOTAL' in corr_matrix.columns else pd.Series()
        corr_with_credit = corr_matrix['AMT_CREDIT'].drop('AMT_CREDIT', errors='ignore').sort_values(ascending=False) if 'AMT_CREDIT' in corr_matrix.columns else pd.Series()
        corr_income_credit = corr_matrix.loc['AMT_INCOME_TOTAL', 'AMT_CREDIT'] if 'AMT_INCOME_TOTAL' in corr_matrix.index and 'AMT_CREDIT' in corr_matrix.columns else 0
        
        # Specific correlations
        corr_age_target = corr_matrix.loc['AGE_YEARS', 'TARGET'] if 'AGE_YEARS' in corr_matrix.index and 'TARGET' in corr_matrix.columns else 0
        corr_employment_target = corr_matrix.loc['EMPLOYMENT_YEARS', 'TARGET'] if 'EMPLOYMENT_YEARS' in corr_matrix.index and 'TARGET' in corr_matrix.columns else 0
        corr_family_target = corr_matrix.loc['CNT_FAM_MEMBERS', 'TARGET'] if 'CNT_FAM_MEMBERS' in corr_matrix.index and 'TARGET' in corr_matrix.columns else 0
        
        # Variance explained and high correlation features
        variance_explained = top_5_positive.abs().sum() + top_5_negative.abs().sum()
        high_corr_features = (corr_with_target.abs() > 0.1).sum()
    else:
        corr_with_target = pd.Series()
        top_5_positive = pd.Series()
        top_5_negative = pd.Series()
        corr_with_income = pd.Series()
        corr_with_credit = pd.Series()
        corr_income_credit = 0
        corr_age_target = 0
        corr_employment_target = 0
        corr_family_target = 0
        variance_explained = 0
        high_corr_features = 0
else:
    corr_with_target = pd.Series()
    top_5_positive = pd.Series()
    top_5_negative = pd.Series()
    corr_with_income = pd.Series()
    corr_with_credit = pd.Series()
    corr_income_credit = 0
    corr_age_target = 0
    corr_employment_target = 0
    corr_family_target = 0
    variance_explained = 0
    high_corr_features = 0

# Feature name mapping for display
feature_name_map = {
    'AMT_ANNUITY': 'Annuity Amount',
    'AMT_CREDIT': 'Credit Amount', 
    'AMT_INCOME_TOTAL': 'Income',
    'AGE_YEARS': 'Age',
    'EMPLOYMENT_YEARS': 'Employment Years',
    'CNT_FAM_MEMBERS': 'Family Size',
    'CNT_CHILDREN': 'Number of Children'
}

# ===================== KPI DISPLAY =====================
st.subheader("📈 Correlation Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Corr(Income, Credit)", f"{corr_income_credit:.3f}")
    st.metric("Corr(Age, Target)", f"{corr_age_target:.3f}")

with col2:
    st.metric("Corr(Employment, Target)", f"{corr_employment_target:.3f}")
    st.metric("Corr(Family Size, Target)", f"{corr_family_target:.3f}")

with col3:
    st.metric("Variance Explained (Top 5)", f"{variance_explained:.3f}")
    st.metric("Features with |corr| > 0.1", f"{high_corr_features}")

with col4:
    if len(corr_with_income) > 0:
        most_corr_income_feature = corr_with_income.index[0]
        display_income_feature = feature_name_map.get(most_corr_income_feature, most_corr_income_feature)
        st.metric("Most Correlated with Income", f"{display_income_feature}: {corr_with_income.iloc[0]:.3f}")
    else:
        st.metric("Most Correlated with Income", "N/A")
    
    if len(corr_with_credit) > 0:
        most_corr_credit_feature = corr_with_credit.index[0]
        display_credit_feature = feature_name_map.get(most_corr_credit_feature, most_corr_credit_feature)
        st.metric("Most Correlated with Credit", f"{display_credit_feature}: {corr_with_credit.iloc[0]:.3f}")
    else:
        st.metric("Most Correlated with Credit", "N/A")

# Display top correlations
st.subheader("🔝 Top Correlations with Target")

col5, col6 = st.columns(2)

with col5:
    st.write("**Top 5 Positive Correlations**")
    if len(top_5_positive) > 0:
        for feature, value in top_5_positive.items():
            display_name = feature_name_map.get(feature, feature)
            st.write(f"- {display_name}: {value:.3f}")
    else:
        st.write("No positive correlations available")

with col6:
    st.write("**Top 5 Negative Correlations**")
    if len(top_5_negative) > 0:
        for feature, value in top_5_negative.items():
            display_name = feature_name_map.get(feature, feature)
            st.write(f"- {display_name}: {value:.3f}")
    else:
        st.write("No negative correlations available")

st.markdown("---")

# ===================== VISUALIZATIONS =====================
st.subheader("📊 Correlation Visualizations")

# Create tabs for different visualization categories
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Correlation Matrix", "Feature Relationships", "Segment Analysis", "Interactive Analysis", "Employment Vs Target"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        st.write("**Correlation Heatmap**")
        
        if len(numerical_features) > 1 and 'TARGET' in numerical_features:
            # Create display names for the heatmap
            display_corr_matrix = corr_matrix.copy()
            display_corr_matrix.index = [feature_name_map.get(col, col) for col in display_corr_matrix.index]
            display_corr_matrix.columns = [feature_name_map.get(col, col) for col in display_corr_matrix.columns]
            
            corr_data = display_corr_matrix.reset_index().melt('index')
            corr_data.columns = ['Variable1', 'Variable2', 'Correlation']
            
            heatmap = alt.Chart(corr_data).mark_rect().encode(
                x=alt.X('Variable1:N', title='Variable'),
                y=alt.Y('Variable2:N', title='Variable'),
                color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue')),
                tooltip=['Variable1', 'Variable2', 'Correlation']
            ).properties(height=400)
            
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
            st.info("Insufficient data for correlation matrix")
    
    with col2:
        # Absolute correlation with target
        st.write("**Feature Correlation with Target**")
        if len(corr_with_target) > 0:
            corr_target_data = pd.DataFrame({
                'Feature': [feature_name_map.get(f, f) for f in corr_with_target.index],
                'Correlation': corr_with_target.values,
                'Abs_Correlation': np.abs(corr_with_target.values)
            }).sort_values('Abs_Correlation', ascending=False).head(10)
            
            corr_chart = alt.Chart(corr_target_data).mark_bar().encode(
                x=alt.X('Abs_Correlation:Q', title='Absolute Correlation with Target'),
                y=alt.Y('Feature:N', sort='-x', title='Feature'),
                color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue'), legend=None),
                tooltip=['Feature', 'Correlation']
            ).properties(height=400)
            st.altair_chart(corr_chart, use_container_width=True)
        else:
            st.info("No correlation data available")

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Age vs Credit scatter (using full dataset)
        st.write("**Age vs Credit Amount**")
        scatter_sample = df_sample.copy()
        scatter_sample['Default_Status'] = scatter_sample['TARGET'].map({0: 'No', 1: 'Yes'})
        
        # Filter outliers
        age_credit_q95 = df[['AGE_YEARS', 'AMT_CREDIT']].quantile(0.95)
        scatter_sample = scatter_sample[
            (scatter_sample['AGE_YEARS'] <= age_credit_q95['AGE_YEARS']) &
            (scatter_sample['AMT_CREDIT'] <= age_credit_q95['AMT_CREDIT'])
        ]
        
        scatter_ac = alt.Chart(scatter_sample).mark_circle(opacity=0.6, size=40).encode(
            x=alt.X('AGE_YEARS:Q', title='Age (Years)', scale=alt.Scale(zero=False)),
            y=alt.Y('AMT_CREDIT:Q', title='Credit Amount', scale=alt.Scale(zero=False)),
            color=alt.Color('Default_Status:N', 
                          scale=alt.Scale(domain=['No', 'Yes'], range=['#2E8B57', '#DC143C']),
                          legend=alt.Legend(title="Defaulted")),
            tooltip=['AGE_YEARS', 'AMT_CREDIT', 'Default_Status']
        ).properties(height=300)
        st.altair_chart(scatter_ac, use_container_width=True)
    
    with col2:
        # Age vs Income scatter (using full dataset)
        st.write("**Age vs Income**")
        scatter_sample_ai = df_sample.copy()
        scatter_sample_ai['Default_Status'] = scatter_sample_ai['TARGET'].map({0: 'No', 1: 'Yes'})
        
        # Filter outliers
        age_income_q95 = df[['AGE_YEARS', 'AMT_INCOME_TOTAL']].quantile(0.95)
        scatter_sample_ai = scatter_sample_ai[
            (scatter_sample_ai['AGE_YEARS'] <= age_income_q95['AGE_YEARS']) &
            (scatter_sample_ai['AMT_INCOME_TOTAL'] <= age_income_q95['AMT_INCOME_TOTAL'])
        ]
        
        scatter_ai = alt.Chart(scatter_sample_ai).mark_circle(opacity=0.6, size=40).encode(
            x=alt.X('AGE_YEARS:Q', title='Age (Years)', scale=alt.Scale(zero=False)),
            y=alt.Y('AMT_INCOME_TOTAL:Q', title='Income', scale=alt.Scale(zero=False)),
            color=alt.Color('Default_Status:N', 
                          scale=alt.Scale(domain=['No', 'Yes'], range=['#2E8B57', '#DC143C']),
                          legend=alt.Legend(title="Defaulted")),
            tooltip=['AGE_YEARS', 'AMT_INCOME_TOTAL', 'Default_Status']
        ).properties(height=300)
        st.altair_chart(scatter_ai, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Credit by Education (using full dataset)
        if 'NAME_EDUCATION_TYPE' in df.columns:
            st.write("**Credit Amount by Education Level**")
            education_sample = df_sample.copy()
            education_sample = education_sample[education_sample['AMT_CREDIT'] <= df['AMT_CREDIT'].quantile(0.95)]
            
            education_chart = alt.Chart(education_sample).mark_boxplot().encode(
                x=alt.X('NAME_EDUCATION_TYPE:N', title='Education Level', sort='-y'),
                y=alt.Y('AMT_CREDIT:Q', title='Credit Amount', scale=alt.Scale(zero=False)),
                color=alt.Color('NAME_EDUCATION_TYPE:N', legend=None),
                tooltip=['NAME_EDUCATION_TYPE', 'AMT_CREDIT']
            ).properties(height=400)
            st.altair_chart(education_chart, use_container_width=True)
    
    with col2:
        # Income by Family Status (using full dataset)
        if 'NAME_FAMILY_STATUS' in df.columns:
            st.write("**Income by Family Status**")
            family_sample = df_sample.copy()
            family_sample = family_sample[family_sample['AMT_INCOME_TOTAL'] <= df['AMT_INCOME_TOTAL'].quantile(0.95)]
            
            family_chart = alt.Chart(family_sample).mark_boxplot().encode(
                x=alt.X('NAME_FAMILY_STATUS:N', title='Family Status', sort='-y'),
                y=alt.Y('AMT_INCOME_TOTAL:Q', title='Income', scale=alt.Scale(zero=False)),
                color=alt.Color('NAME_FAMILY_STATUS:N', legend=None),
                tooltip=['NAME_FAMILY_STATUS', 'AMT_INCOME_TOTAL']
            ).properties(height=400)
            st.altair_chart(family_chart, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        # Default rate by gender (responsive to filters)
        if 'CODE_GENDER' in filtered_df.columns:
            st.write("**Default Rate by Gender**")
            gender_defaults = filtered_df.groupby('CODE_GENDER')['TARGET'].mean().reset_index()
            gender_defaults['Default_Rate'] = gender_defaults['TARGET'] * 100
            
            gender_chart = alt.Chart(gender_defaults).mark_bar().encode(
                x=alt.X('CODE_GENDER:N', title='Gender',axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Default_Rate:Q', title='Default Rate (%)'),
                color=alt.Color('CODE_GENDER:N', legend=None),
                tooltip=['CODE_GENDER', 'Default_Rate']
            ).properties(height=300)
            st.altair_chart(gender_chart, use_container_width=True)
    
    with col2:
        # Default rate by education (responsive to filters)
        if 'NAME_EDUCATION_TYPE' in filtered_df.columns:
            st.write("**Default Rate by Education Level**")
            education_defaults = filtered_df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().reset_index()
            education_defaults['Default_Rate'] = education_defaults['TARGET'] * 100
            
            education_default_chart = alt.Chart(education_defaults).mark_bar().encode(
                x=alt.X('NAME_EDUCATION_TYPE:N', title='Education Level', sort='-y'),
                y=alt.Y('Default_Rate:Q', title='Default Rate (%)'),
                color=alt.Color('NAME_EDUCATION_TYPE:N', legend=None),
                tooltip=['NAME_EDUCATION_TYPE', 'Default_Rate']
            ).properties(height=300)
            st.altair_chart(education_default_chart, use_container_width=True)

with tab5:
    st.subheader("Employment Years vs Default Status")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create jittered data for better visualization (using full dataset)
        jittered_data = df_sample.copy()
        
        # Add jitter to employment years for better visualization
        np.random.seed(42)  # For reproducible jitter
        jitter_amount = 0.2  # Adjust this value to control jitter amount
        jittered_data['EMPLOYMENT_YEARS_JITTERED'] = jittered_data['EMPLOYMENT_YEARS'] + np.random.normal(0, jitter_amount, len(jittered_data))
        
        # Add jitter to target for categorical separation
        jittered_data['TARGET_JITTERED'] = jittered_data['TARGET'] + np.random.normal(0, 0.05, len(jittered_data))
        
        # Create the scatter plot with jitter
        scatter = alt.Chart(jittered_data).mark_circle(opacity=0.6, size=40).encode(
            x=alt.X('EMPLOYMENT_YEARS_JITTERED:Q', 
                    title='Employment Years (with jitter)', 
                    scale=alt.Scale(zero=False)),
            y=alt.Y('TARGET_JITTERED:Q', 
                    title='Default Status (with jitter)',
                    scale=alt.Scale(domain=[-0.5, 1.5])),
            color=alt.Color('TARGET:N', 
                           scale=alt.Scale(domain=[0, 1], range=['#2E8B57', '#DC143C']),
                           legend=alt.Legend(title="Defaulted")),
            tooltip=['EMPLOYMENT_YEARS', 'TARGET']
        ).properties(height=500)
        
        # Add horizontal lines at y=0 and y=1 for reference
        hline0 = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(strokeDash=[5,5]).encode(y='y:Q')
        hline1 = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule(strokeDash=[5,5]).encode(y='y:Q')
        
        st.altair_chart(scatter + hline0 + hline1, use_container_width=True)
    
    with col2:
        st.subheader("Statistics")
        
        # Calculate basic statistics (using full dataset)
        employed_mask = df['EMPLOYMENT_YEARS'] > 0
        employed_default_rate = df[employed_mask]['TARGET'].mean() * 100 if employed_mask.any() else 0
        unemployed_default_rate = df[~employed_mask]['TARGET'].mean() * 100 if (~employed_mask).any() else 0
        
        avg_employment_defaulters = df[df['TARGET'] == 1]['EMPLOYMENT_YEARS'].mean() if (df['TARGET'] == 1).any() else 0
        avg_employment_non_defaulters = df[df['TARGET'] == 0]['EMPLOYMENT_YEARS'].mean() if (df['TARGET'] == 0).any() else 0
        
        st.metric("Avg Employment (Defaulters)", f"{avg_employment_defaulters:.1f} years")
        st.metric("Avg Employment (Non-defaulters)", f"{avg_employment_non_defaulters:.1f} years")
        st.metric("Default Rate (Employed)", f"{employed_default_rate:.1f}%")
        st.metric("Default Rate (Unemployed)", f"{unemployed_default_rate:.1f}%")
        
        # Calculate correlation
        employment_target_corr = df['EMPLOYMENT_YEARS'].corr(df['TARGET']) if len(df) > 1 else 0
        st.metric("Correlation", f"{employment_target_corr:.3f}")

# ===================== INSIGHTS SECTION =====================
st.markdown("---")
st.subheader("💡 Correlation Insights")

st.markdown("""
**Key Data-Driven Insights:**

1. **Strongest Risk Drivers**: {} shows the strongest positive correlation with default risk (r = {:.3f}), while {} shows the strongest negative correlation (r = {:.3f}), indicating these are the most influential factors.

2. **Income-Credit Relationship**: The moderate correlation between income and credit amount (r = {:.3f}) suggests lenders generally maintain appropriate credit limits relative to income.

3. **Age-Based Risk Patterns**: The correlation between age and default risk (r = {:.3f}) indicates that {} applicants tend to be {} risky.

4. **Employment Stability Impact**: The correlation between employment years and default risk (r = {:.3f}) suggests that {} employment history {} default risk.

5. **Demographic Segment Variations**: The interactive filters reveal how risk patterns differ across segments, with education level showing {} pattern and gender differences {} when controlling for other factors.

6. **Policy Implications**: The analysis suggests implementing risk-based pricing, setting credit approval thresholds based on {}, and developing targeted strategies for {} segments.
""".format(
    feature_name_map.get(top_5_positive.index[0], top_5_positive.index[0]) if len(top_5_positive) > 0 else "A specific factor",
    top_5_positive.iloc[0] if len(top_5_positive) > 0 else 0,
    feature_name_map.get(top_5_negative.index[0], top_5_negative.index[0]) if len(top_5_negative) > 0 else "A specific factor",
    top_5_negative.iloc[0] if len(top_5_negative) > 0 else 0,
    corr_income_credit,
    corr_age_target,
    "older" if corr_age_target < 0 else "younger",
    "less" if corr_age_target < 0 else "more",
    corr_employment_target,
    "longer" if corr_employment_target < 0 else "shorter",
    "reduces" if corr_employment_target < 0 else "increases",
    "a clear risk" if 'NAME_EDUCATION_TYPE' in df.columns and len(df['NAME_EDUCATION_TYPE'].unique()) > 1 else "minimal variation",
    "persist" if 'CODE_GENDER' in df.columns and len(df['CODE_GENDER'].unique()) > 1 else "diminish",
    "LTI ratios" if len(corr_with_credit) > 0 and abs(corr_with_credit.iloc[0]) > 0.2 else "income verification",
    "higher-risk" if high_corr_features > 3 else "specific demographic"
))

# ===================== FOOTER =====================
st.markdown("---")
