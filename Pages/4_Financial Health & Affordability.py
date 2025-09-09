import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns   
from utils.load_data import load_data

# Load data
df = load_data()

# Page configuration
st.set_page_config(page_title="💰 Financial Health", layout="wide", page_icon="💰")

# Create a sample of the data for visualization to reduce data transfer
sample_size = min(5000, len(df))
df_sample = df.sample(sample_size, random_state=42) if len(df) > 5000 else df

# ===================== HEADER =====================
st.title("💰 Financial Health & Affordability Analysis")
st.markdown("""
This dashboard analyzes applicants' financial health, affordability indicators, 
and how these factors relate to credit risk and repayment ability.
""")

# ===================== KPI CALCULATIONS =====================
# Basic financial metrics
avg_income = df['AMT_INCOME_TOTAL'].mean()
median_income = df['AMT_INCOME_TOTAL'].median()
avg_credit = df['AMT_CREDIT'].mean()
avg_annuity = df['AMT_ANNUITY'].mean() if 'AMT_ANNUITY' in df.columns else 0
avg_goods_price = df['AMT_GOODS_PRICE'].mean() if 'AMT_GOODS_PRICE' in df.columns else 0

# Debt ratios
if 'AMT_ANNUITY' in df.columns:
    df['DTI'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    avg_dti = df['DTI'].mean()
else:
    avg_dti = 0

df['LTI'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
avg_lti = df['LTI'].mean()

# Income and credit gaps between defaulters and non-defaulters
defaulters = df[df['TARGET'] == 1]
non_defaulters = df[df['TARGET'] == 0]

income_gap = non_defaulters['AMT_INCOME_TOTAL'].mean() - defaulters['AMT_INCOME_TOTAL'].mean()
credit_gap = non_defaulters['AMT_CREDIT'].mean() - defaulters['AMT_CREDIT'].mean()

# High credit percentage
percent_high_credit = (df['AMT_CREDIT'] > 1000000).mean() * 100

# Calculate DTI and LTI risk thresholds (simplified for this example)
dti_threshold = 0.4  # Example threshold
lti_threshold = 5.0  # Example threshold

# Calculate default rates above and below thresholds
if 'DTI' in df.columns:
    high_dti_default_rate = df[df['DTI'] > dti_threshold]['TARGET'].mean() * 100
    low_dti_default_rate = df[df['DTI'] <= dti_threshold]['TARGET'].mean() * 100
else:
    high_dti_default_rate, low_dti_default_rate = 0, 0

high_lti_default_rate = df[df['LTI'] > lti_threshold]['TARGET'].mean() * 100
low_lti_default_rate = df[df['LTI'] <= lti_threshold]['TARGET'].mean() * 100

# ===================== KPI DISPLAY =====================
st.subheader("📊 Financial Health Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Annual Income", f"${avg_income:,.0f}")
col2.metric("Median Annual Income", f"${median_income:,.0f}")
col3.metric("Avg Credit Amount", f"${avg_credit:,.0f}")
col4.metric("Avg Annuity Payment", f"${avg_annuity:,.0f}")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Avg Goods Price", f"${avg_goods_price:,.0f}")
col6.metric("Avg Debt-to-Income (DTI)", f"{avg_dti:.3f}")
col7.metric("Avg Loan-to-Income (LTI)", f"{avg_lti:.2f}")
col8.metric("High Credit (>$1M)", f"{percent_high_credit:.1f}%")

col9, col10, col11, col12 = st.columns(4)
col9.metric("Income Gap (Non-def − Def)", f"${income_gap:,.0f}")
col10.metric("Credit Gap (Non-def − Def)", f"${credit_gap:,.0f}")

st.markdown("---")

# ===================== VISUALIZATIONS =====================
st.subheader("📈 Financial Distributions & Relationships")

# Create tabs for different visualization categories
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Distributions", "Comparisons", "Relationships", "Affordability", "KDE"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Income distribution
        st.write("**Income Distribution**")
        income_q99 = df['AMT_INCOME_TOTAL'].quantile(0.99)
        income_filtered = df_sample[df_sample['AMT_INCOME_TOTAL'] <= income_q99]
        
        income_chart = alt.Chart(income_filtered).mark_bar(color='steelblue', opacity=0.7).encode(
            alt.X("AMT_INCOME_TOTAL:Q", bin=alt.Bin(maxbins=30), title="Annual Income"),
            alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip('AMT_INCOME_TOTAL:Q', bin=True), 'count()']
        ).properties(height=300)
        st.altair_chart(income_chart, use_container_width=True)
        
        # Annuity distribution
        if 'AMT_ANNUITY' in df.columns:
            st.write("**Annuity Distribution**")
            annuity_q99 = df['AMT_ANNUITY'].quantile(0.99)
            annuity_filtered = df_sample[df_sample['AMT_ANNUITY'] <= annuity_q99]
            
            annuity_chart = alt.Chart(annuity_filtered).mark_bar(color='orange', opacity=0.7).encode(
                alt.X("AMT_ANNUITY:Q", bin=alt.Bin(maxbins=30), title="Annuity Amount"),
                alt.Y("count()", title="Count"),
                tooltip=[alt.Tooltip('AMT_ANNUITY:Q', bin=True), 'count()']
            ).properties(height=300)
            st.altair_chart(annuity_chart, use_container_width=True)
    
    with col2:
        # Credit distribution
        st.write("**Credit Distribution**")
        credit_q99 = df['AMT_CREDIT'].quantile(0.99)
        credit_filtered = df_sample[df_sample['AMT_CREDIT'] <= credit_q99]
        
        credit_chart = alt.Chart(credit_filtered).mark_bar(color='green', opacity=0.7).encode(
            alt.X("AMT_CREDIT:Q", bin=alt.Bin(maxbins=30), title="Credit Amount"),
            alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip('AMT_CREDIT:Q', bin=True), 'count()']
        ).properties(height=300)
        st.altair_chart(credit_chart, use_container_width=True)
        

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Income by Target
        st.write("**Income by Repayment Status**")
        income_sample = df_sample.copy()
        income_sample['Default_Status'] = income_sample['TARGET'].map({0: 'No', 1: 'Yes'})
        income_q99 = df['AMT_INCOME_TOTAL'].quantile(0.99)
        income_sample = income_sample[income_sample['AMT_INCOME_TOTAL'] <= income_q99]
        
        income_box = alt.Chart(income_sample).mark_boxplot().encode(
            x=alt.X('Default_Status:N', title='Defaulted'),
            y=alt.Y('AMT_INCOME_TOTAL:Q', title='Income', scale=alt.Scale(zero=False)),
            color=alt.Color('Default_Status:N', scale=alt.Scale(domain=['No', 'Yes'], range=['#2E8B57', '#DC143C']),
                          legend=alt.Legend(title="Defaulted")),
            tooltip=['Default_Status', 'AMT_INCOME_TOTAL']
        ).properties(height=300)
        st.altair_chart(income_box, use_container_width=True)
    
    with col2:
        # Credit by Target
        st.write("**Credit by Repayment Status**")
        credit_sample = df_sample.copy()
        credit_sample['Default_Status'] = credit_sample['TARGET'].map({0: 'No', 1: 'Yes'})
        credit_q99 = df['AMT_CREDIT'].quantile(0.99)
        credit_sample = credit_sample[credit_sample['AMT_CREDIT'] <= credit_q99]
        
        credit_box = alt.Chart(credit_sample).mark_boxplot().encode(
            x=alt.X('Default_Status:N', title='Defaulted'),
            y=alt.Y('AMT_CREDIT:Q', title='Credit Amount', scale=alt.Scale(zero=False)),
            color=alt.Color('Default_Status:N', scale=alt.Scale(domain=['No', 'Yes'], range=['#2E8B57', '#DC143C']),
                          legend=alt.Legend(title="Defaulted")),
            tooltip=['Default_Status', 'AMT_CREDIT']
        ).properties(height=300)
        st.altair_chart(credit_box, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Income vs Credit scatter - FIXED
        st.write("**Income vs Credit Amount**")
        
        # Create a smaller sample for scatter plots to improve performance
        scatter_sample_size = min(1000, len(df))
        scatter_sample = df.sample(scatter_sample_size, random_state=42) if len(df) > 1000 else df
        
        # Filter outliers
        income_credit_q99 = df[['AMT_INCOME_TOTAL', 'AMT_CREDIT']].quantile(0.95)
        scatter_sample = scatter_sample[
            (scatter_sample['AMT_INCOME_TOTAL'] <= income_credit_q99['AMT_INCOME_TOTAL']) &
            (scatter_sample['AMT_CREDIT'] <= income_credit_q99['AMT_CREDIT'])
        ]
        
        # Convert target to categorical for better legend
        scatter_sample = scatter_sample.copy()
        scatter_sample['Default_Status'] = scatter_sample['TARGET'].map({0: 'No', 1: 'Yes'})
        
        scatter_ic = alt.Chart(scatter_sample).mark_circle(opacity=0.6, size=40).encode(
            x=alt.X('AMT_INCOME_TOTAL:Q', title='Income', scale=alt.Scale(zero=False)),
            y=alt.Y('AMT_CREDIT:Q', title='Credit Amount', scale=alt.Scale(zero=False)),
            color=alt.Color('Default_Status:N', 
                          scale=alt.Scale(domain=['No', 'Yes'], range=['#2E8B57', '#DC143C']),
                          legend=alt.Legend(title="Defaulted")),
            tooltip=['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'Default_Status']
        ).properties(height=400, width=500)
        st.altair_chart(scatter_ic, use_container_width=True)
    
    with col2:
        # Income vs Annuity scatter - FIXED
        if 'AMT_ANNUITY' in df.columns:
            st.write("**Income vs Annuity Payment**")
            
            # Use the same sample for consistency
            scatter_sample_ia = scatter_sample.copy()
            
            # Filter outliers for annuity
            income_annuity_q99 = df[['AMT_INCOME_TOTAL', 'AMT_ANNUITY']].quantile(0.95)
            scatter_sample_ia = scatter_sample_ia[
                (scatter_sample_ia['AMT_INCOME_TOTAL'] <= income_annuity_q99['AMT_INCOME_TOTAL']) &
                (scatter_sample_ia['AMT_ANNUITY'] <= income_annuity_q99['AMT_ANNUITY'])
            ]
            
            scatter_ia = alt.Chart(scatter_sample_ia).mark_circle(opacity=0.6, size=40).encode(
                x=alt.X('AMT_INCOME_TOTAL:Q', title='Income', scale=alt.Scale(zero=False)),
                y=alt.Y('AMT_ANNUITY:Q', title='Annuity Payment', scale=alt.Scale(zero=False)),
                color=alt.Color('Default_Status:N', 
                              scale=alt.Scale(domain=['No', 'Yes'], range=['#2E8B57', '#DC143C']),
                              legend=alt.Legend(title="Defaulted")),
                tooltip=['AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'Default_Status']
            ).properties(height=400, width=500)
            st.altair_chart(scatter_ia, use_container_width=True)

with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        # Income brackets vs default rate
        st.write("**Default Rate by Income Brackets**")
        # Create income brackets
        income_brackets = pd.cut(df['AMT_INCOME_TOTAL'], bins=10, precision=0)
        bracket_defaults = df.groupby(income_brackets)['TARGET'].mean().reset_index()
        bracket_defaults['Default_Rate'] = bracket_defaults['TARGET'] * 100
        bracket_defaults['Income_Bracket'] = bracket_defaults['AMT_INCOME_TOTAL'].astype(str)
        
        bracket_chart = alt.Chart(bracket_defaults).mark_bar().encode(
            x=alt.X('Income_Bracket:N', title='Income Bracket', sort='-y'),
            y=alt.Y('Default_Rate:Q', title='Default Rate (%)'),
            color=alt.Color('Default_Rate:Q', scale=alt.Scale(scheme='redyellowgreen'), legend=None),
            tooltip=['Income_Bracket', 'Default_Rate']
        ).properties(height=300)
        st.altair_chart(bracket_chart, use_container_width=True)
    
    with col2:
        # Financial variable correlations heatmap
        st.write("**Financial Variable Correlations**")
        
        # Select relevant columns for correlation
        financial_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'TARGET']
        if 'AMT_ANNUITY' in df.columns:
            financial_cols.append('AMT_ANNUITY')
        if 'AMT_GOODS_PRICE' in df.columns:
            financial_cols.append('AMT_GOODS_PRICE')
        if 'DTI' in df.columns:
            financial_cols.append('DTI')
        financial_cols.append('LTI')
        
        # Calculate correlations
        corr_data = df[financial_cols].corr().reset_index().melt('index')
        corr_data.columns = ['Variable1', 'Variable2', 'Correlation']
        
        # Create heatmap
        heatmap = alt.Chart(corr_data).mark_rect().encode(
            x='Variable1:N',
            y='Variable2:N',
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue')),
            tooltip=['Variable1', 'Variable2', 'Correlation']
        ).properties(width=400, height=400)
        
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

with tab5:
    st.header("Joint Income-Credit Distribution")
    st.markdown("This visualization shows the joint distribution of income and credit amount through a Kernel Density Estimation (KDE) plot.")

    # Check if required columns exist
    if 'AMT_INCOME_TOTAL' in df.columns and 'AMT_CREDIT' in df.columns:
        # Create a sample of the data for better performance
        sample_size = min(5000, len(df))
        df_sample = df.sample(sample_size, random_state=42) if len(df) > 5000 else df
        
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create the KDE plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Filter out extreme outliers for better visualization
            income_q99 = df_sample['AMT_INCOME_TOTAL'].quantile(0.99)
            credit_q99 = df_sample['AMT_CREDIT'].quantile(0.99)
            
            filtered_df = df_sample[
                (df_sample['AMT_INCOME_TOTAL'] <= income_q99) & 
                (df_sample['AMT_CREDIT'] <= credit_q99)
            ]
            
            # Create the KDE plot
            sns.kdeplot(
                data=filtered_df, 
                x='AMT_INCOME_TOTAL', 
                y='AMT_CREDIT',
                fill=True,
                cmap='viridis',
                alpha=0.7,
                ax=ax
            )
            
            ax.set_xlabel('Total Income')
            ax.set_ylabel('Credit Amount')
            ax.set_title('Density of Income vs Credit Amount')
            
            # Add a colorbar
            norm = plt.Normalize(filtered_df['AMT_INCOME_TOTAL'].min(), filtered_df['AMT_INCOME_TOTAL'].max())
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Density Intensity')
            
            st.pyplot(fig)
        
        with col2:
            st.subheader("Statistics")
            
            # Calculate basic statistics
            income_mean = df['AMT_INCOME_TOTAL'].mean()
            income_median = df['AMT_INCOME_TOTAL'].median()
            credit_mean = df['AMT_CREDIT'].mean()
            credit_median = df['AMT_CREDIT'].median()
            
            st.metric("Mean Income", f"${income_mean:,.0f}")
            st.metric("Median Income", f"${income_median:,.0f}")
            st.metric("Mean Credit Amount", f"${credit_mean:,.0f}")
            st.metric("Median Credit Amount", f"${credit_median:,.0f}")
            
            # Calculate correlation
            correlation = df['AMT_INCOME_TOTAL'].corr(df['AMT_CREDIT'])
            st.metric("Income-Credit Correlation", f"{correlation:.3f}")

st.markdown("---")

# ===================== INSIGHTS SECTION =====================
st.subheader("💡 Financial Health Insights")

st.markdown("""
**Key Financial Risk Factors:**

1. **Income Disparity**: Non-defaulters earn ${:,.0f} more than defaulters on average, highlighting income as a key differentiator in repayment ability.

2. **Credit Appropriateness**: The ${:,.0f} difference in credit amounts suggests more conservative lending practices may reduce defaults.

3. **Debt-to-Income Ratio**: Applicants with DTI above {:.2f} experience default rates {:.1f}% compared to {:.1f}% for those below this threshold.

4. **Loan-to-Income Ratio**: Applicants with LTI above {:.1f} show default rates of {:.1f}% compared to {:.1f}% for those with more conservative borrowing.

5. **High-Credit Risk**: {:.1f}% of applicants have credit amounts exceeding $1M, representing a concentrated risk segment.

6. **Affordability Thresholds**: DTI of {:.2f} and LTI of {:.1f} appear to be critical thresholds where default risk increases significantly.
""".format(
    income_gap,
    -credit_gap,
    dti_threshold,
    high_dti_default_rate,
    low_dti_default_rate,
    lti_threshold,
    high_lti_default_rate,
    low_lti_default_rate,
    percent_high_credit,
    dti_threshold,
    lti_threshold
))

# ===================== FOOTER =====================
st.markdown("---")
