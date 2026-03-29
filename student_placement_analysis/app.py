

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import shap


# --- Load Data and Model ---
@st.cache_data
def load_data():
    return pd.read_csv('data_clean.csv')

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

def get_domain_options():
    return ['Data Science', 'MERN', 'DevOps', 'Cybersecurity']


# --- KPI Dashboard ---
def kpi_dashboard(df):
    st.markdown("""
    <div style='margin-bottom: 1.5rem;'></div>
    <h2 style='color:#F5F6FA; font-weight:700; letter-spacing:0.5px; margin-bottom:0.5rem;'>Key Performance Indicators</h2>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3, gap="large")
    placement_rate = (df['Placement_Status'].value_counts(normalize=True).get('Job', 0) +
                      df['Placement_Status'].value_counts(normalize=True).get('Intern', 0)) * 100
    avg_attendance = df['Attendance'].mean()
    avg_score = df['Assignment_Score'].mean()
    with col1:
        st.metric('Placement Rate (%)', f"{placement_rate:.1f}", help="% of students placed (Job or Intern)")
    with col2:
        st.metric('Avg Attendance', f"{avg_attendance:.1f}", help="Average attendance across all students")
    with col3:
        st.metric('Avg Assignment Score', f"{avg_score:.1f}", help="Average assignment score across all students")


# --- Visual Insights ---
def visual_insights(df):
    st.markdown("""
    <div style='margin-top:2rem; margin-bottom: 1.5rem;'></div>
    <h2 style='color:#F5F6FA; font-weight:700; letter-spacing:0.5px; margin-bottom:0.5rem;'>Visual Insights</h2>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<span style="color:#A3A6B1; font-size:1.1rem; font-weight:600;">Placement Rate by Domain</span>', unsafe_allow_html=True)
        domain_placement = df.groupby('Domain')['Placement_Status'].apply(lambda x: (x.isin(['Job','Intern']).mean())*100)
        fig1, ax1 = plt.subplots(figsize=(4,3))
        sns.barplot(x=domain_placement.index, y=domain_placement.values, palette="viridis", ax=ax1)
        ax1.set_ylabel('Placement Rate (%)')
        ax1.set_xlabel('Domain')
        ax1.set_xticklabels(domain_placement.index, rotation=20, ha='right')
        ax1.set_facecolor('#23272F')
        fig1.patch.set_facecolor('#23272F')
        st.pyplot(fig1, use_container_width=True)
        # Advanced: Pie chart for placement status by domain
        st.markdown('<span style="color:#A3A6B1; font-size:1.1rem; font-weight:600;">Placement Status by Domain (Pie)</span>', unsafe_allow_html=True)
        import plotly.express as px
        pie_df = df.groupby(['Domain','Placement_Status']).size().reset_index(name='Count')
        fig_pie = px.pie(pie_df, values='Count', names='Placement_Status', color='Placement_Status',
                        title='Placement Status Distribution (Pie)',
                        hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.markdown('<span style="color:#A3A6B1; font-size:1.1rem; font-weight:600;">Placement Status Distribution</span>', unsafe_allow_html=True)
        status_counts = df['Placement_Status'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(4,3))
        sns.barplot(x=status_counts.index, y=status_counts.values, palette="mako", ax=ax2)
        ax2.set_ylabel('Count')
        ax2.set_xlabel('Placement Status')
        ax2.set_facecolor('#23272F')
        fig2.patch.set_facecolor('#23272F')
        st.pyplot(fig2, use_container_width=True)
        # Advanced: Violin plot for assignment score by placement
        st.markdown('<span style="color:#A3A6B1; font-size:1.1rem; font-weight:600;">Assignment Score by Placement (Violin)</span>', unsafe_allow_html=True)
        fig_v, ax_v = plt.subplots(figsize=(4,3))
        sns.violinplot(x='Placement_Status', y='Assignment_Score', data=df, palette="crest", ax=ax_v)
        ax_v.set_facecolor('#23272F')
        fig_v.patch.set_facecolor('#23272F')
        st.pyplot(fig_v, use_container_width=True)
    st.markdown('<span style="color:#A3A6B1; font-size:1.1rem; font-weight:600;">Attendance vs Placement</span>', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(7,3.5))
    sns.boxplot(x='Placement_Status', y='Attendance', data=df, palette="rocket", ax=ax3)
    ax3.set_facecolor('#23272F')
    fig3.patch.set_facecolor('#23272F')
    st.pyplot(fig3, use_container_width=True)


# --- At-Risk Students ---
def at_risk_students(df, model, le_domain, le_status):
    st.markdown("""
    <div style='margin-top:2rem; margin-bottom: 1.5rem;'></div>
    <h2 style='color:#F5F6FA; font-weight:700; letter-spacing:0.5px; margin-bottom:0.5rem;'>At-Risk Students</h2>
    """, unsafe_allow_html=True)
    # Use all model features for prediction
    feature_cols = ['Attendance', 'Assignment_Score', 'Domain_enc', 'Project_Count', 'Certifications', 'Hackathon_Participation', 'Soft_Skills_Score', 'Interview_Score']
    X = df[feature_cols].copy()
    proba = model.predict_proba(X)
    risk = np.max(proba, axis=1)
    risk_level = pd.cut(risk, bins=[-0.01,0.4,0.7,1.01], labels=['High','Medium','Low'])
    df_risk = df.copy()
    df_risk['Risk'] = risk_level
    high_risk = df_risk[df_risk['Risk']=='High']
    st.markdown('<div style="background:#f8f9fa; border-radius:12px; padding:1.5rem; margin-bottom:1.5rem;">', unsafe_allow_html=True)
    st.dataframe(high_risk[['Attendance','Assignment_Score','Domain','Project_Count','Certifications','Hackathon_Participation','Soft_Skills_Score','Interview_Score','Placement_Status','Risk']], use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<span style="color:#495057; font-size:0.95rem;">High risk = model confidence < 0.4</span>', unsafe_allow_html=True)
    st.markdown("""
    <div style='margin-top:2rem; margin-bottom: 1.5rem;'></div>
    <h2 style='color:#22223b; font-weight:700; letter-spacing:0.5px; margin-bottom:0.5rem;'>Visual Insights</h2>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<span style="color:#495057; font-size:1.1rem; font-weight:600;">Placement Rate by Domain</span>', unsafe_allow_html=True)
        domain_placement = df.groupby('Domain')['Placement_Status'].apply(lambda x: (x.isin(['Job','Intern']).mean())*100)
        import plotly.express as px
        fig1 = px.bar(x=domain_placement.index, y=domain_placement.values, labels={'x':'Domain','y':'Placement Rate (%)'}, color=domain_placement.index, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig1.update_layout(template='simple_white', showlegend=False, title=None)
        st.plotly_chart(fig1, use_container_width=True)
        # Pie chart for placement status by domain
        st.markdown('<span style="color:#495057; font-size:1.1rem; font-weight:600;">Placement Status by Domain (Pie)</span>', unsafe_allow_html=True)
        pie_df = df.groupby(['Domain','Placement_Status']).size().reset_index(name='Count')
        fig_pie = px.pie(pie_df, values='Count', names='Placement_Status', color='Placement_Status',
                        title='', hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(template='simple_white')
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.markdown('<span style="color:#495057; font-size:1.1rem; font-weight:600;">Placement Status Distribution</span>', unsafe_allow_html=True)
        status_counts = df['Placement_Status'].value_counts()
        fig2 = px.bar(x=status_counts.index, y=status_counts.values, labels={'x':'Placement Status','y':'Count'}, color=status_counts.index, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(template='simple_white', showlegend=False, title=None)
        st.plotly_chart(fig2, use_container_width=True)
        # Violin plot for assignment score by placement
        st.markdown('<span style="color:#495057; font-size:1.1rem; font-weight:600;">Assignment Score by Placement (Violin)</span>', unsafe_allow_html=True)
        fig_v = px.violin(df, x='Placement_Status', y='Assignment_Score', color='Placement_Status', box=True, points='all', color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_v.update_layout(template='simple_white', showlegend=False)
        st.plotly_chart(fig_v, use_container_width=True)
    st.markdown('<span style="color:#495057; font-size:1.1rem; font-weight:600;">Attendance vs Placement</span>', unsafe_allow_html=True)
    fig3 = px.box(df, x='Placement_Status', y='Attendance', color='Placement_Status', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig3.update_layout(template='simple_white', showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
# --- Actionable Strategies ---
def actionable_strategies():
    st.markdown("""
    <div style='margin-top:2rem; margin-bottom: 1.5rem;'></div>
    <h2 style='color:#F5F6FA; font-weight:700; letter-spacing:0.5px; margin-bottom:0.5rem;'>Actionable Strategies</h2>
    """, unsafe_allow_html=True)
    st.markdown('''
    <ul style="color:#A3A6B1; font-size:1.1rem;">
    <li><b>Encourage high attendance (&gt;80%)</b></li>
    <li><b>Provide extra assignment support for low-score students</b></li>
    <li><b>Focus training on low-placement domains</b></li>
    <li><b>Monitor at-risk students weekly</b></li>
    </ul>
    ''', unsafe_allow_html=True)


# --- Main App ---
def main():
    st.set_page_config(page_title='Student Placement Analysis', layout='wide', page_icon='🎓', initial_sidebar_state='expanded')
    st.markdown("""
    <style>
    html, body, [class*="css"]  {
        background-color: #18191A !important;
        color: #E4E6EB !important;
    }
    .stApp { background-color: #18191A; }
    .block-container { padding-top: 2.5rem; padding-bottom: 2.5rem; }
    .stMetric { background: #23272F; border-radius: 16px; padding: 1.5rem 0.7rem; margin-bottom: 1.2rem; box-shadow: 0 2px 8px #0002; }
    .stDataFrame { background: #23272F; border-radius: 14px; }
    .stButton>button { background: linear-gradient(90deg,#00C896,#00B4D8); color: #18191A; border-radius: 10px; font-weight: 700; font-size:1.1rem; padding:0.7rem 2.2rem; }
    .stSlider>div>div { color: #00C896; }
    .stRadio>div>label { color: #A3A6B1; }
    .stSidebar { background: #23272F; }
    .stForm { background: #23272F; border-radius: 14px; padding: 2rem; box-shadow: 0 2px 8px #0002; }
    h2 { font-size: 2rem !important; }
    /* Branding */
    .branding-bar { background: linear-gradient(90deg,#00C896,#00B4D8); height: 6px; border-radius: 8px; margin-bottom: 1.2rem; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="branding-bar"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex; align-items:center; gap:0.7rem; margin-bottom:1.5rem;'>
        <span style='font-size:2.2rem;'>🎓</span>
        <span style='font-size:2.1rem; font-weight:700; color:#F5F6FA; letter-spacing:0.5px;'>Student Placement Analysis Dashboard</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex; align-items:center; gap:0.7rem; margin-bottom:1.5rem;'>
        <span style='font-size:2.2rem;'>🎓</span>
        <span style='font-size:2.1rem; font-weight:700; color:#F5F6FA; letter-spacing:0.5px;'>Student Placement Analysis Dashboard</span>
    </div>
    """, unsafe_allow_html=True)
    df = load_data()
    model = load_model()
    le_domain = joblib.load('le_domain.pkl')
    le_status = joblib.load('le_status.pkl')
    menu = [
        'KPI Dashboard',
        'Visual Insights',
        'At-Risk Students',
        'Prediction Tool',
        'Actionable Strategies'
    ]
    with st.sidebar:
        st.markdown('<span style="font-size:1.3rem; color:#F5F6FA; font-weight:600;">Navigation</span>', unsafe_allow_html=True)
        choice = st.radio('', menu)
        st.markdown('<div style="margin-top:2rem; color:#A3A6B1; font-size:0.95rem;">Powered by Streamlit & Copilot</div>', unsafe_allow_html=True)
    if choice == 'KPI Dashboard':
        kpi_dashboard(df)
    elif choice == 'Visual Insights':
        visual_insights(df)
    elif choice == 'At-Risk Students':
        at_risk_students(df, model, le_domain, le_status)
    elif choice == 'Prediction Tool':
        prediction_tool(model, le_domain, le_status)
    elif choice == 'Actionable Strategies':
        actionable_strategies()

if __name__ == '__main__':
    main()
