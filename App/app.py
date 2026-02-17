"""
Smart Waste Management Dashboard
================================
Interactive Streamlit dashboard for the waste management ML system.
Provides predictions, priority scoring, and model insights.

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Smart Waste Management System",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .priority-critical {
        background-color: #ff4444;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    .priority-high {
        background-color: #ffaa00;
        color: black;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    .priority-medium {
        background-color: #00C851;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    .priority-low {
        background-color: #33b5e5;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data
def load_data():
    """Load the features dataset."""
    # Try full file first, then sample
    data_path = Path(__file__).parent.parent / "data" / "processed" / "features.csv"
    sample_path = Path(__file__).parent.parent / "data" / "processed" / "features_sample.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        return df
    elif sample_path.exists():
        df = pd.read_csv(sample_path)
        return df
    return None

@st.cache_data
def load_bins_data():
    """Load bins dataset."""
    bins_path = Path(__file__).parent.parent / "data" / "raw" / "bins.csv"
    if bins_path.exists():
        return pd.read_csv(bins_path)
    return None

@st.cache_resource
def load_models():
    """Load trained models from the latest bundle."""
    models_dir = Path(__file__).parent.parent / "models"
    
    if not models_dir.exists():
        return None, None, None, None
    
    # Find latest model bundle
    bundles = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("waste_management_models")]
    if not bundles:
        return None, None, None, None
    
    latest_bundle = max(bundles, key=lambda x: x.stat().st_mtime)
    
    # Load models
    fill_model = None
    waste_model = None
    preprocessor = None
    config = None
    
    try:
        fill_model_path = latest_bundle / "fill_level_model.joblib"
        waste_model_path = latest_bundle / "waste_type_model.joblib"
        preprocessor_path = latest_bundle / "preprocessor.joblib"
        config_path = latest_bundle / "feature_config.json"
        
        if fill_model_path.exists():
            fill_model = joblib.load(fill_model_path)
        if waste_model_path.exists():
            waste_model = joblib.load(waste_model_path)
        if preprocessor_path.exists():
            preprocessor = joblib.load(preprocessor_path)
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
    except Exception as e:
        st.warning(f"Could not load models: {e}")
        return None, None, None, None
    
    return fill_model, waste_model, preprocessor, config

@st.cache_data
def load_metrics():
    """Load model metrics."""
    models_dir = Path(__file__).parent.parent / "models"
    
    if not models_dir.exists():
        return None, None
    
    bundles = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("waste_management_models")]
    if not bundles:
        return None, None
    
    latest_bundle = max(bundles, key=lambda x: x.stat().st_mtime)
    
    fill_metrics = None
    waste_metrics = None
    
    fill_metrics_path = latest_bundle / "fill_level_metrics.json"
    waste_metrics_path = latest_bundle / "waste_type_metrics.json"
    
    if fill_metrics_path.exists():
        with open(fill_metrics_path) as f:
            fill_metrics = json.load(f)
    if waste_metrics_path.exists():
        with open(waste_metrics_path) as f:
            waste_metrics = json.load(f)
    
    return fill_metrics, waste_metrics

# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_fill_level(model, preprocessor, features_dict, config):
    """Make fill level prediction."""
    if model is None or preprocessor is None:
        return None, None
    
    # Get required features
    feature_cols = config.get('fill_level_features', [])
    
    # Build feature vector
    X = pd.DataFrame([features_dict])
    
    # Encode categorical columns
    cat_cols = [c for c in config.get('categorical_cols', []) if c in X.columns]
    for col in cat_cols:
        if col in preprocessor.label_encoders:
            le = preprocessor.label_encoders[col]
            val = X[col].iloc[0]
            if val in le.classes_:
                X[col] = le.transform([val])[0]
            else:
                X[col] = 0
    
    # Select available features
    available = [f for f in feature_cols if f in X.columns]
    X = X[available].fillna(0)
    
    # Predict
    pred = model.predict(X.values)[0]
    
    # Get probabilities if available
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X.values)[0]
    
    # Map prediction to label
    labels = ['critical', 'high', 'low', 'medium']
    pred_label = labels[int(pred)] if int(pred) < len(labels) else 'unknown'
    
    return pred_label, proba

def predict_waste_type(model, preprocessor, features_dict, config):
    """Make waste type prediction."""
    if model is None or preprocessor is None:
        return None, None
    
    feature_cols = config.get('waste_type_features', [])
    
    X = pd.DataFrame([features_dict])
    
    cat_cols = [c for c in config.get('categorical_cols', []) if c in X.columns]
    for col in cat_cols:
        if col in preprocessor.label_encoders:
            le = preprocessor.label_encoders[col]
            val = X[col].iloc[0]
            if val in le.classes_:
                X[col] = le.transform([val])[0]
            else:
                X[col] = 0
    
    available = [f for f in feature_cols if f in X.columns]
    X = X[available].fillna(0)
    
    pred = model.predict(X.values)[0]
    
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X.values)[0]
    
    labels = ['construction', 'e_waste', 'general', 'hazardous', 'organic', 'recyclable']
    pred_label = labels[int(pred)] if int(pred) < len(labels) else 'unknown'
    
    return pred_label, proba

def calculate_priority_score(fill_level, waste_type, population, overflow=0, odor=0):
    """Calculate priority score."""
    score = 0
    
    # Fill level contribution
    fill_scores = {'low': 10, 'medium': 30, 'high': 60, 'critical': 90}
    score += fill_scores.get(fill_level, 30)
    
    # Waste type contribution
    if waste_type == 'hazardous':
        score += 30
    elif waste_type == 'e_waste':
        score += 15
    elif waste_type == 'organic':
        score += 5
    
    # Population factor
    score += min(population / 200, 10)
    
    # Incident flags
    if overflow:
        score += 20
    if odor:
        score += 10
    
    return min(score, 100)

def get_priority_level(score):
    """Get priority level from score."""
    if score >= 80:
        return 'critical'
    elif score >= 60:
        return 'high'
    elif score >= 40:
        return 'medium'
    else:
        return 'low'

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_fill_level_distribution(df):
    """Create fill level distribution chart."""
    if 'fill_level_category' not in df.columns:
        return None
    
    counts = df['fill_level_category'].value_counts()
    
    colors = {
        'low': '#33b5e5',
        'medium': '#00C851',
        'high': '#ffaa00',
        'critical': '#ff4444'
    }
    
    fig = px.pie(
        values=counts.values,
        names=counts.index,
        color=counts.index,
        color_discrete_map=colors,
        hole=0.4
    )
    fig.update_layout(
        title="Fill Level Distribution",
        showlegend=True,
        height=350
    )
    return fig

def create_waste_type_distribution(df):
    """Create waste type distribution chart."""
    if 'waste_type_primary' not in df.columns:
        return None
    
    counts = df['waste_type_primary'].value_counts()
    
    fig = px.bar(
        x=counts.index,
        y=counts.values,
        color=counts.index,
        labels={'x': 'Waste Type', 'y': 'Count'}
    )
    fig.update_layout(
        title="Waste Type Distribution",
        showlegend=False,
        height=350
    )
    return fig

def create_feature_importance_chart(metrics, title):
    """Create feature importance bar chart."""
    if metrics is None or 'feature_importances' not in metrics:
        return None
    
    importances = metrics['feature_importances']
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    
    fig = px.bar(
        x=[v for _, v in sorted_features],
        y=[k for k, _ in sorted_features],
        orientation='h',
        labels={'x': 'Importance', 'y': 'Feature'}
    )
    fig.update_layout(
        title=title,
        yaxis={'categoryorder': 'total ascending'},
        height=400
    )
    return fig

def create_location_heatmap(df):
    """Create location type vs fill level heatmap."""
    if 'location_type' not in df.columns or 'fill_level_category' not in df.columns:
        return None
    
    cross_tab = pd.crosstab(df['location_type'], df['fill_level_category'], normalize='index') * 100
    
    fig = px.imshow(
        cross_tab,
        labels=dict(x="Fill Level", y="Location Type", color="Percentage"),
        color_continuous_scale='RdYlGn_r',
        aspect='auto'
    )
    fig.update_layout(
        title="Fill Level by Location Type (%)",
        height=400
    )
    return fig

def create_time_series(df):
    """Create time series of collections."""
    if 'collection_date' not in df.columns:
        return None
    
    df_ts = df.copy()
    df_ts['collection_date'] = pd.to_datetime(df_ts['collection_date'])
    daily = df_ts.groupby('collection_date').agg({
        'fill_level_percent': 'mean',
        'bin_id': 'count'
    }).reset_index()
    daily.columns = ['Date', 'Avg Fill Level', 'Collections']
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=daily['Date'], y=daily['Avg Fill Level'], name="Avg Fill Level", line=dict(color='#667eea')),
        secondary_y=False
    )
    fig.add_trace(
        go.Bar(x=daily['Date'], y=daily['Collections'], name="Collections", opacity=0.3),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Collection Trends Over Time",
        height=400
    )
    fig.update_yaxes(title_text="Avg Fill Level (%)", secondary_y=False)
    fig.update_yaxes(title_text="Number of Collections", secondary_y=True)
    
    return fig

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🗑️ Smart Waste Management System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Collection Optimization for Ghanaian Municipalities</p>', unsafe_allow_html=True)
    
    # Load data and models
    df = load_data()
    bins_df = load_bins_data()
    fill_model, waste_model, preprocessor, config = load_models()
    fill_metrics, waste_metrics = load_metrics()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/waste.png", width=80)
        st.markdown("### Navigation")
        st.markdown("---")
        
        if fill_model is not None:
            st.success("✅ Models Loaded")
        else:
            st.warning("⚠️ No models found. Run training first.")
        
        if df is not None:
            st.info(f"📊 {len(df):,} records loaded")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard provides:
        - Real-time bin status monitoring
        - AI-powered fill level predictions
        - Collection priority scoring
        - Model performance insights
        """)
        
        st.markdown("---")
        st.markdown("**Built by:** David Quayefio")
        st.markdown("**Tech:** Python, Scikit-learn, XGBoost, Streamlit")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🔮 Predictions", 
        "🚨 Priority Queue",
        "📈 Model Performance",
        "📋 Data Explorer"
    ])
    
    # ==========================================================================
    # TAB 1: DASHBOARD
    # ==========================================================================
    with tab1:
        st.markdown("### Overview")
        
        if df is not None:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_bins = df['bin_id'].nunique() if 'bin_id' in df.columns else 0
                st.metric("Total Bins", f"{total_bins:,}")
            
            with col2:
                critical_count = len(df[df['fill_level_category'] == 'critical']) if 'fill_level_category' in df.columns else 0
                st.metric("Critical Bins", f"{critical_count:,}", delta="Needs attention", delta_color="inverse")
            
            with col3:
                avg_fill = df['fill_level_percent'].mean() if 'fill_level_percent' in df.columns else 0
                st.metric("Avg Fill Level", f"{avg_fill:.1f}%")
            
            with col4:
                hazardous_count = len(df[df['waste_type_primary'] == 'hazardous']) if 'waste_type_primary' in df.columns else 0
                st.metric("Hazardous Waste", f"{hazardous_count:,}")
            
            st.markdown("---")
            
            # Charts row 1
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_fill_level_distribution(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_waste_type_distribution(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Charts row 2
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_location_heatmap(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = create_time_series(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available. Please ensure features.csv exists in data/processed/")
    
    # ==========================================================================
    # TAB 2: PREDICTIONS
    # ==========================================================================
    with tab2:
        st.markdown("### 🔮 Make Predictions")
        st.markdown("Enter bin characteristics to predict fill level, waste type, and collection priority.")
        
        if fill_model is None or waste_model is None:
            st.error("Models not loaded. Please run the training pipeline first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Bin Characteristics")
                
                location_type = st.selectbox(
                    "Location Type",
                    ['residential', 'commercial', 'market', 'industrial', 'institutional', 'hospitality']
                )
                
                district = st.selectbox(
                    "District",
                    ['Accra Central', 'Tema', 'Madina', 'Teshie-Nungua', 'Ashaiman', 
                     'Kasoa', 'Adenta', 'La', 'Osu', 'Dansoman', 'Achimota', 
                     'Spintex', 'East Legon', 'Airport Residential', 'Kaneshie']
                )
                
                capacity = st.selectbox("Capacity (liters)", [120, 240, 660, 1100])
                
                population = st.slider("Nearby Population", 50, 5000, 500)
                
                has_lid = st.checkbox("Has Lid", value=True)
            
            with col2:
                st.markdown("#### Collection Context")
                
                days_since = st.slider("Days Since Last Collection", 1, 14, 3)
                
                fill_rate_avg = st.slider("7-Day Avg Fill Rate (%/day)", 1.0, 30.0, 10.0)
                
                prev_fill = st.slider("Previous Fill Level (%)", 0, 100, 40)
                
                is_weekend = st.checkbox("Is Weekend")
                is_holiday = st.checkbox("Is Holiday")
                is_festival = st.checkbox("Is Festival Period")
                
                overflow = st.checkbox("Overflow Reported")
                odor = st.checkbox("Odor Complaint")
            
            st.markdown("---")
            
            if st.button("🔮 Generate Prediction", type="primary", use_container_width=True):
                # Build features dict
                features = {
                    'location_type': location_type,
                    'district': district,
                    'capacity_liters': capacity,
                    'nearby_population': population,
                    'has_lid': 1 if has_lid else 0,
                    'days_since_last_collection': days_since,
                    'fill_rate_7day_avg': fill_rate_avg,
                    'prev_fill_level': prev_fill,
                    'is_weekend': 1 if is_weekend else 0,
                    'is_holiday': 1 if is_holiday else 0,
                    'is_festival_period': 1 if is_festival else 0,
                    'overflow_reported': 1 if overflow else 0,
                    'odor_complaint': 1 if odor else 0,
                    'month': datetime.now().month,
                    'temperature_c': 30,
                    'rainfall_mm': 5,
                    'road_accessibility': 'easy',
                    'fill_level_percent': prev_fill + (days_since * fill_rate_avg),
                    'fill_rate_per_day': fill_rate_avg,
                    'waste_weight_kg': capacity * 0.3,
                }
                
                # Make predictions
                fill_pred, fill_proba = predict_fill_level(fill_model, preprocessor, features, config)
                waste_pred, waste_proba = predict_waste_type(waste_model, preprocessor, features, config)
                
                # Calculate priority
                priority_score = calculate_priority_score(
                    fill_pred, waste_pred, population, overflow, odor
                )
                priority_level = get_priority_level(priority_score)
                
                # Display results
                st.markdown("### 📊 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### Fill Level")
                    color_map = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
                    st.markdown(f"### {color_map.get(fill_pred, '⚪')} {fill_pred.upper()}")
                    if fill_proba is not None:
                        st.progress(float(max(fill_proba)))
                        st.caption(f"Confidence: {max(fill_proba)*100:.1f}%")
                
                with col2:
                    st.markdown("#### Waste Type")
                    waste_icons = {
                        'organic': '🥬', 'recyclable': '♻️', 'general': '🗑️',
                        'hazardous': '☢️', 'e_waste': '💻', 'construction': '🧱'
                    }
                    st.markdown(f"### {waste_icons.get(waste_pred, '❓')} {waste_pred.upper()}")
                    if waste_proba is not None:
                        st.progress(float(max(waste_proba)))
                        st.caption(f"Confidence: {max(waste_proba)*100:.1f}%")
                
                with col3:
                    st.markdown("#### Priority Score")
                    priority_colors = {'low': 'blue', 'medium': 'green', 'high': 'orange', 'critical': 'red'}
                    st.markdown(f"### {priority_score:.0f}/100")
                    st.markdown(f"**Level:** :{priority_colors[priority_level]}[{priority_level.upper()}]")
                
                # Recommended action
                st.markdown("---")
                st.markdown("#### 📋 Recommended Action")
                
                if priority_level == 'critical':
                    st.error("🚨 **URGENT:** Dispatch collection vehicle immediately. High risk of overflow/health hazard.")
                elif priority_level == 'high':
                    st.warning("⚠️ **HIGH PRIORITY:** Schedule collection within 24 hours.")
                elif priority_level == 'medium':
                    st.info("ℹ️ **STANDARD:** Include in next scheduled collection route.")
                else:
                    st.success("✅ **LOW PRIORITY:** Monitor status. No immediate action required.")
                
                # Risk factors
                if waste_pred == 'hazardous':
                    st.error("☢️ **HAZARDOUS WASTE DETECTED** - Ensure proper handling protocols!")
    
    # ==========================================================================
    # TAB 3: PRIORITY QUEUE
    # ==========================================================================
    with tab3:
        st.markdown("### 🚨 Collection Priority Queue")
        st.markdown("Bins ranked by urgency for collection scheduling.")
        
        if df is not None and fill_model is not None:
            # Sample recent data and calculate priorities
            sample_df = df.sample(min(100, len(df)), random_state=42).copy()
            
            # Calculate priority for each bin
            priorities = []
            for _, row in sample_df.iterrows():
                fill_level = row.get('fill_level_category', 'medium')
                waste_type = row.get('waste_type_primary', 'general')
                population = row.get('nearby_population', 500)
                overflow = row.get('overflow_reported', 0)
                odor = row.get('odor_complaint', 0)
                
                score = calculate_priority_score(fill_level, waste_type, population, overflow, odor)
                priorities.append({
                    'bin_id': row.get('bin_id', 'Unknown'),
                    'location_type': row.get('location_type', 'Unknown'),
                    'fill_level': fill_level,
                    'waste_type': waste_type,
                    'priority_score': score,
                    'priority_level': get_priority_level(score),
                    'days_since_collection': row.get('days_since_last_collection', 0)
                })
            
            priority_df = pd.DataFrame(priorities)
            priority_df = priority_df.sort_values('priority_score', ascending=False)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                critical = len(priority_df[priority_df['priority_level'] == 'critical'])
                st.metric("🔴 Critical", critical)
            with col2:
                high = len(priority_df[priority_df['priority_level'] == 'high'])
                st.metric("🟠 High", high)
            with col3:
                medium = len(priority_df[priority_df['priority_level'] == 'medium'])
                st.metric("🟡 Medium", medium)
            with col4:
                low = len(priority_df[priority_df['priority_level'] == 'low'])
                st.metric("🟢 Low", low)
            
            st.markdown("---")
            
            # Filter
            filter_level = st.multiselect(
                "Filter by Priority Level",
                ['critical', 'high', 'medium', 'low'],
                default=['critical', 'high']
            )
            
            filtered_df = priority_df[priority_df['priority_level'].isin(filter_level)]
            
            # Display table
            st.dataframe(
                filtered_df.style.apply(
                    lambda x: ['background-color: #ffcccc' if v == 'critical' 
                              else 'background-color: #ffe6cc' if v == 'high'
                              else 'background-color: #ffffcc' if v == 'medium'
                              else 'background-color: #ccffcc' for v in x],
                    subset=['priority_level']
                ),
                use_container_width=True,
                height=400
            )
            
            # Export option
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download Priority List",
                csv,
                "priority_queue.csv",
                "text/csv"
            )
        else:
            st.warning("Load data and models to view priority queue.")
    
    # ==========================================================================
    # TAB 4: MODEL PERFORMANCE
    # ==========================================================================
    with tab4:
        st.markdown("### 📈 Model Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Fill Level Model")
            if fill_metrics:
                st.metric("Model", fill_metrics.get('model_name', 'Unknown'))
                
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Accuracy", f"{fill_metrics.get('accuracy', 0)*100:.1f}%")
                with m2:
                    st.metric("F1-Macro", f"{fill_metrics.get('f1_macro', 0):.3f}")
                
                m3, m4 = st.columns(2)
                with m3:
                    st.metric("CV Mean", f"{fill_metrics.get('cv_mean', 0):.3f}")
                with m4:
                    st.metric("CV Std", f"{fill_metrics.get('cv_std', 0):.4f}")
                
                # Feature importance chart
                fig = create_feature_importance_chart(fill_metrics, "Fill Level - Feature Importance")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No fill level metrics available.")
        
        with col2:
            st.markdown("#### Waste Type Model")
            if waste_metrics:
                st.metric("Model", waste_metrics.get('model_name', 'Unknown'))
                
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Accuracy", f"{waste_metrics.get('accuracy', 0)*100:.1f}%")
                with m2:
                    st.metric("F1-Macro", f"{waste_metrics.get('f1_macro', 0):.3f}")
                
                m3, m4 = st.columns(2)
                with m3:
                    st.metric("CV Mean", f"{waste_metrics.get('cv_mean', 0):.3f}")
                with m4:
                    st.metric("CV Std", f"{waste_metrics.get('cv_std', 0):.4f}")
                
                # Feature importance chart
                fig = create_feature_importance_chart(waste_metrics, "Waste Type - Feature Importance")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Safety warning
                st.warning("⚠️ **Note:** Hazardous waste recall is below 85% threshold. Use with caution for safety-critical decisions.")
            else:
                st.warning("No waste type metrics available.")
    
    # ==========================================================================
    # TAB 5: DATA EXPLORER
    # ==========================================================================
    with tab5:
        st.markdown("### 📋 Data Explorer")
        
        if df is not None:
            st.markdown(f"**Dataset:** {len(df):,} records × {len(df.columns)} columns")
            
            # Column selector
            cols_to_show = st.multiselect(
                "Select columns to display",
                df.columns.tolist(),
                default=['bin_id', 'location_type', 'fill_level_percent', 'fill_level_category', 
                        'waste_type_primary', 'days_since_last_collection'][:min(6, len(df.columns))]
            )
            
            if cols_to_show:
                st.dataframe(df[cols_to_show].head(100), use_container_width=True)
            
            # Statistics
            st.markdown("---")
            st.markdown("#### Summary Statistics")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        else:
            st.warning("No data available.")

# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()