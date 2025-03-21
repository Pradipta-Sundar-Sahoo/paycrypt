import math
from sklearn.neighbors import LocalOutlierFactor
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import datetime
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import warnings
from typing import Dict,List,Any

from utils import AutomatedInsightsGenerator, EnhancedLaneOptimizationSystem
warnings.filterwarnings('ignore')
from openai import OpenAI
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns

# =============================================================================
# LOAD DATA AND MODELS
# =============================================================================
@st.cache_data
def load_data(): # type: ignore
    df = pd.read_csv('data/preprocessed_tollplaza_data.csv')
    df['initiated_time'] = pd.to_datetime(df['initiated_time'])
    df['time_interval'] = pd.to_datetime(df['time_interval'])
    return df

df = load_data()

# Load models (if available)
try:
    traffic_model = joblib.load('models/traffic_prediction_model.pkl')
except Exception as e:
    traffic_model = None
    st.warning("Traffic prediction model not loaded.")

try:
    anomaly_model, anomaly_scaler = joblib.load('models/traffic_anomaly_models.pkl')
except Exception as e:
    anomaly_model, anomaly_scaler = None, None
    st.warning("Anomaly detection model not loaded.")

try:
    traffic_models, traffic_scaler = joblib.load('models/traffic_anomaly_models.pkl')
    
    # toll_models = {
    #     # 'isolation_forest': joblib.load('models/toll_isolation_forest.pkl'),
    #     'ocsvm': joblib.load('models/toll_oneclass_svm.pkl'),
    #     # 'kmeans': joblib.load('models/toll_kmeans.pkl'),
    #     'xgb': joblib.load('models/toll_xgboost.pkl'),
    #     'pca': joblib.load('models/toll_pca.pkl')
    # }
    # toll_scaler = joblib.load('models/toll_scaler.pkl')
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    traffic_models, traffic_scaler = None, None
    toll_models, toll_scaler = None, None

    
try:
    vc_model = joblib.load('models/vehicle_class_prediction_model.pkl')
    vc_model_loaded = True
except:
    vc_model_loaded = False
    st.warning("Vehicle class model not loaded.")

try:
    skip_model, skip_scaler = joblib.load('models/toll_skipping_models.pkl')
    skip_model_loaded = True
except:
    skip_model_loaded = False
    st.warning("Toll skipping model not loaded.")

try:
    vehicle_classifier=joblib.load('models/vehicle_behavior_classifier.pkl')
except:
    st.warning("Vehicle behavior classifier not loaded.")

# =============================================================================
# LANE OPTIMIZATION SYSTEM
# =============================================================================

# Check for OpenAI key (implement your own key handling)
openai_key = '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'
if not openai_key:
    print("No OpenAI API key found. Using fallback insight generation.")

# Initialize insights generator
insights_generator = AutomatedInsightsGenerator(df, openai_key)

# =============================================================================
# TOLL SKIPPING DETECTION
# =============================================================================
def debug_dataframe(df):
    df.columns = df.columns.str.strip()
    return df


# Initialize our systems
# lane_optimizer = LaneOptimizationSystem(df)
openai_key = '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'
lane_optimizer = EnhancedLaneOptimizationSystem(df, openai_api_key=openai_key)

# insights_generator = AutomatedInsightsGenerator(df)
# toll_skipping_detector = TollSkippingDetection(df)

# =============================================================================
# STREAMLIT APP WITH TABS
# =============================================================================
st.title("Toll Plaza Smart Solutions Implementation")

tab0, tab1, tab4 ,tab_smart_solutions= st.tabs([
    'Data Explorations',
    "Traffic Prediction",  
    "Toll Skipping, Vehicle Behavior",
    "Smart Solutions"
])

# (Assuming your other tabs are already implemented in previous code)
# We add the Smart Solutions tab here:
with tab_smart_solutions:
    st.header("Smart Solutions")
    sub_tab_lane, sub_tab_insights= st.tabs([
        "Lane Optimization",
        "Automated Insights"
    ])

    with sub_tab_lane:
        st.subheader("Lane Optimization System")
        
        # Create two columns for controls
        col1, col2 = st.columns(2)
        
        with col1:
            plaza_list = df['merchant_name'].unique().tolist()
            selected_plaza = st.selectbox("Select a Plaza", plaza_list)
            selected_hour = st.slider("Select Hour (0-23)", 0, 23, 8)
        
        with col2:
            expected_traffic = st.number_input("Expected Traffic (optional)", min_value=0, value=0)
            if expected_traffic == 0:
                expected_traffic = None
            use_ai = st.checkbox("Use AI-Enhanced Recommendations", value=True)
        
        if st.button("Get Lane Recommendations"):
            with st.spinner("Generating recommendations..."):
                recommendations = lane_optimizer.get_lane_recommendations(
                    selected_plaza, selected_hour, expected_traffic, use_ai=use_ai
                )
                
                if "error" in recommendations:
                    st.error(recommendations["error"])
                else:
                    # Create tabs for different views
                    rec_tab1, rec_tab2, rec_tab3 = st.tabs(["Recommendations", "Lane Efficiency", "Vehicle Distribution"])
                    
                    with rec_tab1:
                        # Show AI badge if enhanced
                        if recommendations.get("ai_enhanced", False):
                            st.success("✨ AI-Enhanced Recommendations")
                        
                        # Display recommended lanes in a nice table
                        st.subheader("Recommended Lanes")
                        lanes_df = pd.DataFrame(recommendations["recommended_lanes"])
                        st.dataframe(lanes_df, use_container_width=True)
                        
                        # Summary metrics
                        st.subheader("Summary")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Traffic", f"{recommendations['expected_traffic']:,}")
                        col2.metric("Lanes Needed", recommendations["lanes_needed"])
                        col3.metric("Avg Processing Time", f"{recommendations['historical_average_processing_time']:.2f}s")
                        
                        # Show optimization notes if available
                        if "optimization_notes" in recommendations and recommendations["optimization_notes"]:
                            st.subheader("Optimization Notes")
                            st.write(recommendations["optimization_notes"])
                        
                        # Full JSON for developers
                        with st.expander("View Full Recommendation Data"):
                            st.json(recommendations)
                    
                    with rec_tab2:
                        st.subheader("Lane Efficiency Analysis")
                        
                        # Get lane efficiency data
                        plaza_eff = lane_optimizer.lane_efficiency[lane_optimizer.lane_efficiency['plaza'] == selected_plaza]
                        
                        # Create two columns for different charts
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Efficiency score chart
                            fig1, ax1 = plt.subplots(figsize=(8, 4))
                            bars = ax1.bar(plaza_eff['lane'], plaza_eff['efficiency_score'])
                            
                            # Safely extract recommended lane IDs with error handling
                            recommended_lane_ids = []
                            for lane in recommendations.get("recommended_lanes", []):
                                if isinstance(lane, dict) and "lane" in lane:
                                    recommended_lane_ids.append(lane["lane"])
                                elif isinstance(lane, dict) and any(k in lane for k in ["id", "lane_id", "name"]):
                                    # Try alternative keys if "lane" is not available
                                    for key in ["id", "lane_id", "name"]:
                                        if key in lane:
                                            recommended_lane_ids.append(lane[key])
                                            break
                            
                            # Highlight recommended lanes
                            for i, lane in enumerate(plaza_eff['lane']):
                                if lane in recommended_lane_ids:
                                    bars[i].set_color('green')
                            
                            ax1.set_xlabel("Lane")
                            ax1.set_ylabel("Efficiency Score (lower is better)")
                            ax1.set_title("Lane Efficiency Scores")
                            st.pyplot(fig1)
                        
                        with col2:
                            # Processing time chart
                            fig2, ax2 = plt.subplots(figsize=(8, 4))
                            ax2.bar(plaza_eff['lane'], plaza_eff['avg_processing_time'])
                            ax2.set_xlabel("Lane")
                            ax2.set_ylabel("Avg Processing Time (sec)")
                            ax2.set_title("Lane Processing Times")
                            st.pyplot(fig2)
                        
                        # Reliability score
                        if 'reliability_score' in plaza_eff.columns:
                            fig3, ax3 = plt.subplots(figsize=(10, 4))
                            ax3.bar(plaza_eff['lane'], plaza_eff['reliability_score'])
                            ax3.set_xlabel("Lane")
                            ax3.set_ylabel("Reliability Score (higher is better)")
                            ax3.set_title("Lane Reliability Scores")
                            st.pyplot(fig3)
                    
                    with rec_tab3:
                        st.subheader("Vehicle Distribution")
                        
                        # Convert vehicle distribution to DataFrame
                        veh_dist = pd.DataFrame({
                            'Vehicle Type': list(recommendations["vehicle_distribution"].keys()),
                            'Percentage': [v * 100 for v in recommendations["vehicle_distribution"].values()]
                        })
                        
                        # Plot vehicle distribution
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.pie(veh_dist['Percentage'], labels=veh_dist['Vehicle Type'], autopct='%1.1f%%')
                        ax.set_title("Vehicle Type Distribution")
                        st.pyplot(fig)
                        
                        # Show as table
                        st.dataframe(veh_dist, use_container_width=True)
        
        # Add option to get AI insights
        st.divider()
        # if st.button("Get AI Optimization Insights"):
        #     with st.spinner("Analyzing plaza operations..."):
        #         insights = lane_optimizer.get_optimization_insights(selected_plaza)
        #         print(insights)
        #         if "error" in insights:
        #             st.error(insights["error"])
        #         else:
        #             st.subheader("AI Optimization Insights")
                    
        #             # Key findings
        #             st.write("### Key Findings")
        #             for i, finding in enumerate(insights.get("key_findings", [])):
        #                 st.markdown(f"{i+1}. {finding}")
                    
        #             # Optimization opportunities with difficulty ratings
        #             st.write("### Optimization Opportunities")
        #             opportunities = insights.get("optimization_opportunities", [])
        #             difficulties = insights.get("implementation_difficulty", [])
                    
        #             # Create a table of opportunities and difficulties
        #             if opportunities and len(opportunities) == len(difficulties):
        #                 opp_data = []
        #                 for i in range(len(opportunities)):
        #                     opp_data.append({
        #                         "Opportunity": opportunities[i],
        #                         "Difficulty": difficulties[i]
        #                     })
        #                 opp_df = pd.DataFrame(opp_data)
        #                 st.dataframe(opp_df, use_container_width=True)
        #             else:
        #                 for opp in opportunities:
        #                     st.markdown(f"- {opp}")
                    
        #             # Projected benefits
        #             if "projected_benefits" in insights:
        #                 st.write("### Projected Benefits")
        #                 benefits = insights["projected_benefits"]
                        
        #                 cols = st.columns(len(benefits))
        #                 for i, (metric, value) in enumerate(benefits.items()):
        #                     cols[i].metric(metric.replace("_", " ").title(), value)
                    
        #             # Full insights
        #             with st.expander("View Full Insights Data"):
        #                 st.json(insights)
        if st.button("Get AI Optimization Insights"):
            with st.spinner("Analyzing plaza operations..."):
                insights = lane_optimizer.get_optimization_insights(selected_plaza)
                print(insights)
                
                if "error" in insights:
                    st.error(insights["error"])
                else:
                    st.subheader("AI Optimization Insights")
                    
                    # Key findings
                    st.write("### Key Findings")
                    for i, finding in enumerate(insights.get("key_findings", [])):
                        st.markdown(f"{i+1}. {finding}")
                    
                    # Optimization opportunities with difficulty ratings
                    st.write("### Optimization Opportunities")
                    opportunities = insights.get("optimization_opportunities", [])
                    
                    if opportunities:
                        opp_data = []
                        for opp in opportunities:
                            opp_data.append({
                                "Opportunity": opp["description"] if 'description' in opp else opp['recommendation'],  # Extract description
                                "Difficulty": opp["difficulty"] if 'difficulty' in opp else opp['implementation_difficulty']  # Extract difficulty directly from opportunity
                            })
                        
                        opp_df = pd.DataFrame(opp_data)
                        st.dataframe(opp_df, use_container_width=True)
                    else:
                        st.write("No optimization opportunities found.")
                    
                    # Projected benefits
                    if "projected_benefits" in insights:
                        st.write("### Projected Benefits")
                        benefits = insights["projected_benefits"]
                        
                        # Adjust number of columns to fit content
                        num_cols = min(len(benefits), 3)  # Max 3 columns to prevent overflow
                        cols = st.columns(num_cols)

                        for i, (metric, value) in enumerate(benefits.items()):
                            # Use st.write() for full text instead of st.metric()
                            with cols[i % num_cols]:  
                                st.write(f"**{metric.replace('_', ' ').title()}**")
                                st.write(value)
                                        
                    # Full insights
                    with st.expander("View Full Insights Data"):
                        st.json(insights)

    with sub_tab_insights:
        st.subheader("Automated Insights")
        if st.button("Generate Insights Report"):
            report = insights_generator.generate_insights_report()
            st.markdown(report['narrative'])
            st.json(report['summary'])

    # with sub_tab_toll:
    #     st.subheader("Toll Skipping Detection")
    #     if st.button("Detect Toll Skipping Incidents"):
    #         incidents = toll_skipping_detector.detect_potential_toll_skipping()
    #         if incidents:
    #             st.json(incidents)
    #         else:
    #             st.write("No potential toll skipping incidents detected.")


def load_data():
    file_path = "data/Bangalore_1day_NETC.csv"  
    df = pd.read_csv(file_path)
    df['fixed_time'] = df['initiated_time'].str[-5:]
    df['initiated_time'] = pd.to_datetime('2024-03-19 ' + df['fixed_time'], format='%Y-%m-%d %H:%M')
    df.drop(columns=['fixed_time'], inplace=True)
    
    df['hour'] = df['initiated_time'].dt.hour
    df['day_of_week'] = df['initiated_time'].dt.dayofweek
    
    scaler = MinMaxScaler()
    df['txn_amount_scaled'] = scaler.fit_transform(df[['txn_amount']])
    
    encoder = LabelEncoder()
    df['vehicle_class_code_enc'] = encoder.fit_transform(df['vehicle_class_code'])
    df['merchant_name_enc'] = encoder.fit_transform(df['merchant_name'])
    
    return df

df = load_data()


with tab0:
    st.subheader("Dataset Overview")
    st.write(df.head())
    
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    
    st.subheader("Traffic Volume by Hour")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='hour', data=df, palette='viridis', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Vehicle Class Distribution")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.countplot(y='vehicle_class_code', data=df, palette='coolwarm', ax=ax)
    st.pyplot(fig)
    
    st.subheader("Transaction Amount Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['txn_amount'], kde=True, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Traffic by Time of Day")
    fig, ax = plt.subplots(figsize=(12, 6))
    time_of_day_counts = df['hour'].value_counts().sort_index()
    sns.barplot(x=time_of_day_counts.index, y=time_of_day_counts.values, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Hourly Traffic Pattern")
    fig, ax = plt.subplots(figsize=(14, 6))
    hourly_traffic = df.groupby('hour').size()
    sns.lineplot(x=hourly_traffic.index, y=hourly_traffic.values, marker='o', ax=ax)
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("Top 10 Busiest Toll Plazas")
    fig, ax = plt.subplots(figsize=(12, 8))
    top_merchants = df['merchant_name'].value_counts().head(10)
    sns.barplot(y=top_merchants.index, x=top_merchants.values, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Traffic by Direction")
    fig, ax = plt.subplots(figsize=(10, 6))
    direction_counts = df['direction'].value_counts()
    sns.barplot(x=direction_counts.index, y=direction_counts.values, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Transaction Amount by Vehicle Class")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(x='vehicle_class_code', y='txn_amount', data=df, ax=ax)
    st.pyplot(fig)
    
    st.subheader("Traffic by Hour at Top 5 Toll Plazas")
    top5_plazas = df['merchant_name'].value_counts().head(5).index
    plaza_hour_df = df[df['merchant_name'].isin(top5_plazas)]
    heatmap_data = pd.crosstab(plaza_hour_df['merchant_name'], plaza_hour_df['hour'])
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt='d', ax=ax)
    st.pyplot(fig)
# -------------------------------
# Traffic Prediction
# -------------------------------
with tab1:
    st.header("Traffic Volume Prediction")
    with st.form("traffic_form"):
        st.subheader("Enter Prediction Parameters")
        
        # Time-based features section
        st.markdown("### Time Parameters")
        col1, col2 = st.columns(2)
        with col1:
            hour = st.slider("Hour of Day", 0, 23, 12)
            day_of_week = st.selectbox("Day of Week", 
                                      ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 
                                      index=3)
        
        with col2:
            is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
            st.write(f"Is Weekend: {'Yes' if is_weekend else 'No'}")
            
            quarter_of_day = hour // 6
            day_parts = ["Night (12AM-6AM)", "Morning (6AM-12PM)", "Afternoon (12PM-6PM)", "Evening (6PM-12AM)"]
            st.write(f"Day Part: {day_parts[quarter_of_day]}")
            day_part = quarter_of_day  # This is the numeric value for the model
        
        # Compute cyclical time features (hidden from user)
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        day_num = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
        day_sin = math.sin(2 * math.pi * day_num / 7)
        day_cos = math.cos(2 * math.pi * day_num / 7)
        
        # NEW: Merchant and Vehicle Class selection
        st.markdown("### Merchant and Vehicle Class")
        col1, col2 = st.columns(2)
        
        with col1:
            # Add a dropdown for merchant selection with "Overall" as default
            merchant_options = ["Overall", "ATTIBELLE ", "Devanahalli Toll Plaza", "ELECTRONIC  CITY Phase 1", "Banglaore-Nelamangala Plaza", "Hoskote Toll Plaza"]
            selected_merchant = st.selectbox("Select Merchant", merchant_options, index=0)
            
            # Display message about merchant selection
            if selected_merchant == "Overall":
                st.write("Predicting for all merchants combined")
            else:
                st.write(f"Predicting specifically for {selected_merchant}")
        
        with col2:
            # Add a dropdown for vehicle class selection with "All Classes" as default
            vehicle_options = ["All Classes", "VC14", "VC5", "VC6", "VC7", "VC8", "VC9", "VC10", 
                              "VC11", "VC12", "VC13", "VC4", "VC15", "VC16", "VC20"]
            selected_vehicle = st.selectbox("Select Vehicle Class", vehicle_options, index=0)
            
            # Display message about vehicle class selection
            if selected_vehicle == "All Classes":
                st.write("Predicting for all vehicle classes")
            else:
                st.write(f"Predicting specifically for {selected_vehicle}")
        
        # Historical data section
        st.markdown("### Historical Traffic Data")
        st.markdown("*The following values represent recent traffic patterns*")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Previous Hours' Traffic Volumes")
            lag_1 = st.slider("1 Hour Ago", 0, 200, 50)
            lag_2 = st.slider("2 Hours Ago", 0, 200, 48)
            lag_3 = st.slider("3 Hours Ago", 0, 200, 52)
        
        with col2:
            st.markdown("#### More Historical Data")
            lag_4 = st.slider("4 Hours Ago", 0, 200, 49)
            lag_5 = st.slider("5 Hours Ago", 0, 200, 51)
        
        # Rolling statistics
        st.markdown("### Traffic Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rolling_mean_3 = st.slider("3-Hour Average", 0, 200, 50)
            rolling_std_3 = st.slider("3-Hour Std Dev", 0, 30, 5)
        
        with col2:
            rolling_mean_6 = st.slider("6-Hour Average", 0, 200, 50)
            rolling_min_6 = st.slider("6-Hour Minimum", 0, 200, 40)
        
        with col3:
            rolling_max_6 = st.slider("6-Hour Maximum", 0, 200, 60)
        
        # Traffic change metrics
        st.markdown("### Traffic Change Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            tx_diff_1 = st.slider("Traffic Change (1h)", -50, 50, 2)
            tx_diff_2 = st.slider("Traffic Change (2h)", -50, 50, 3)
        
        with col2:
            tx_per_second = st.slider("Transactions Per Second", 0.0, 2.0, 0.5)
            avg_amount_per_tx = st.slider("Avg Transaction Amount ($)", 0.0, 50.0, 10.0)
        
        # Submit button
        submitted = st.form_submit_button("Predict Traffic Volume")
        
        if submitted:
            if traffic_model is None:
                st.error("Traffic model not available")
            else:
                try:
                    # Initialize merchant & vehicle class features with zeros first
                    merchant_features = {}
                    for m in merchant_options[1:]:  # Skip 'Overall'
                        merchant_key = f"merchant_{m.replace(' ', '_')}"
                        merchant_features[merchant_key] = [0.0]
                        # Add percentage features as well (which model might expect)
                        merchant_features[f"{merchant_key}_pct"] = [0.0]
                    
                    vehicle_features = {}
                    for vc in vehicle_options[1:]:  # Skip 'All Classes'
                        vehicle_key = f"vc_{vc}"
                        vehicle_features[vehicle_key] = [0.0]
                        # Add percentage features as well (which model might expect)
                        vehicle_features[f"{vehicle_key}_pct"] = [0.0]
                    
                    # Set merchant percentages
                    if selected_merchant != "Overall":
                        merchant_key = f"merchant_{selected_merchant.replace(' ', '_')}"
                        merchant_features[merchant_key] = [100.0]
                        merchant_features[f"{merchant_key}_pct"] = [100.0]
                        
                        # Reset other merchants to zero
                        for m in merchant_options[1:]:
                            if m != selected_merchant:
                                merchant_key = f"merchant_{m.replace(' ', '_')}"
                                merchant_features[merchant_key] = [0.0]
                                merchant_features[f"{merchant_key}_pct"] = [0.0]
                    else:
                        # For overall case, set even distribution
                        merchant_count = len(merchant_options) - 1  # Subtract 1 for "Overall"
                        even_percentage = 100.0 / merchant_count if merchant_count > 0 else 0.0
                        for m in merchant_options[1:]:
                            merchant_key = f"merchant_{m.replace(' ', '_')}"
                            merchant_features[merchant_key] = [even_percentage]
                            merchant_features[f"{merchant_key}_pct"] = [even_percentage]
                    
                    # Set vehicle class percentages
                    if selected_vehicle != "All Classes":
                        vehicle_key = f"vc_{selected_vehicle}"
                        vehicle_features[vehicle_key] = [100.0]
                        vehicle_features[f"{vehicle_key}_pct"] = [100.0]
                        
                        # Reset other vehicle classes to zero
                        for vc in vehicle_options[1:]:
                            if vc != selected_vehicle:
                                vehicle_key = f"vc_{vc}"
                                vehicle_features[vehicle_key] = [0.0]
                                vehicle_features[f"{vehicle_key}_pct"] = [0.0]
                    else:
                        # For all classes case, set even distribution
                        vehicle_count = len(vehicle_options) - 1  # Subtract 1 for "All Classes"
                        even_percentage = 100.0 / vehicle_count if vehicle_count > 0 else 0.0
                        for vc in vehicle_options[1:]:
                            vehicle_key = f"vc_{vc}"
                            vehicle_features[vehicle_key] = [even_percentage]
                            vehicle_features[f"{vehicle_key}_pct"] = [even_percentage]
                    
                    # Create features DataFrame with all features
                    base_features = {
                        'hour': [hour],
                        'day_of_week': [day_num],
                        'is_weekend': [is_weekend],
                        'quarter_of_day': [quarter_of_day],
                        'day_part': [day_part],
                        'hour_sin': [hour_sin],
                        'hour_cos': [hour_cos],
                        'day_sin': [day_sin],
                        'day_cos': [day_cos],
                        'lag_1': [lag_1],
                        'lag_2': [lag_2],
                        'lag_3': [lag_3],
                        'lag_4': [lag_4],
                        'lag_5': [lag_5],
                        'rolling_mean_3': [rolling_mean_3],
                        'rolling_std_3': [rolling_std_3],
                        'rolling_mean_6': [rolling_mean_6],
                        'rolling_max_6': [rolling_max_6],
                        'rolling_min_6': [rolling_min_6],
                        'tx_diff_1': [tx_diff_1],
                        'tx_diff_2': [tx_diff_2],
                        'tx_per_second': [tx_per_second],
                        'avg_amount_per_tx': [avg_amount_per_tx]
                    }
                    
                    # Combine all features
                    features_dict = {**base_features, **merchant_features, **vehicle_features}
                    features = pd.DataFrame(features_dict)
                    # print("siubsiubsuibsiubsubsbusbsubsbsubs",features)
                    
                    # # DEBUG: Show current feature values to help diagnose the issue
                    # with st.expander("Debug - Feature Values"):
                    #     st.write("Vehicle Class Features:")
                    #     for key, value in features_dict.items():
                    #         if key.startswith('vc_'):
                    #             st.write(f"{key}: {value[0]}")
                                
                    #     st.write("Merchant Features:")
                    #     for key, value in features_dict.items():
                    #         if key.startswith('merchant_'):
                    #             st.write(f"{key}: {value[0]}")
                    
                    # # DEBUG: Check if model contains these features
                    if hasattr(traffic_model, 'feature_names_in_'):
                        with st.expander("Model Features"):
                            model_features = traffic_model.feature_names_in_
                            st.write("Features expected by model:")
                            st.write(model_features)
                            
                            missing_features = [f for f in model_features if f not in features.columns]
                            st.write("Missing features (in model but not in input):")
                            st.write(missing_features)
                            
                            extra_features = [f for f in features.columns if f not in model_features]
                            st.write("Extra features (in input but not used by model):")
                            st.write(extra_features)
                    
                    # Make sure features match what the model expects
                    # If your model is a pipeline with preprocessor
                    # if hasattr(traffic_model, 'named_steps') and 'preprocessor' in traffic_model.named_steps:
                    #     # Extract feature names from the preprocessor
                    #     preprocessor = traffic_model.named_steps['preprocessor']
                    #     if hasattr(preprocessor, 'get_feature_names_out'):
                    #         expected_features = preprocessor.get_feature_names_out()
                    #         # Keep only features the model knows about
                    #         valid_features = [f for f in features.columns if f in expected_features]
                    #         features = features[valid_features]
                    # # If your model is a direct estimator
                    # elif hasattr(traffic_model, 'feature_names_in_'):
                    #     # Keep only features the model knows about
                    #     valid_features = [f for f in features.columns if f in traffic_model.feature_names_in_]
                    #     features = features[valid_features]
                    #     print("siubsiubsbs",features)
                    
                    # print("siubssibisbisbisbiubsbs",features)
                    # Make prediction
                    prediction = traffic_model.predict(features)
                    # print("isnisnsi",prediction)
    
                    # Create title with selected merchant and vehicle class
                    prediction_title = "Predicted Traffic Volume"
                    if selected_merchant != "Overall":
                        prediction_title += f" for {selected_merchant}"
                    if selected_vehicle != "All Classes":
                        prediction_title += f" - {selected_vehicle}"
                    
                    # Display prediction with visualizations
                    st.success(f"{prediction_title}: {prediction[0]:.0f} transactions")
                    
                    # Show a gauge chart for the prediction
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction[0],
                        title={'text': prediction_title},
                        gauge={'axis': {'range': [0, 3000]},
                               'bar': {'color': "darkblue"},
                               'steps': [
                                   {'range': [0, 500], 'color': "lightgray"},
                                   {'range': [500, 1000], 'color': "gray"},
                                   {'range': [1000, 2000], 'color': "lightblue"},
                                   {'range': [2000, 3000], 'color': "royalblue"}]}))
                    
                    st.plotly_chart(fig)
                    
                    # Show comparison if we're making a specific selection
                    if selected_merchant != "Overall" or selected_vehicle != "All Classes":
                        # Make a prediction for the overall case for comparison
                        overall_features = pd.DataFrame(base_features)
                        
                        # Add evenly distributed merchant features
                        merchant_count = len(merchant_options) - 1
                        even_percentage = 100.0 / merchant_count if merchant_count > 0 else 0.0
                        for m in merchant_options[1:]:
                            merchant_key = f"merchant_{m.replace(' ', '_')}"
                            if merchant_key in model_features:
                                overall_features[merchant_key] = [even_percentage]
                            if f"{merchant_key}_pct" in model_features:
                                overall_features[f"{merchant_key}_pct"] = [even_percentage]
                        
                        # Add evenly distributed vehicle features
                        vehicle_count = len(vehicle_options) - 1
                        even_percentage = 100.0 / vehicle_count if vehicle_count > 0 else 0.0
                        for vc in vehicle_options[1:]:
                            vehicle_key = f"vc_{vc}"
                            if vehicle_key in model_features:
                                overall_features[vehicle_key] = [even_percentage]
                            if f"{vehicle_key}_pct" in model_features:
                                overall_features[f"{vehicle_key}_pct"] = [even_percentage]
                        
                        # Keep only features the model knows about
                        valid_features = [f for f in overall_features.columns if f in model_features]
                        overall_features = overall_features[valid_features]
                        
                        try:
                            overall_prediction = traffic_model.predict(overall_features)
                            
                            # Show comparison
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Selected Configuration", 
                                         f"{prediction[0]:.0f}", 
                                         f"{prediction[0] - overall_prediction[0]:.0f}")
                            with col2:
                                st.metric("Overall Average", 
                                         f"{overall_prediction[0]:.0f}", 
                                         "")
                        except Exception as e:
                            st.warning(f"Could not generate comparison: {str(e)}")
                    
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.exception(e)

# with tab1:
#     st.header("Traffic Volume Prediction")
#     with st.form("traffic_form"):
#         st.subheader("Enter Prediction Parameters")
        
#         # Time-based features section
#         st.markdown("### Time Parameters")
#         col1, col2 = st.columns(2)
#         with col1:
#             hour = st.slider("Hour of Day", 0, 23, 12)
#             day_of_week = st.selectbox("Day of Week", 
#                                       ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 
#                                       index=3)
        
#         with col2:
#             is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
#             st.write(f"Is Weekend: {'Yes' if is_weekend else 'No'}")
            
#             quarter_of_day = hour // 6
#             day_parts = ["Night (12AM-6AM)", "Morning (6AM-12PM)", "Afternoon (12PM-6PM)", "Evening (6PM-12AM)"]
#             st.write(f"Day Part: {day_parts[quarter_of_day]}")
#             day_part = quarter_of_day  # This is the numeric value for the model
        
#         # Compute cyclical time features (hidden from user)
#         hour_sin = math.sin(2 * math.pi * hour / 24)
#         hour_cos = math.cos(2 * math.pi * hour / 24)
#         day_num = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)
#         day_sin = math.sin(2 * math.pi * day_num / 7)
#         day_cos = math.cos(2 * math.pi * day_num / 7)
        
#         # NEW: Merchant and Vehicle Class selection
#         st.markdown("### Merchant and Vehicle Class")
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # Add a dropdown for merchant selection with "Overall" as default
#             merchant_options = ["Overall", "ATTIBELLE ", "Devanahalli Toll Plaza", "ELECTRONIC  CITY Phase 1", "Banglaore-Nelamangala Plaza", "Hoskote Toll Plaza"]
#             selected_merchant = st.selectbox("Select Merchant", merchant_options, index=0)
            
#             # Display message about merchant selection
#             if selected_merchant == "Overall":
#                 st.write("Predicting for all merchants combined")
#             else:
#                 st.write(f"Predicting specifically for {selected_merchant}")
        
#         with col2:
#             # Add a dropdown for vehicle class selection with "All Classes" as default
#             vehicle_options = ["All Classes", "VC4", "VC5", "VC6", "VC7", "VC8", "VC9", "VC10", 
#                               "VC11", "VC12", "VC13", "VC14", "VC15", "VC16", "VC20"]
#             selected_vehicle = st.selectbox("Select Vehicle Class", vehicle_options, index=0)
            
#             # Display message about vehicle class selection
#             if selected_vehicle == "All Classes":
#                 st.write("Predicting for all vehicle classes")
#             else:
#                 st.write(f"Predicting specifically for {selected_vehicle}")
        
#         # Historical data section
#         st.markdown("### Historical Traffic Data")
#         st.markdown("*The following values represent recent traffic patterns*")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("#### Previous Hours' Traffic Volumes")
#             lag_1 = st.slider("1 Hour Ago", 0, 200, 50)
#             lag_2 = st.slider("2 Hours Ago", 0, 200, 48)
#             lag_3 = st.slider("3 Hours Ago", 0, 200, 52)
        
#         with col2:
#             st.markdown("#### More Historical Data")
#             lag_4 = st.slider("4 Hours Ago", 0, 200, 49)
#             lag_5 = st.slider("5 Hours Ago", 0, 200, 51)
        
#         # Rolling statistics
#         st.markdown("### Traffic Statistics")
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             rolling_mean_3 = st.slider("3-Hour Average", 0, 200, 50)
#             rolling_std_3 = st.slider("3-Hour Std Dev", 0, 30, 5)
        
#         with col2:
#             rolling_mean_6 = st.slider("6-Hour Average", 0, 200, 50)
#             rolling_min_6 = st.slider("6-Hour Minimum", 0, 200, 40)
        
#         with col3:
#             rolling_max_6 = st.slider("6-Hour Maximum", 0, 200, 60)
        
#         # Traffic change metrics
#         st.markdown("### Traffic Change Metrics")
#         col1, col2 = st.columns(2)
        
#         with col1:
#             tx_diff_1 = st.slider("Traffic Change (1h)", -50, 50, 2)
#             tx_diff_2 = st.slider("Traffic Change (2h)", -50, 50, 3)
        
#         with col2:
#             tx_per_second = st.slider("Transactions Per Second", 0.0, 2.0, 0.5)
#             avg_amount_per_tx = st.slider("Avg Transaction Amount ($)", 0.0, 50.0, 10.0)
        
#         # Submit button
#         submitted = st.form_submit_button("Predict Traffic Volume")
        
#         if submitted:
#             traffic_model = traffic_model
#             if traffic_model is None:
#                 st.error("Traffic model not available")
#             else:
#                 try:
#                     # Initialize merchant & vehicle class features with zeros first
#                     merchant_features = {}
#                     for m in merchant_options[1:]:  # Skip 'Overall'
#                         merchant_key = f"merchant_{m.replace(' ', '_')}"
#                         merchant_features[merchant_key] = [0.0]
                    
#                     vehicle_features = {}
#                     for vc in vehicle_options[1:]:  # Skip 'All Classes'
#                         vehicle_key = f"vc_{vc}"
#                         vehicle_features[vehicle_key] = [0.0]
                    
#                     # Set selected merchant to 100% if not overall
#                     if selected_merchant != "Overall":
#                         merchant_key = f"merchant_{selected_merchant.replace(' ', '_')}"
#                         merchant_features[merchant_key] = [100.0]
#                     else:
#                         # For overall case, set even distribution or use historical averages
#                         merchant_count = len(merchant_options) - 1  # Subtract 1 for "Overall"
#                         even_percentage = 100.0 / merchant_count if merchant_count > 0 else 0.0
#                         for key in merchant_features:
#                             merchant_features[key] = [even_percentage]
                    
#                     # Set selected vehicle class to 100% if not all classes
#                     if selected_vehicle != "All Classes":
#                         vehicle_key = f"vc_{selected_vehicle}"
#                         vehicle_features[vehicle_key] = [100.0]
#                     else:
#                         # For all classes case, set even distribution or use historical averages
#                         vehicle_count = len(vehicle_options) - 1  # Subtract 1 for "All Classes"
#                         even_percentage = 100.0 / vehicle_count if vehicle_count > 0 else 0.0
#                         for key in vehicle_features:
#                             vehicle_features[key] = [even_percentage]
                    
#                     # Create features DataFrame with all features
#                     base_features = {
#                         'hour': [hour],
#                         'day_of_week': [day_num],
#                         'is_weekend': [is_weekend],
#                         'quarter_of_day': [quarter_of_day],
#                         'day_part': [day_part],
#                         'hour_sin': [hour_sin],
#                         'hour_cos': [hour_cos],
#                         'day_sin': [day_sin],
#                         'day_cos': [day_cos],
#                         'lag_1': [lag_1],
#                         'lag_2': [lag_2],
#                         'lag_3': [lag_3],
#                         'lag_4': [lag_4],
#                         'lag_5': [lag_5],
#                         'rolling_mean_3': [rolling_mean_3],
#                         'rolling_std_3': [rolling_std_3],
#                         'rolling_mean_6': [rolling_mean_6],
#                         'rolling_max_6': [rolling_max_6],
#                         'rolling_min_6': [rolling_min_6],
#                         'tx_diff_1': [tx_diff_1],
#                         'tx_diff_2': [tx_diff_2],
#                         'tx_per_second': [tx_per_second],
#                         'avg_amount_per_tx': [avg_amount_per_tx]
#                     }
                    
#                     # Combine all features
#                     features_dict = {**base_features, **merchant_features, **vehicle_features}
#                     features = pd.DataFrame(features_dict)
                    
#                     # Make prediction
#                     prediction = traffic_model.predict(features)
                    
#                     # Create title with selected merchant and vehicle class
#                     prediction_title = "Predicted Traffic Volume"
#                     if selected_merchant != "Overall":
#                         prediction_title += f" for {selected_merchant}"
#                     if selected_vehicle != "All Classes":
#                         prediction_title += f" - {selected_vehicle}"
                    
#                     # Display prediction with visualizations
#                     st.success(f"{prediction_title}: {prediction[0]:.0f} transactions")
                    
#                     # Show a gauge chart for the prediction
#                     fig = go.Figure(go.Indicator(
#                         mode="gauge+number",
#                         value=prediction[0],
#                         title={'text': prediction_title},
#                         gauge={'axis': {'range': [0, 200]},
#                                'bar': {'color': "darkblue"},
#                                'steps': [
#                                    {'range': [0, 50], 'color': "lightgray"},
#                                    {'range': [50, 100], 'color': "gray"},
#                                    {'range': [100, 150], 'color': "lightblue"},
#                                    {'range': [150, 200], 'color': "royalblue"}]}))
                    
#                     st.plotly_chart(fig)
                    
#                 except Exception as e:
#                     st.error(f"Prediction error: {str(e)}")
#                     st.exception(e)

    st.header("Vehicle Class Distribution Prediction")
    with st.form("vehicle_class_form"):
        st.subheader("Enter Parameters for Vehicle Class Distribution")
        
        # Time parameters
        col1, col2 = st.columns(2)
        
        with col1:
            hour_vc = st.slider("Hour of Day", 0, 23, 12, key="vc_hour")
            day_selector = st.selectbox("Day of Week", 
                                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], 
                                    index=3, key="vc_day")
            day_of_week_vc = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_selector)
        
        with col2:
            is_weekend_vc = 1 if day_selector in ["Saturday", "Sunday"] else 0
            st.write(f"Is Weekend: {'Yes' if is_weekend_vc else 'No'}")
            
            day_part_options = ["Night (12AM-6AM)", "Morning (6AM-12PM)", "Afternoon (12PM-6PM)", "Evening (6PM-12AM)"]
            day_part_name = day_part_options[hour_vc // 6]
            day_part_vc = hour_vc // 6
            st.write(f"Day Part: {day_part_name}")
        
        # NEW: Merchant selection
        st.markdown("### Merchant Selection")
        merchant_options = ["Overall", "ATTIBELLE ", "Devanahalli Toll Plaza", "ELECTRONIC  CITY Phase 1", "Banglaore-Nelamangala Plaza", "Hoskote Toll Plaza"]
        selected_merchant_vc = st.selectbox("Select Merchant", merchant_options, index=0, key="vc_merchant")
        
        if selected_merchant_vc == "Overall (All Merchants)":
            st.write("Predicting vehicle class distribution across all merchants")
        else:
            st.write(f"Predicting vehicle class distribution for {selected_merchant_vc}")
        
        # Calculate cyclical features (hidden from user)
        hour_sin_vc = math.sin(2 * math.pi * hour_vc / 24)
        hour_cos_vc = math.cos(2 * math.pi * hour_vc / 24)
        day_sin_vc = math.sin(2 * math.pi * day_of_week_vc / 7)
        day_cos_vc = math.cos(2 * math.pi * day_of_week_vc / 7)
        
        # Additional contextual features
        st.markdown("### Additional Context (Optional)")
        
        col1, col2 = st.columns(2)
        with col1:
            has_weather = st.checkbox("Include Weather Information")
            temperature = None
            precipitation = None
            
            if has_weather:
                temperature = st.slider("Temperature (°F)", 0, 100, 70)
                precipitation = st.slider("Precipitation (mm)", 0.0, 50.0, 0.0)
        
        with col2:
            has_directions = st.checkbox("Include Traffic Direction")
            direction_n = None
            direction_s = None
            direction_e = None
            direction_w = None
            
            if has_directions:
                st.markdown("Traffic Direction Distribution (%)")
                direction_n = st.slider("Northbound", 0, 100, 25)
                direction_s = st.slider("Southbound", 0, 100, 25)
                direction_e = st.slider("Eastbound", 0, 100, 25)
                direction_w = st.slider("Westbound", 0, 100, 25)
                
                # Normalize to ensure total = 100%
                total = direction_n + direction_s + direction_e + direction_w
                if total > 0:
                    direction_n = round((direction_n / total) * 100)
                    direction_s = round((direction_s / total) * 100)
                    direction_e = round((direction_e / total) * 100)
                    direction_w = round((direction_w / total) * 100)
                
                st.write(f"Normalized Distribution: N:{direction_n}%, S:{direction_s}%, E:{direction_e}%, W:{direction_w}%")
        
        # Merchant-specific features
        merchant_features = {}
        if selected_merchant_vc != "Overall (All Merchants)":
            for merchant in merchant_options[1:]:  # Skip "Overall"
                if merchant == selected_merchant_vc:
                    key=f"merchant_{merchant}"
                    merchant_features[key] = 100
                else:
                    key=f"merchant_{merchant}"
                    merchant_features[key] = 0
        
        # Submit button
        submit_button = st.form_submit_button("Predict Vehicle Class Distribution")
        
        if submit_button:
            vc_model = vc_model
            if vc_model is None:
                st.error("vc_model not available")
            # Create feature vector for prediction
            features = {
                'hour': hour_vc,
                'day_of_week': day_of_week_vc,
                'is_weekend': is_weekend_vc,
                'day_part': day_part_vc,
                'hour_sin': hour_sin_vc,
                'hour_cos': hour_cos_vc,
                'day_sin': day_sin_vc,
                'day_cos': day_cos_vc
            }
            
            # Add optional features if selected
            if has_weather:
                features['temperature'] = temperature
                features['precipitation'] = precipitation
            
            if has_directions:
                features['direction_N'] = direction_n
                features['direction_S'] = direction_s
                features['direction_E'] = direction_e
                features['direction_W'] = direction_w
                features["merchant_ATTIBELLE "]=70
            else:
                features['direction_N'] = 25
                features['direction_S'] = 25
                features['direction_E'] = 25
                features['direction_W'] = 25
                features["merchant_ATTIBELLE "]=70
            
            # Add merchant features if applicable
            for key, value in merchant_features.items():
                features[key] = value
            
            # Create a DataFrame with one row for prediction
            X_pred = pd.DataFrame([features])
            
            # Make prediction
            prediction = vc_model.predict(X_pred)[0]
            vehicle_classes = ['VC14', 'VC20', 'VC5', 'VC10', 'VC12', 'VC7', 'VC13', 'VC11', 'VC4', 'VC9', 'VC8', 'VC15', 'VC16', 'VC6']
            # Convert prediction to a pandas Series for easier manipulation
            pred_series = pd.Series(prediction, index=vehicle_classes)
            
            # Display results
            st.success("Prediction successful!")
            
            # Display as a bar chart
            st.subheader("Predicted Vehicle Class Distribution")
            fig = plt.figure(figsize=(10, 6))
            bars = plt.bar(pred_series.index, pred_series.values, color='skyblue')
            plt.xlabel('Vehicle Class')
            plt.ylabel('Predicted Proportion')
            plt.title(f'Predicted Vehicle Class Distribution for {day_selector} at {hour_vc}:00')
            plt.xticks(rotation=45)
            
            # Add percentage labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display as a table too
            st.subheader("Detailed Distribution")
            result_df = pd.DataFrame({
                'Vehicle Class': pred_series.index,
                'Predicted Proportion': pred_series.values.round(2)
            })
            st.dataframe(result_df.sort_values('Predicted Proportion', ascending=False))
            
                
            
            

    
  


# -------------------------------
# Toll Skipping Detection
# -------------------------------
with tab4:
    st.header("Traffic Anomaly Detection")
    with st.form("anomaly_form"):
        st.subheader("Enter Anomaly Detection Parameters")
        transaction_count = st.number_input("Transaction Count", value=50.0, key="anomaly_tx")
        hour_anomaly = st.number_input("Hour", value=12.0, key="anomaly_hour")
        day_of_week_anomaly = st.number_input("Day of Week (0=Mon, 6=Sun)", value=3.0, key="anomaly_day")
        is_weekend_anomaly = st.number_input("Is Weekend (0/1)", value=0, key="anomaly_weekend")
        month_anomaly = st.number_input("Month", value=6, key="anomaly_month")
        day_anomaly = st.number_input("Day", value=15, key="anomaly_daynum")
        submitted_anomaly = st.form_submit_button("Detect Anomaly")
        
        if submitted_anomaly:
            if anomaly_scaler is None or not anomaly_model:
                st.error("Anomaly models not available")
            else:
                try:
                    features = np.array([[
                        transaction_count, hour_anomaly, day_of_week_anomaly,
                        is_weekend_anomaly, month_anomaly, day_anomaly
                    ]])
                    
                    # Scale the features using the pre-fitted scaler
                    X_scaled = anomaly_scaler.transform(features)
                    
                    # Get models
                    if_model, _ = anomaly_model['isolation_forest']
                    dbscan_model, _ = anomaly_model['dbscan']
                    
                    # Get predictions
                    if_pred = if_model.predict(X_scaled)
                    if_score = if_model.score_samples(X_scaled)[0]
                    dbscan_pred = dbscan_model.fit_predict(X_scaled)
                    
                    # Ensemble: mark as anomaly if either method flags it
                    ensemble = 1 if (if_pred[0] == -1 or dbscan_pred[0] == -1) else 0
                    
                    # Create a container for the results
                    results_container = st.container()
                    
                    # Display main result
                    if ensemble == 1:
                        results_container.error("🚨 ANOMALY DETECTED")
                    else:
                        results_container.success("✅ No Anomaly Detected")
                    
                    # Create columns for detailed information
                    col1, col2 = results_container.columns(2)
                    
                    # Display detailed model results
                    col1.subheader("Model Results")
                    col1.info(f"Isolation Forest: {'🚨 ANOMALY' if if_pred[0] == -1 else '✅ Normal'}")
                    col1.info(f"DBSCAN: {'🚨 ANOMALY' if dbscan_pred[0] == -1 else '✅ Normal'}")
                    col1.info(f"Anomaly Score: {if_score:.4f} (lower is more anomalous)")
                    
                    # Display feature analysis
                    col2.subheader("Feature Analysis")
                    
                    # Get day name from day of week
                    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
                                3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
                    day_name = day_names.get(int(day_of_week_anomaly), 'Unknown')
                    
                    # Analyze each feature for anomalies
                    feature_analysis = []
                    
                    # Transaction count analysis
                    if transaction_count > 100:
                        feature_analysis.append("📈 High transaction volume")
                    elif transaction_count < 10:
                        feature_analysis.append("📉 Low transaction volume")
                    
                    # Time analysis
                    if 0 <= hour_anomaly < 6:
                        feature_analysis.append("🌙 Unusual overnight activity")
                    
                    # Weekend analysis
                    if is_weekend_anomaly == 1 and transaction_count > 80:
                        feature_analysis.append("🚗 Unusually high weekend traffic")
                    
                    # Display analysis
                    if feature_analysis:
                        for analysis in feature_analysis:
                            col2.warning(analysis)
                    else:
                        col2.info("No specific feature anomalies identified")
                    
                    # Display context information
                    results_container.subheader("Context Information")
                    
                    # Create explanation
                    explanation = f"""
                    **Input Details:**
                    - Transaction Count: {transaction_count}
                    - Time: {int(hour_anomaly)}:00 hours
                    - Day: {day_name}
                    - {'Weekend' if is_weekend_anomaly == 1 else 'Weekday'}
                    - Date: {int(day_anomaly)}/{int(month_anomaly)}
                    
                    **Anomaly Analysis:**
                    """
                    
                    if ensemble == 1:
                        if if_pred[0] == -1 and dbscan_pred[0] == -1:
                            explanation += """
                            **Strong anomaly detected** - Both models identified this as an anomaly.
                            This pattern significantly deviates from normal traffic patterns.
                            """
                        else:
                            explanation += """
                            **Moderate anomaly detected** - One model identified this as an anomaly.
                            This pattern shows some deviation from expected traffic patterns.
                            """
                            
                        if transaction_count > 100:
                            explanation += f"""
                            The transaction count of {transaction_count} is unusually high for this time period.
                            This could indicate a traffic surge or potential system error.
                            """
                        elif transaction_count < 10:
                            explanation += f"""
                            The transaction count of {transaction_count} is unusually low for this time period.
                            This could indicate a system outage, holiday, or other disruption.
                            """
                            
                        if is_weekend_anomaly == 1 and 10 <= hour_anomaly <= 16:
                            explanation += """
                            Weekend daytime patterns are showing anomalous behavior.
                            This could indicate a special event, road closure, or other unusual circumstance.
                            """
                    else:
                        explanation += """
                        No anomaly detected. This traffic pattern falls within expected normal ranges.
                        """
                    
                    results_container.markdown(explanation)
                    
                    # Display recommendation
                    results_container.subheader("Recommended Actions")
                    
                    if ensemble == 1:
                        actions = """
                        1. **Investigate** the specific time period for external factors
                        2. **Compare** with historical data for similar days/times
                        3. **Check** for system errors or data collection issues
                        4. **Monitor** closely for the next few periods
                        """
                    else:
                        actions = """
                        1. **Normal monitoring** - No special action required
                        2. **Continue** regular data collection and analysis
                        """
                    
                    results_container.markdown(actions)
                    
                except Exception as e:
                    st.error(f"Detection error: {str(e)}")
                    st.info("Error details: " + traceback.format_exc())
                    
                    
    st.header("Toll Skipping Detection")
    with st.form("toll_skipping_form"):
        st.subheader("Enter Toll Skipping Parameters")
        total_trips = st.number_input("Total Trips", value=10.0, key="ts_total")
        unique_plazas = st.number_input("Unique Plazas", value=2.0, key="ts_plazas")
        trips_per_hour = st.number_input("Trips per Hour", value=1.0, key="ts_trips")
        avg_amount_per_trip = st.number_input("Avg Amount per Trip", value=15.0, key="ts_amount")
        plaza_to_trip_ratio = st.number_input("Plaza to Trip Ratio", value=0.2, key="ts_ratio")
        submitted_toll = st.form_submit_button("Detect Toll Skipping")
        
        if submitted_toll:
            if skip_scaler is None or not skip_model:
                st.error("Toll skipping models not available")
            else:
                try:
                    features = np.array([[
                        total_trips, unique_plazas, trips_per_hour,
                        avg_amount_per_trip, plaza_to_trip_ratio
                    ]])
                    features_scaled = skip_scaler.transform(features)
                    # Using KMeans clustering
                    kmeans = skip_model['kmeans']
                    cluster = kmeans.predict(features_scaled)
                    # For demonstration, assume cluster 0 is the potential toll skipping group
                    is_toll_skip = cluster[0] == 0
                    
                    if is_toll_skip:
                        st.warning("⚠️ Potential Toll Skipping Detected")
                    else:
                        st.success("✅ Normal Vehicle Behavior")
                        
                    st.info(f"Cluster: {int(cluster[0])}")
                except Exception as e:
                    st.error(f"Detection error: {str(e)}")

# -------------------------------
# Vehicle Behavior Classification
# -------------------------------

    st.header("Vehicle Behavior Classification")
    with st.form("vehicle_behavior_form"):
        st.subheader("Enter Vehicle Behavior Parameters")
        total_trips_vb = st.number_input("Total Trips", value=10.0, key="vb_total")
        unique_plazas_vb = st.number_input("Unique Plazas", value=2.0, key="vb_plazas")
        trips_per_hour_vb = st.number_input("Trips per Hour", value=1.0, key="vb_trips")
        avg_amount_per_trip_vb = st.number_input("Avg Amount per Trip", value=15.0, key="vb_amount")
        plaza_to_trip_ratio_vb = st.number_input("Plaza to Trip Ratio", value=0.2, key="vb_ratio")
        submitted_vb = st.form_submit_button("Classify Vehicle Behavior")
        
        if submitted_vb:
            if vehicle_classifier is None:
                st.error("Vehicle behavior classifier not available")
            else:
                try:
                    features = np.array([[
                        total_trips_vb, unique_plazas_vb, trips_per_hour_vb,
                        avg_amount_per_trip_vb, plaza_to_trip_ratio_vb
                    ]])
                    prediction = vehicle_classifier.predict(features)
                    suspicious = prediction[0]
                    
                    if suspicious:
                        st.warning("⚠️ Suspicious Vehicle Behavior Detected")
                        insights = []
                        if total_trips_vb > 50:
                            insights.append("High number of total trips detected.")
                        if plaza_to_trip_ratio_vb < 0.3:
                            insights.append("Low plaza-to-trip ratio, indicating repetitive behavior.")
                        if trips_per_hour_vb > 5:
                            insights.append("Frequent trips per hour, unusual activity detected.")
                        if avg_amount_per_trip_vb < 5:
                            insights.append("Low amount per trip, possible fare evasion attempt.")
                        
                        if insights:
                            st.write("### Possible reasons for classification:")
                            for insight in insights:
                                st.write(f"- {insight}")
                    else:
                        st.success("✅ Normal Vehicle Behavior")
                except Exception as e:
                    st.error(f"Classification error: {str(e)}")

# Anomaly Detection Tab
# with tab4:
#     st.header("Enhanced Anomaly Detection System")
    
#     # Traffic Anomaly Detection Section
#     with st.expander("🚦 Traffic Pattern Anomaly Detection"):
#         with st.form("enhanced_anomaly_form"):
#             st.subheader("Real-time Traffic Monitoring")
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 transaction_count = st.number_input("Transaction Count", min_value=0.0, value=85.0)
#                 hour_anomaly = st.slider("Hour of Day", 0, 23, 12)
#                 day_of_week_anomaly = st.selectbox("Day of Week", 
#                                                 ["Monday", "Tuesday", "Wednesday", "Thursday", 
#                                                 "Friday", "Saturday", "Sunday"], index=3)
#                 quarter_of_day = st.slider("Quarter of Day", 0, 3, 1)
                
#             with col2:
#                 is_weekend_anomaly = st.checkbox("Weekend")
#                 month_anomaly = st.slider("Month", 1, 12, 6)
#                 day_anomaly = st.slider("Day of Month", 1, 31, 15)
#                 is_business_hours = st.checkbox("Business Hours", value=True)

#             # Add these new features
#             rolling_mean = st.number_input("3h Rolling Mean", min_value=0.0, value=75.0)
#             rolling_std = st.number_input("3h Rolling Std", min_value=0.0, value=12.5)
#             z_score = st.number_input("Z-Score", value=0.0)
            
#             submitted_anomaly = st.form_submit_button("Analyze Traffic Pattern")
            
#             if submitted_anomaly:
#                 if not traffic_models or traffic_scaler is None:
#                     st.error("Traffic anomaly models not loaded")
#                 else:
#                     try:
#                         # Convert inputs to numerical features
#                         day_map = {"Monday":0, "Tuesday":1, "Wednesday":2,
#                                  "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
#                         # Update the features array to match training dimensions
#                         features = np.array([[
#                             transaction_count,
#                             hour_anomaly,
#                             day_map[day_of_week_anomaly],
#                             int(is_weekend_anomaly),
#                             month_anomaly,
#                             day_anomaly,
#                             quarter_of_day,
#                             int(is_business_hours),
#                             rolling_mean,
#                             rolling_std,
#                             z_score
#                         ]])
                        
#                         # Scale features and predict
#                         X_scaled = traffic_scaler.transform(features)
                        
#                         # With this (access first element of tuple):
#                         if_pred = traffic_models['isolation_forest'][0].predict(X_scaled)
#                         ocsvm_pred = traffic_models['ocsvm'][0].predict(X_scaled)
#                         dbscan_pred = traffic_models['dbscan'][0].fit_predict(X_scaled)
                        
#                         # Calculate ensemble score
#                         ensemble_score = (
#                             (if_pred == -1) * 0.35 + 
#                             (lof_pred == -1) * 0.2 + 
#                             (ocsvm_pred == -1) * 0.15 + 
#                             (dbscan_pred == -1) * 0.3
#                         )
                        
#                         # Display results
#                         st.subheader("Results")
#                         if ensemble_score[0] >= 0.3:
#                             st.error("🚨 Critical Anomaly Detected")
#                         else:
#                             st.success("✅ Normal Traffic Pattern")
                            
#                         st.metric("Anomaly Score", f"{ensemble_score[0]:.2f}/1.00", 
#                                 delta="High Risk" if ensemble_score[0] >= 0.3 else "Low Risk")
                        
#                         # Model consensus
#                         st.write("### Model Agreement")
#                         cols = st.columns(4)
#                         model_data = {
#                             "Isolation Forest": "Anomaly" if if_pred[0] == -1 else "Normal",
#                             "Local Outlier": "Anomaly" if lof_pred[0] == -1 else "Normal",
#                             "One-Class SVM": "Anomaly" if ocsvm_pred[0] == -1 else "Normal",
#                             "OPTICS": "Anomaly" if dbscan_pred[0] == -1 else "Normal"
#                         }
#                         for (name, result), col in zip(model_data.items(), cols):
#                             col.metric(name, result)
                            
#                     except Exception as e:
#                         st.error(f"Analysis failed: {str(e)}")

#     # Enhanced Toll Skipping Detection
#     with st.expander("🚗 Advanced Toll Skipping Detection"):
#         with st.form("enhanced_toll_form"):
#             st.subheader("Vehicle Behavior Analysis")
#             cols = st.columns(3)
            
#             with cols[0]:
#                 total_trips = st.number_input("Total Trips", min_value=1, value=15)
#                 unique_plazas = st.number_input("Unique Plazas", min_value=1, value=3)
#                 trips_per_hour = st.number_input("Trips/Hour", min_value=0.1, value=2.5)
#                 plaza_ratio = st.slider("Plaza/Trip Ratio", 0.0, 1.0, 0.3)
                
#             with cols[1]:
#                 avg_amount = st.number_input("Avg Toll Amount", min_value=0.0, value=12.5)
#                 trip_duration_mean = st.number_input("Avg Trip Duration (min)", min_value=1, value=45)
#                 trip_duration_std = st.number_input("Trip Duration Std Dev (min)", min_value=0.0, value=10.0)
#                 transactions_per_trip = st.number_input("Transactions/Trip", min_value=1, value=3)
                
#             with cols[2]:
#                 amount_std = st.number_input("Amount Std Dev", min_value=0.0, value=5.0)
#                 total_transactions = st.number_input("Total Transactions", min_value=1, value=45)
#                 txn_count_mean = st.number_input("Avg Transactions per Trip", min_value=1, value=3)
#                 txn_count_std = st.number_input("Transactions Std Dev", min_value=0.0, value=1.0)

#             submitted_toll = st.form_submit_button("Analyze Vehicle Behavior")
            
#             if submitted_toll:
#                 if not toll_models or toll_scaler is None:
#                     st.error("Toll skipping models not loaded")
#                 else:
#                     try:
#                         # Prepare features with EXACTLY the same columns as training
#                         features = pd.DataFrame([{
#                             'total_trips': total_trips,
#                             'unique_plazas': unique_plazas,
#                             'trips_per_hour': trips_per_hour,
#                             'avg_amount_per_transaction': avg_amount,
#                             'plaza_to_trip_ratio': plaza_ratio,
#                             'trip_duration_min_mean': trip_duration_mean,
#                             'transactions_per_trip': transactions_per_trip,
#                             'amount_std': amount_std,
#                             'total_transactions': total_transactions,
#                             'merchant_name_nunique_mean': unique_plazas,  # Example mapping
#                             'txn_amount_sum_std': amount_std,  # Example mapping
#                             'txn_amount_count_mean': txn_count_mean,
#                             'txn_amount_count_std': txn_count_std,
#                             'trip_duration_min_std': trip_duration_std
#                         }])
                        
#                         # Transform features
#                         X_scaled = toll_scaler.transform(features)
#                         if toll_models['pca']:
#                             X_transformed = toll_models['pca'].transform(X_scaled)
#                         else:
#                             X_transformed = X_scaled
                            
#                         # Get predictions
#                         if_pred = toll_models['isolation_forest'].predict(X_transformed)
#                         ocsvm_pred = toll_models['ocsvm'].predict(X_transformed)
#                         xgb_prob = toll_models['xgb'].predict_proba(features)[:, 1]
                        
#                         # Calculate combined probability
#                         combined_prob = (np.mean([if_pred == -1, ocsvm_pred == -1]) * 0.5) + (xgb_prob * 0.5)
                        
#                         # Display results
#                         st.subheader("Risk Assessment")
#                         risk_level = "🟢 Low Risk" if combined_prob[0] < 0.4 else "🟠 Medium Risk" if combined_prob[0] < 0.7 else "🔴 High Risk"
#                         st.metric("Fraud Probability", f"{combined_prob[0]*100:.1f}%", risk_level)
                        
#                         # Risk factors
#                         st.write("### Risk Factors")
#                         factors = []
#                         if plaza_ratio < 0.25: factors.append("Low plaza/trip ratio")
#                         if trips_per_hour > 4: factors.append("High trip frequency")
#                         if avg_amount < 8: factors.append("Low average payment")
#                         if trip_duration_mean < 30: factors.append("Short trip duration")
                        
#                         if factors:
#                             for factor in factors:
#                                 st.warning(factor)
#                         else:
#                             st.info("No significant risk factors detected")
                            
#                     except Exception as e:
#                         st.error(f"Analysis failed: {str(e)}")

# Footer
st.markdown("---")
st.caption("Made in 1 day with ❤️  by Pradipta--- Could be better if time given")
st.markdown("---")