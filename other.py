import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import datetime
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
from typing import Dict,List,Any
warnings.filterwarnings('ignore')
from openai import OpenAI
# =============================================================================
# LOAD DATA AND MODELS
# =============================================================================
@st.cache_data
def load_data():
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

# =============================================================================
# LANE OPTIMIZATION SYSTEM
# =============================================================================
class LaneOptimizationSystem:
    def __init__(self, data):
        self.df = data
        self.plaza_lanes = self._get_plaza_lanes()
        self.vehicle_types = sorted(self.df['vehicle_class_code'].unique())
        self.lane_efficiency = self._calculate_lane_efficiency()
        
    def _get_plaza_lanes(self) -> Dict[str, List[str]]:
        plaza_lanes = {}
        for plaza in self.df['merchant_name'].unique():
            lanes = self.df[self.df['merchant_name'] == plaza]['lane'].unique()
            # Convert lanes to strings and sort
            plaza_lanes[plaza] = sorted(lanes, key=lambda x: str(x))
        return plaza_lanes

        
    def _calculate_lane_efficiency(self):
        lane_stats = self.df.groupby(['merchant_name', 'lane'])['inn_rr_time_sec'].agg(['mean', 'count', 'std']).reset_index()
        lane_stats.columns = ['plaza', 'lane', 'avg_processing_time', 'volume', 'time_std']
        scaler = MinMaxScaler()
        lane_stats[['norm_time', 'norm_volume']] = scaler.fit_transform(lane_stats[['avg_processing_time', 'volume']])
        lane_stats['efficiency_score'] = 0.7 * lane_stats['norm_time'] - 0.3 * lane_stats['norm_volume']
        return lane_stats
    
    def get_lane_recommendations(self, plaza, hour, expected_traffic=None):
        plaza_data = self.df[self.df['merchant_name'] == plaza].copy()
        if plaza_data.empty:
            return {"error": f"No data available for plaza {plaza}"}
        available_lanes = self.plaza_lanes.get(plaza, [])
        if not available_lanes:
            return {"error": f"No lanes data available for plaza {plaza}"}
        hourly_data = plaza_data[plaza_data['initiated_time'].dt.hour == hour]
        if hourly_data.empty:
            return {"error": f"No data available for plaza {plaza} at hour {hour}"}
        if 'vehicle_comvehicle' in hourly_data.columns:
            hourly_data['vehicle_comvehicle'] = pd.to_numeric(
                hourly_data['vehicle_comvehicle'].replace({'F': 0, 'T': 1}),
                errors='coerce'
            )
            commercial_ratio = hourly_data['vehicle_comvehicle'].mean()
            if pd.isna(commercial_ratio):
                commercial_ratio = 0.3
        else:
            commercial_ratio = 0.3
        vehicle_dist = hourly_data['vehicle_class_code'].value_counts(normalize=True)
        plaza_efficiency = self.lane_efficiency[self.lane_efficiency['plaza'] == plaza].copy()
        if expected_traffic is None:
            expected_traffic = len(hourly_data)
        lanes_needed = max(2, int(np.ceil(expected_traffic / 100)))
        lanes_needed = min(lanes_needed, len(available_lanes))
        best_lanes = plaza_efficiency.sort_values('efficiency_score').head(lanes_needed)
        commercial_lanes = max(1, int(np.round(commercial_ratio * lanes_needed)))
        recommended_lanes = []
        for i, (_, lane_data) in enumerate(best_lanes.iterrows()):
            lane_role = "Commercial" if i < commercial_lanes else "Non-commercial"
            recommended_lanes.append({
                "lane": lane_data['lane'],
                "role": lane_role,
                "expected_volume": int(expected_traffic / lanes_needed),
                "processing_time": float(lane_data['avg_processing_time'])
            })
        recommendations = {
            "plaza": plaza,
            "hour": hour,
            "expected_traffic": expected_traffic,
            "lanes_needed": lanes_needed,
            "commercial_ratio": float(commercial_ratio),
            "recommended_lanes": recommended_lanes,
            "vehicle_distribution": {k: float(v) for k, v in vehicle_dist.items()},
            "historical_average_processing_time": float(hourly_data['inn_rr_time_sec'].mean())
        }
        return recommendations
    
    def get_dynamic_pricing_recommendations(self, plaza, hour):
        plaza_data = self.df[self.df['merchant_name'] == plaza].copy()
        if plaza_data.empty:
            return {"error": f"No data available for plaza {plaza}"}
        hourly_data = plaza_data[plaza_data['initiated_time'].dt.hour == hour]
        if hourly_data.empty:
            return {"error": f"No data available for plaza {plaza} at hour {hour}"}
        max_hourly_traffic = plaza_data.groupby(plaza_data['initiated_time'].dt.hour).size().max()
        current_hourly_traffic = len(hourly_data)
        congestion_level = current_hourly_traffic / max_hourly_traffic if max_hourly_traffic > 0 else 0
        current_pricing = {}
        for vc in self.vehicle_types:
            vc_data = hourly_data[hourly_data['vehicle_class_code'] == vc]
            if not vc_data.empty:
                current_pricing[vc] = float(vc_data['txn_amount'].median())
        recommended_pricing = {}
        for vc, base_price in current_pricing.items():
            if congestion_level > 0.7:
                factor = 1.0 + (congestion_level - 0.7) * (0.2 / 0.3)
            elif congestion_level < 0.3:
                factor = 1.0 - (0.3 - congestion_level) * (0.1 / 0.3)
            else:
                factor = 1.0
            recommended_pricing[vc] = round(base_price * factor, 2)
        recommendations = {
            "plaza": plaza,
            "hour": hour,
            "congestion_level": float(congestion_level),
            "current_pricing": current_pricing,
            "recommended_pricing": recommended_pricing,
            "expected_impact": {
                "revenue_change": f"{(sum(recommended_pricing.values()) - sum(current_pricing.values())) / sum(current_pricing.values()) * 100:.2f}%",
                "expected_traffic_reduction": f"{5 * (congestion_level - 0.5) if congestion_level > 0.5 else 0:.2f}%"
            }
        }
        return recommendations

# =============================================================================
# AUTOMATED INSIGHTS GENERATOR
# =============================================================================
class AutomatedInsightsGenerator:
    def __init__(self, data, openai_api_key=None):
        self.df = data
        self.client = None
        if openai_api_key:
            try:
                self.client = OpenAI(api_key=openai_api_key)
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                print("Automated insights will be generated without OpenAI.")
    
    def _get_basic_stats(self) -> Dict[str, Any]:
        stats = {
            "total_transactions": len(self.df),
            "total_plazas": self.df['merchant_name'].nunique(),
            "total_vehicles": self.df['vehicle_regn_number'].nunique(),
            "total_revenue": float(self.df['txn_amount'].sum()),
            "avg_transaction_amount": float(self.df['txn_amount'].mean()),
            "busiest_plaza": self.df['merchant_name'].value_counts().index[0],
            "busiest_hour": self.df.groupby(self.df['initiated_time'].dt.hour).size().idxmax(),
            "most_common_vehicle_type": self.df['vehicle_class_code'].value_counts().index[0]
        }
        return stats
    
    def _get_traffic_insights(self) -> Dict[str, Any]:
        hourly_traffic = self.df.groupby(self.df['initiated_time'].dt.hour).size()
        peak_hour = hourly_traffic.idxmax()
        off_peak_hour = hourly_traffic.idxmin()
        morning_traffic = hourly_traffic.loc[6:12].sum()
        evening_traffic = hourly_traffic.loc[16:20].sum()
        night_traffic = hourly_traffic.loc[[*range(0, 6), *range(21, 24)]].sum()
        plaza_traffic = self.df['merchant_name'].value_counts()
        busiest_plaza = plaza_traffic.index[0]
        quietest_plaza = plaza_traffic.index[-1]
        direction_traffic = self.df['direction'].value_counts(normalize=True)
        main_direction = direction_traffic.index[0]
        main_direction_pct = float(direction_traffic.iloc[0] * 100)
        
        insights = {
            "peak_hour": int(peak_hour),
            "peak_hour_traffic": int(hourly_traffic[peak_hour]),
            "off_peak_hour": int(off_peak_hour),
            "off_peak_hour_traffic": int(hourly_traffic[off_peak_hour]),
            "peak_to_offpeak_ratio": float(hourly_traffic[peak_hour] / hourly_traffic[off_peak_hour]),
            "morning_vs_evening": {
                "morning_traffic": int(morning_traffic),
                "evening_traffic": int(evening_traffic),
                "ratio": float(morning_traffic / evening_traffic) if evening_traffic > 0 else float('inf')
            },
            "night_traffic_percentage": float(night_traffic / hourly_traffic.sum() * 100),
            "busiest_plaza": busiest_plaza,
            "busiest_plaza_transactions": int(plaza_traffic[busiest_plaza]),
            "quietest_plaza": quietest_plaza,
            "quietest_plaza_transactions": int(plaza_traffic[quietest_plaza]),
            "main_travel_direction": main_direction,
            "main_direction_percentage": main_direction_pct
        }
        return insights
    
    def _get_vehicle_insights(self) -> Dict[str, Any]:
        vehicle_dist = self.df['vehicle_class_code'].value_counts(normalize=True)
        
        if 'vehicle_comvehicle' in self.df.columns:
            # Convert the column to numeric: 'F' -> 0, 'T' -> 1, non-convertible values become NaN
            vc_series = pd.to_numeric(
                self.df['vehicle_comvehicle'].replace({'F': 0, 'T': 1}),
                errors='coerce'
            )
            commercial_pct = float(vc_series.mean() * 100)
        else:
            commercial_classes = ['VC5', 'VC6', 'VC7', 'VC8', 'VC10', 'VC11', 'VC12', 'VC13', 'VC14', 'VC15', 'VC16', 'VC17', 'VC20']
            commercial_pct = float(self.df['vehicle_class_code'].isin(commercial_classes).mean() * 100)
        
        revenue_by_class = self.df.groupby('vehicle_class_code')['txn_amount'].sum()
        top_revenue_class = revenue_by_class.idxmax()
        avg_by_class = self.df.groupby('vehicle_class_code')['txn_amount'].mean().sort_values(ascending=False)
        
        insights = {
            "top_vehicle_class": vehicle_dist.index[0],
            "top_vehicle_class_percentage": float(vehicle_dist.iloc[0] * 100),
            "commercial_vehicle_percentage": commercial_pct,
            "top_revenue_vehicle_class": top_revenue_class,
            "top_revenue_vehicle_class_amount": float(revenue_by_class[top_revenue_class]),
            "highest_fare_vehicle_class": avg_by_class.index[0],
            "highest_fare_amount": float(avg_by_class.iloc[0]),
            "vehicle_class_distribution": {k: float(v * 100) for k, v in vehicle_dist.items()}
        }
        return insights

    
    def _get_operational_insights(self) -> Dict[str, Any]:
        avg_processing = float(self.df['inn_rr_time_sec'].mean())
        plaza_processing = self.df.groupby('merchant_name')['inn_rr_time_sec'].mean().sort_values()
        fastest_plaza = plaza_processing.index[0]
        slowest_plaza = plaza_processing.index[-1]
        class_processing = self.df.groupby('vehicle_class_code')['inn_rr_time_sec'].mean().sort_values()
        fastest_class = class_processing.index[0]
        slowest_class = class_processing.index[-1]
        hour_processing = self.df.groupby(self.df['initiated_time'].dt.hour)['inn_rr_time_sec'].mean()
        fastest_hour = hour_processing.idxmin()
        slowest_hour = hour_processing.idxmax()
        
        insights = {
            "average_processing_time": avg_processing,
            "fastest_plaza": fastest_plaza,
            "fastest_plaza_time": float(plaza_processing[fastest_plaza]),
            "slowest_plaza": slowest_plaza,
            "slowest_plaza_time": float(plaza_processing[slowest_plaza]),
            "fastest_vehicle_class": fastest_class,
            "fastest_vehicle_class_time": float(class_processing[fastest_class]),
            "slowest_vehicle_class": slowest_class,
            "slowest_vehicle_class_time": float(class_processing[slowest_class]),
            "fastest_hour": int(fastest_hour),
            "fastest_hour_time": float(hour_processing[fastest_hour]),
            "slowest_hour": int(slowest_hour),
            "slowest_hour_time": float(hour_processing[slowest_hour])
        }
        return insights
    
    def _get_revenue_insights(self) -> Dict[str, Any]:
        hourly_revenue = self.df.groupby(self.df['initiated_time'].dt.hour)['txn_amount'].sum()
        peak_revenue_hour = hourly_revenue.idxmax()
        plaza_revenue = self.df.groupby('merchant_name')['txn_amount'].sum().sort_values(ascending=False)
        top_revenue_plaza = plaza_revenue.index[0]
        plaza_avg_revenue = self.df.groupby('merchant_name')['txn_amount'].mean().sort_values(ascending=False)
        highest_avg_plaza = plaza_avg_revenue.index[0]
        
        insights = {
            "total_daily_revenue": float(self.df['txn_amount'].sum()),
            "average_transaction_amount": float(self.df['txn_amount'].mean()),
            "median_transaction_amount": float(self.df['txn_amount'].median()),
            "peak_revenue_hour": int(peak_revenue_hour),
            "peak_hour_revenue": float(hourly_revenue[peak_revenue_hour]),
            "top_revenue_plaza": top_revenue_plaza,
            "top_plaza_revenue": float(plaza_revenue[top_revenue_plaza]),
            "top_plaza_revenue_percentage": float(plaza_revenue[top_revenue_plaza] / plaza_revenue.sum() * 100),
            "highest_average_revenue_plaza": highest_avg_plaza,
            "highest_average_amount": float(plaza_avg_revenue[highest_avg_plaza])
        }
        return insights
    
    def generate_natural_language_insights(self, data_summary: Dict[str, Any]) -> str:
        if self.client:
            try:
                prompt = f"""
                You are a transportation analytics expert. Based on the following toll plaza data insights, 
                generate a comprehensive analysis that highlights key patterns, anomalies, and actionable 
                recommendations for toll plaza operators. Format your response as a bullet-point summary
                followed by paragraphs of detailed analysis. Focus on actionable insights.
                
                DATA INSIGHTS:
                {data_summary}
                """
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "system", "content": "You are a toll plaza analytics expert."},
                              {"role": "user", "content": prompt}],
                    temperature=0.5
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error using OpenAI: {e}")
                return self._generate_fallback_insights(data_summary)
        else:
            return self._generate_fallback_insights(data_summary)
    
    def _generate_fallback_insights(self, data_summary: Dict[str, Any]) -> str:
        traffic = data_summary.get("traffic_insights", {})
        vehicle = data_summary.get("vehicle_insights", {})
        operation = data_summary.get("operational_insights", {})
        revenue = data_summary.get("revenue_insights", {})
        
        insights = f"""
        # Toll Plaza Analytics Insights

        ## Key Highlights
        * Total transactions: {data_summary.get('basic_stats', {}).get('total_transactions', 'N/A')}
        * Total revenue: ₹{data_summary.get('basic_stats', {}).get('total_revenue', 'N/A'):,.2f}
        * Peak hour: {traffic.get('peak_hour', 'N/A')}:00 with {traffic.get('peak_hour_traffic', 'N/A')} transactions
        * Busiest plaza: {traffic.get('busiest_plaza', 'N/A')} with {traffic.get('busiest_plaza_transactions', 'N/A')} transactions
        
        ## Traffic Patterns
        The peak hour ({traffic.get('peak_hour', 'N/A')}:00) has {traffic.get('peak_to_offpeak_ratio', 'N/A'):.1f}x more traffic than the off-peak hour ({traffic.get('off_peak_hour', 'N/A')}:00). 
        Morning traffic is {traffic.get('morning_vs_evening', {}).get('ratio', 'N/A'):.2f}x the evening traffic.
        
        ## Vehicle Distribution
        {vehicle.get('commercial_vehicle_percentage', 'N/A'):.1f}% of vehicles are commercial, with {vehicle.get('top_vehicle_class', 'N/A')} being the most common type.
        The {vehicle.get('top_revenue_vehicle_class', 'N/A')} class generates the most revenue at ₹{vehicle.get('top_revenue_vehicle_class_amount', 'N/A'):,.2f}.
        
        ## Operational Efficiency
        Average processing time is {operation.get('average_processing_time', 'N/A'):.2f} seconds.
        {operation.get('fastest_plaza', 'N/A')} is the most efficient plaza at {operation.get('fastest_plaza_time', 'N/A'):.2f}s.
        
        ## Revenue Insights
        Total daily revenue is ₹{revenue.get('total_daily_revenue', 'N/A'):,.2f} with an average transaction of ₹{revenue.get('average_transaction_amount', 'N/A'):.2f}.
        Peak revenue hour is {revenue.get('peak_revenue_hour', 'N/A')}:00, generating ₹{revenue.get('peak_hour_revenue', 'N/A'):,.2f}.
        
        ## Recommendations
        1. Optimize lane allocation during peak hours.
        2. Consider dynamic pricing to balance traffic.
        3. Improve processing times at less efficient plazas.
        4. Target revenue optimization for {vehicle.get('top_revenue_vehicle_class', 'N/A')} vehicles.
        """
        return insights
    
    def generate_insights_report(self) -> Dict[str, Any]:
        basic_stats = self._get_basic_stats()
        traffic_insights = self._get_traffic_insights()
        vehicle_insights = self._get_vehicle_insights()
        operational_insights = self._get_operational_insights()
        revenue_insights = self._get_revenue_insights()
        
        data_summary = {
            "basic_stats": basic_stats,
            "traffic_insights": traffic_insights,
            "vehicle_insights": vehicle_insights,
            "operational_insights": operational_insights,
            "revenue_insights": revenue_insights
        }
        
        narrative = self.generate_natural_language_insights(data_summary)
        
        report = {
            "summary": data_summary,
            "narrative": narrative,
            "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_date": self.df['initiated_time'].dt.date.min().strftime("%Y-%m-%d")
        }
        
        return report
    
    def generate_plaza_insights(self, plaza_name: str) -> Dict[str, Any]:
        plaza_data = self.df[self.df['merchant_name'] == plaza_name]
        if plaza_data.empty:
            return {"error": f"No data found for plaza {plaza_name}"}
        
        temp_generator = AutomatedInsightsGenerator(plaza_data, None)
        return temp_generator.generate_insights_report()

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

class TollSkippingDetection:
    def __init__(self, data):
        self.df = debug_dataframe(data)
        self.graph = self._build_route_graph()
        self.expected_routes = self._identify_common_routes()
        self.tag_routes = self._extract_vehicle_routes()

    def _build_route_graph(self):
        G = nx.DiGraph()
        for _, group in self.df.groupby('tag_id'):
            plazas = group.sort_values('initiated_time')['merchant_name'].tolist()
            for i in range(len(plazas) - 1):
                edge_data = G.get_edge_data(plazas[i], plazas[i+1], {'weight': 0})
                G.add_edge(plazas[i], plazas[i+1], weight=edge_data['weight'] + 1)
        return G

    def _identify_common_routes(self):
        common_routes = {}
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            common_routes[(source, target)] = list(nx.all_simple_paths(self.graph, source, target))
        return common_routes

    def _extract_vehicle_routes(self):
        vehicle_routes = {}
        for tag_id, group in self.df.groupby('tag_id'):
            sorted_group = group.sort_values('initiated_time')
            vehicle_routes[tag_id] = list(zip(sorted_group['merchant_name'], sorted_group['initiated_time']))
        return vehicle_routes

    def detect_potential_toll_skipping(self):
        incidents = []
        for tag_id, route in self.tag_routes.items():
            for i in range(len(route) - 1):
                source, target = route[i][0], route[i+1][0]
                if (source, target) in self.expected_routes and len(self.expected_routes[(source, target)]) > 2:
                    incidents.append({
                        "tag_id": tag_id,
                        "source_plaza": source,
                        "target_plaza": target,
                        "skipped_tolls": len(self.expected_routes[(source, target)]) - 2,
                        "estimated_loss": (len(self.expected_routes[(source, target)]) - 2) * 50
                    })
        return incidents

# Initialize our systems
lane_optimizer = LaneOptimizationSystem(df)
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
        plaza_list = df['merchant_name'].unique().tolist()
        selected_plaza = st.selectbox("Select a Plaza", plaza_list)
        selected_hour = st.slider("Select Hour (0-23)", 0, 23, 8)
        expected_traffic = st.number_input("Expected Traffic (optional)", min_value=0, value=0)
        if expected_traffic == 0:
            expected_traffic = None
        if st.button("Get Lane Recommendations"):
            recommendations = lane_optimizer.get_lane_recommendations(selected_plaza, selected_hour, expected_traffic)
            st.json(recommendations)
            st.write("Lane Efficiency Chart")
            plaza_eff = lane_optimizer.lane_efficiency[lane_optimizer.lane_efficiency['plaza'] == selected_plaza]
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(plaza_eff['lane'], plaza_eff['efficiency_score'])
            ax.set_xlabel("Lane")
            ax.set_ylabel("Efficiency Score")
            st.pyplot(fig)

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



import streamlit as st
import joblib
import numpy as np
import time
import math
import pandas as pd

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# -------------------------------
# Load Trained Models
# -------------------------------
@st.cache_resource
def load_models():
    try:
        traffic_model = joblib.load("models/traffic_prediction_model.pkl")
        vc_model = joblib.load("models/vehicle_class_prediction_model.pkl")
        anomaly_models, anomaly_scaler = joblib.load("models/traffic_anomaly_models.pkl")
        skip_models, skip_scaler = joblib.load("models/toll_skipping_models.pkl")
        vehicle_classifier = joblib.load("models/vehicle_behavior_classifier.pkl")
        return {
            "traffic_model": traffic_model,
            "vc_model": vc_model,
            "anomaly_models": anomaly_models,
            "anomaly_scaler": anomaly_scaler,
            "skip_models": skip_models,
            "skip_scaler": skip_scaler,
            "vehicle_classifier": vehicle_classifier
        }
    except Exception as e:
        st.warning(f"Error loading models: {e}")
        # For testing purposes, you can set dummy models here.
        return {
            "traffic_model": None,
            "vc_model": None,
            "anomaly_models": {},
            "anomaly_scaler": None,
            "skip_models": {},
            "skip_scaler": None,
            "vehicle_classifier": None
        }

# Load models
models = load_models()


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
        
        # User inputs (simplified)
        hour = st.slider("Hour of Day", 0, 23, 12)
        day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=3)
        is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
        quarter_of_day = hour // 6
        day_part = ["night", "morning", "afternoon", "evening"].index(
            "night" if hour < 6 else "morning" if hour < 12 else "afternoon" if hour < 18 else "evening"
        )
        
        # Compute cyclical time features
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        day_sin = math.sin(2 * math.pi * (["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)) / 7)
        day_cos = math.cos(2 * math.pi * (["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)) / 7)
        
        # Default values for lag/rolling features
        lag_1, lag_2, lag_3, lag_4, lag_5 = 50, 48, 52, 49, 51
        rolling_mean_3, rolling_std_3, rolling_mean_6 = 50, 5, 50
        rolling_max_6, rolling_min_6 = 60, 40
        tx_diff_1, tx_diff_2 = 2, 3
        tx_per_second, avg_amount_per_tx = 0.5, 10.0
        
        submitted = st.form_submit_button("Predict Traffic")
        
        if submitted:
            traffic_model=models['traffic_model']
            if traffic_model is None:
                st.error("Traffic model not available")
            else:
                try:
                    features = pd.DataFrame({
                        'hour': [hour],
                        'day_of_week': [["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week)],
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
                    })
                    
                    prediction = traffic_model.predict(features)
                    st.success(f"Predicted Traffic Volume at selected hour: {prediction[0]:.2f}")
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

    # -------------------------------
    # Vehicle Class Prediction
    # -------------------------------
    st.header("Vehicle Class Prediction")
    with st.form("vehicle_class_form"):
        st.subheader("Enter Vehicle Class Parameters")
        
        hour_vc = st.slider("Hour", 0, 23, 12)
        day_of_week_vc = st.slider("Day of Week", 0, 6, 3)
        is_weekend_vc = st.radio("Is Weekend", [0, 1], index=0)
        day_part_vc = st.slider("Day Part", 0, 3, 1)
        hour_sin_vc = math.sin(2 * math.pi * hour_vc / 24)
        hour_cos_vc = math.cos(2 * math.pi * hour_vc / 24)
        day_sin_vc = math.sin(2 * math.pi * day_of_week_vc / 7)
        day_cos_vc = math.cos(2 * math.pi * day_of_week_vc / 7)
        
        submitted_vc = st.form_submit_button("Predict Vehicle Class")
        
        if submitted_vc:
            vc_model = models['vc_model']
            if vc_model:
                features = pd.DataFrame({
                        'hour': [hour_vc],
                        'day_of_week': [day_of_week_vc],
                        'is_weekend': [is_weekend_vc],
                        'day_part': [day_part_vc],
                        'hour_sin': [hour_sin_vc],
                        'hour_cos': [hour_cos_vc],
                        'day_sin': [day_sin_vc],
                        'day_cos': [day_cos_vc]
                    })
                prediction = vc_model.predict(features)[0]
                vehicle_classes = ['VC4', 'VC20', 'VC5', 'VC10', 'VC12', 'VC7', 'VC13', 'VC11', 'VC14', 'VC9', 'VC8', 'VC15', 'VC16', 'VC6']
                prediction_df = pd.DataFrame({'Vehicle Code': vehicle_classes, 'Predicted Volume': prediction})
                st.write(prediction_df)
            else:
                st.error("Vehicle class model not available")

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
            if models["anomaly_scaler"] is None or not models["anomaly_models"]:
                st.error("Anomaly models not available")
            else:
                try:
                    features = np.array([[
                        transaction_count, hour_anomaly, day_of_week_anomaly,
                        is_weekend_anomaly, month_anomaly, day_anomaly
                    ]])
                    
                    # Scale the features using the pre-fitted scaler
                    X_scaled = models["anomaly_scaler"].transform(features)
                    
                    # Get models
                    if_model, _ = models["anomaly_models"]['isolation_forest']
                    dbscan_model, _ = models["anomaly_models"]['dbscan']
                    
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
  


# -------------------------------
# Toll Skipping Detection
# -------------------------------
with tab4:
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
            if models["skip_scaler"] is None or not models["skip_models"]:
                st.error("Toll skipping models not available")
            else:
                try:
                    features = np.array([[
                        total_trips, unique_plazas, trips_per_hour,
                        avg_amount_per_trip, plaza_to_trip_ratio
                    ]])
                    features_scaled = models["skip_scaler"].transform(features)
                    # Using KMeans clustering
                    kmeans = models["skip_models"]['kmeans']
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
            if models["vehicle_classifier"] is None:
                st.error("Vehicle behavior classifier not available")
            else:
                try:
                    features = np.array([[
                        total_trips_vb, unique_plazas_vb, trips_per_hour_vb,
                        avg_amount_per_trip_vb, plaza_to_trip_ratio_vb
                    ]])
                    prediction = models["vehicle_classifier"].predict(features)
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

# Add a footer
st.markdown("---")