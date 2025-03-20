# =============================================================================
# 2. LANE OPTIMIZATION SYSTEM WITH OPENAI ENHANCEMENT
# =============================================================================
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any
from sklearn.preprocessing import MinMaxScaler
import openai
from datetime import datetime, timedelta
import os

class EnhancedLaneOptimizationSystem:
    def __init__(self, data, openai_api_key=None):
        self.df = data
        self.plaza_lanes = self._get_plaza_lanes()
        self.vehicle_types = sorted(self.df['vehicle_class_code'].unique())
        self.lane_efficiency = self._calculate_lane_efficiency()
        
        # Configure OpenAI client
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if self.openai_api_key:
            self.client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            print("Warning: No OpenAI API key provided. Advanced recommendations will be limited.")
            self.client = None
        
        # Create historical patterns cache for faster lookups
        self.historical_patterns = self._analyze_historical_patterns()
        
    def _get_plaza_lanes(self) -> Dict[str, List[str]]:
        plaza_lanes = {}
        
        for plaza in self.df['merchant_name'].unique():
            lanes = self.df[self.df['merchant_name'] == plaza]['lane'].unique()
            plaza_lanes[plaza] = sorted(lanes, key=lambda x: (isinstance(x, str), x))
        return plaza_lanes
        
    def _calculate_lane_efficiency(self) -> pd.DataFrame:
        """Calculate efficiency of each lane with enhanced metrics."""
        lane_stats = self.df.groupby(['merchant_name', 'lane']).agg({
            'inn_rr_time_sec': ['mean', 'count', 'std'],
            'txn_amount': ['mean']
        }).reset_index()
        
        # Flatten multi-level columns
        lane_stats.columns = ['plaza', 'lane', 'avg_processing_time', 'volume', 'time_std', 'avg_revenue']
        
        # Add day of week and hour analysis for each lane
        time_patterns = self.df.groupby(['merchant_name', 'lane', 
                                         self.df['initiated_time'].dt.day_name(),
                                         self.df['initiated_time'].dt.hour])['inn_rr_time_sec'].mean()
        
        # Find busiest periods
        lane_stats['peak_periods'] = lane_stats.apply(
            lambda x: self._find_peak_periods(x['plaza'], x['lane']), axis=1
        )
        
        # Normalize and compute a weighted efficiency score (lower is better)
        scaler = MinMaxScaler()
        lane_stats[['norm_time', 'norm_volume', 'norm_revenue']] = scaler.fit_transform(
            lane_stats[['avg_processing_time', 'volume', 'avg_revenue']])
        
        # Enhanced efficiency score: considers processing time, volume, and revenue
        lane_stats['efficiency_score'] = (
            0.5 * lane_stats['norm_time'] - 
            0.3 * lane_stats['norm_volume'] - 
            0.2 * lane_stats['norm_revenue']
        )
        
        # Calculate reliability score (lower std deviation = more reliable)
        max_std = lane_stats['time_std'].max()
        if max_std > 0:
            lane_stats['reliability_score'] = 1 - (lane_stats['time_std'] / max_std)
        else:
            lane_stats['reliability_score'] = 1.0
            
        return lane_stats
    
    def _find_peak_periods(self, plaza, lane):
        """Find peak busy periods for a specific lane."""
        lane_data = self.df[(self.df['merchant_name'] == plaza) & 
                             (self.df['lane'] == lane)]
        
        if lane_data.empty:
            return []
            
        # Group by day of week and hour
        hourly_counts = lane_data.groupby([
            lane_data['initiated_time'].dt.day_name(),
            lane_data['initiated_time'].dt.hour
        ]).size()
        
        # Get top 3 busiest periods
        if len(hourly_counts) > 0:
            top_periods = hourly_counts.nlargest(3)
            return [{"day": day, "hour": hour, "volume": int(count)} 
                    for (day, hour), count in top_periods.items()]
        return []
    
    def _analyze_historical_patterns(self):
        """Create a cache of historical traffic patterns for faster lookups."""
        patterns = {}
        
        for plaza in self.df['merchant_name'].unique():
            plaza_data = self.df[self.df['merchant_name'] == plaza]
            patterns[plaza] = {}
            
            # Analyze by hour of day
            for hour in range(24):
                hour_data = plaza_data[plaza_data['initiated_time'].dt.hour == hour]
                if not hour_data.empty:
                    patterns[plaza][hour] = {
                        'avg_traffic': len(hour_data),
                        'vehicle_distribution': dict(hour_data['vehicle_class_code'].value_counts(normalize=True)),
                        'avg_processing_time': float(hour_data['inn_rr_time_sec'].mean()),
                        'commercial_ratio': float(hour_data['vehicle_comvehicle'].replace(
                            {'F': 0, 'T': 1}).mean()) if 'vehicle_comvehicle' in hour_data.columns else 0.3
                    }
        
        return patterns
    
    def get_lane_recommendations(self, plaza: str, hour: int, expected_traffic: int = None,
                                use_ai: bool = True) -> Dict[str, Any]:
        """
        Get lane recommendations with enhanced AI-based optimization when available.
        
        Args:
            plaza: The toll plaza name
            hour: Hour of day (0-23)
            expected_traffic: Expected traffic volume (if None, uses historical data)
            use_ai: Whether to use OpenAI for enhanced recommendations
        """
        # First check if we have historical data for this plaza and hour
        if plaza not in self.historical_patterns or hour not in self.historical_patterns[plaza]:
            return {"error": f"No historical data available for plaza {plaza} at hour {hour}"}
        
        # Get historical pattern data
        historical = self.historical_patterns[plaza][hour]
        
        # Get available lanes
        available_lanes = self.plaza_lanes.get(plaza, [])
        if not available_lanes:
            return {"error": f"No lanes data available for plaza {plaza}"}
        
        # Use historical or provided expected traffic
        if expected_traffic is None:
            expected_traffic = historical['avg_traffic']
        
        # Get plaza efficiency data
        plaza_efficiency = self.lane_efficiency[self.lane_efficiency['plaza'] == plaza].copy()
        if plaza_efficiency.empty:
            return {"error": f"No efficiency data available for plaza {plaza}"}
        
        # Calculate recommended number of lanes based on traffic
        recommended_lanes_count = max(2, int(np.ceil(expected_traffic / 100)))
        recommended_lanes_count = min(recommended_lanes_count, len(available_lanes))
        
        # Get AI-enhanced recommendations if available and requested
        if use_ai and self.client:
            ai_recommendations = self._get_ai_lane_recommendations(
                plaza, hour, expected_traffic, recommended_lanes_count, 
                historical, plaza_efficiency, available_lanes
            )
            if ai_recommendations and "error" not in ai_recommendations:
                return ai_recommendations
        
        # Fall back to standard algorithm if AI is not available or fails
        # Select most efficient lanes
        best_lanes = plaza_efficiency.sort_values(['efficiency_score', 'reliability_score'], 
                                                ascending=[True, False]).head(recommended_lanes_count)
        
        # Calculate commercial lanes based on historical ratio
        commercial_ratio = historical['commercial_ratio']
        commercial_lanes = max(1, int(np.round(commercial_ratio * recommended_lanes_count)))
        
        # Allocate lanes
        recommended_lanes = []
        for i, (_, lane_data) in enumerate(best_lanes.iterrows()):
            lane_role = "Commercial" if i < commercial_lanes else "Non-commercial"
            recommended_lanes.append({
                "lane": lane_data['lane'],
                "role": lane_role,
                "expected_volume": int(expected_traffic / recommended_lanes_count),
                "processing_time": float(lane_data['avg_processing_time']),
                "reliability": float(lane_data['reliability_score']),
                "peak_periods": lane_data['peak_periods']
            })
        
        # Create recommendations output
        recommendations = {
            "plaza": plaza,
            "hour": hour,
            "expected_traffic": expected_traffic,
            "lanes_needed": recommended_lanes_count,
            "commercial_ratio": float(commercial_ratio),
            "recommended_lanes": recommended_lanes,
            "vehicle_distribution": historical['vehicle_distribution'],
            "historical_average_processing_time": float(historical['avg_processing_time']),
            "ai_enhanced": False,
            "recommendation_timestamp": datetime.now().isoformat()
        }
        
        return recommendations
    
    # def _get_ai_lane_recommendations(self, plaza, hour, expected_traffic, lanes_needed, 
    #                                 historical, plaza_efficiency, available_lanes):
    #     """Use OpenAI to generate enhanced lane recommendations."""
    #     try:
    #         # Format the context for OpenAI to analyze
    #         context = {
    #             "plaza": plaza,
    #             "hour": hour,
    #             "expected_traffic": expected_traffic,
    #             "historical_avg_traffic": historical['avg_traffic'],
    #             "commercial_ratio": historical['commercial_ratio'],
    #             "available_lanes": len(available_lanes),
    #             "lane_efficiency_data": plaza_efficiency.to_dict(orient='records'),
    #             "vehicle_distribution": historical['vehicle_distribution'],
    #             "avg_processing_time": historical['avg_processing_time']
    #         }
            
    #         # Get AI recommendation
    #         response = self.client.chat.completions.create(
    #             model="gpt-4o",
    #             response_format={"type": "json_object"},
    #             messages=[
    #                 {"role": "system", "content": """
    #                 You are an AI traffic optimization assistant specializing in toll plaza management.
    #                 Analyze the provided toll plaza data and recommend the optimal lane allocation strategy.
    #                 Your response should be a valid JSON object containing recommended lane allocations
    #                 and expected performance metrics. Consider vehicle types, commercial vs non-commercial
    #                 allocations, and processing efficiency. Apply traffic flow optimization principles.
    #                 """},
    #                 {"role": "user", "content": f"""
    #                 Provide lane allocation recommendations for the following toll plaza situation:
    #                 {json.dumps(context, indent=2)}
                    
    #                 Return a JSON with these fields:
    #                 1. recommended_lanes: array of lane objects with lane id, role (Commercial/Non-commercial), expected_volume, and processing_time
    #                 2. expected_wait_time: estimated average wait time in seconds
    #                 3. optimization_notes: key insights about this allocation strategy
                    
    #                 Base your recommendations on efficiency scores (lower is better), reliability scores (higher is better),
    #                 and intelligent distribution of commercial vs non-commercial traffic.
    #                 """} 
    #             ]
    #         )
            
    #         # Extract and parse recommendation
    #         ai_recommendation = json.loads(response.choices[0].message.content)
            
    #         # Create the complete recommendation to return
    #         recommendation = {
    #             "plaza": plaza,
    #             "hour": hour,
    #             "expected_traffic": expected_traffic,
    #             "lanes_needed": lanes_needed,
    #             "commercial_ratio": float(historical['commercial_ratio']),
    #             "recommended_lanes": ai_recommendation.get("recommended_lanes", []),
    #             "vehicle_distribution": historical['vehicle_distribution'],
    #             "historical_average_processing_time": float(historical['avg_processing_time']),
    #             "ai_enhanced": True,
    #             "expected_wait_time": ai_recommendation.get("expected_wait_time"),
    #             "optimization_notes": ai_recommendation.get("optimization_notes", []),
    #             "recommendation_timestamp": datetime.now().isoformat()
    #         }
            
    #         return recommendation
            
    #     except Exception as e:
    #         print(f"Error getting AI lane recommendations: {str(e)}")
    #         return None
    
    def _get_ai_lane_recommendations(self, plaza, hour, expected_traffic, lanes_needed, 
                                    historical, plaza_efficiency, available_lanes):
        """Use OpenAI to generate enhanced lane recommendations."""
        try:
            # Format the context for OpenAI to analyze
            context = {
                "plaza": plaza,
                "hour": hour,
                "expected_traffic": expected_traffic,
                "historical_avg_traffic": historical['avg_traffic'],
                "commercial_ratio": historical['commercial_ratio'],
                "available_lanes": len(available_lanes),
                "lane_efficiency_data": plaza_efficiency.to_dict(orient='records'),
                "vehicle_distribution": historical['vehicle_distribution'],
                "avg_processing_time": historical['avg_processing_time']
            }
            
            # Get AI recommendation
            response = self.client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": """
                    You are an AI traffic optimization assistant specializing in toll plaza management.
                    Analyze the provided toll plaza data and recommend the optimal lane allocation strategy.
                    Your response should be a valid JSON object containing recommended lane allocations
                    and expected performance metrics. Consider vehicle types, commercial vs non-commercial
                    allocations, and processing efficiency. Apply traffic flow optimization principles.
                    """},
                    {"role": "user", "content": f"""
                    Provide lane allocation recommendations for the following toll plaza situation:
                    {json.dumps(context, indent=2)}
                    
                    Return a JSON with these fields:
                    1. recommended_lanes: array of lane objects with MANDATORY fields 'lane' (lane identifier), 
                       'role' (Commercial/Non-commercial), 'expected_volume', and 'processing_time'
                    2. expected_wait_time: estimated average wait time in seconds
                    3. optimization_notes: key insights about this allocation strategy
                    
                    Base your recommendations on efficiency scores (lower is better), reliability scores (higher is better),
                    and intelligent distribution of commercial vs non-commercial traffic.
                    """} 
                ]
            )
            
            # Extract and parse recommendation
            ai_recommendation = json.loads(response.choices[0].message.content)
            
            # Normalize the lane recommendation data structure to ensure consistent keys
            normalized_lanes = []
            for lane_data in ai_recommendation.get("recommended_lanes", []):
                # Ensure the required keys are present with proper naming
                normalized_lane = {
                    "lane": lane_data.get("lane") or lane_data.get("lane_id") or lane_data.get("id", "unknown"),
                    "role": lane_data.get("role", "Unknown"),
                    "expected_volume": lane_data.get("expected_volume", 0),
                    "processing_time": lane_data.get("processing_time", 0.0),
                    "reliability": lane_data.get("reliability", 1.0),
                    "peak_periods": lane_data.get("peak_periods", [])
                }
                normalized_lanes.append(normalized_lane)
            
            # Create the complete recommendation to return
            recommendation = {
                "plaza": plaza,
                "hour": hour,
                "expected_traffic": expected_traffic,
                "lanes_needed": lanes_needed,
                "commercial_ratio": float(historical['commercial_ratio']),
                "recommended_lanes": normalized_lanes,  # Use normalized lanes instead of direct AI response
                "vehicle_distribution": historical['vehicle_distribution'],
                "historical_average_processing_time": float(historical['avg_processing_time']),
                "ai_enhanced": True,
                "expected_wait_time": ai_recommendation.get("expected_wait_time"),
                "optimization_notes": ai_recommendation.get("optimization_notes", []),
                "recommendation_timestamp": datetime.now().isoformat()
            }
            
            return recommendation
            
        except Exception as e:
            print(f"Error getting AI lane recommendations: {str(e)}")
            return None
        
    def get_dynamic_pricing_recommendations(self, plaza: str, hour: int, 
                                         use_ai: bool = True) -> Dict[str, Any]:
        """
        Get AI-enhanced dynamic pricing recommendations for a specific plaza and time.
        """
        # Check if we have historical data
        if plaza not in self.historical_patterns or hour not in self.historical_patterns[plaza]:
            return {"error": f"No historical data available for plaza {plaza} at hour {hour}"}
        
        # Get historical pattern data
        historical = self.historical_patterns[plaza][hour]
        
        # Get current pricing from historical data
        plaza_data = self.df[self.df['merchant_name'] == plaza]
        hourly_data = plaza_data[plaza_data['initiated_time'].dt.hour == hour]
        
        if hourly_data.empty:
            return {"error": f"No transaction data available for plaza {plaza} at hour {hour}"}
        
        # Calculate congestion level
        max_hourly_traffic = plaza_data.groupby(plaza_data['initiated_time'].dt.hour).size().max()
        current_hourly_traffic = len(hourly_data)
        congestion_level = current_hourly_traffic / max_hourly_traffic if max_hourly_traffic > 0 else 0
        
        # Get current pricing by vehicle type
        current_pricing = {}
        for vc in self.vehicle_types:
            vc_data = hourly_data[hourly_data['vehicle_class_code'] == vc]
            if not vc_data.empty:
                current_pricing[vc] = float(vc_data['txn_amount'].median())
        
        # Get AI-enhanced pricing if available and requested
        if use_ai and self.client:
            ai_pricing = self._get_ai_pricing_recommendations(
                plaza, hour, congestion_level, current_pricing, historical
            )
            if ai_pricing and "error" not in ai_pricing:
                return ai_pricing
                
        # Fall back to standard algorithm if AI is not available or fails
        recommended_pricing = {}
        for vc, base_price in current_pricing.items():
            if congestion_level > 0.7:  # High congestion: increase up to 20%
                factor = 1.0 + (congestion_level - 0.7) * (0.2 / 0.3)
            elif congestion_level < 0.3:  # Low congestion: decrease up to 10%
                factor = 1.0 - (0.3 - congestion_level) * (0.1 / 0.3)
            else:
                factor = 1.0
            recommended_pricing[vc] = round(base_price * factor, 2)
        
        # Calculate expected impact
        if sum(current_pricing.values()) > 0:
            revenue_change = (sum(recommended_pricing.values()) - sum(current_pricing.values())) / sum(current_pricing.values()) * 100
        else:
            revenue_change = 0
            
        traffic_reduction = 5 * (congestion_level - 0.5) if congestion_level > 0.5 else 0
        
        # Create recommendations output
        recommendations = {
            "plaza": plaza,
            "hour": hour,
            "congestion_level": float(congestion_level),
            "current_pricing": current_pricing,
            "recommended_pricing": recommended_pricing,
            "expected_impact": {
                "revenue_change": f"{revenue_change:.2f}%",
                "expected_traffic_reduction": f"{traffic_reduction:.2f}%"
            },
            "ai_enhanced": False,
            "recommendation_timestamp": datetime.now().isoformat()
        }
        
        return recommendations
    
    def _get_ai_pricing_recommendations(self, plaza, hour, congestion_level, current_pricing, historical):
        """Use OpenAI to generate enhanced dynamic pricing recommendations."""
        try:
            # Get time context (day of week, etc.)
            now = datetime.now()
            day_of_week = now.strftime("%A")
            
            # Prepare context for the AI
            context = {
                "plaza": plaza,
                "hour": hour,
                "day_of_week": day_of_week,
                "congestion_level": congestion_level,
                "current_pricing": current_pricing,
                "historical_data": {
                    "avg_traffic": historical['avg_traffic'],
                    "vehicle_distribution": historical['vehicle_distribution'],
                },
                "adjacent_hours_context": self._get_adjacent_hours_data(plaza, hour)
            }
            
            # Get AI recommendation
            response = self.client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": """
                    You are an AI pricing optimization specialist for toll plaza management.
                    Analyze the provided data and recommend optimal dynamic pricing strategies.
                    Your response should be a valid JSON object with recommended prices and 
                    expected impact analysis. Consider congestion levels, traffic patterns,
                    vehicle types, and time-of-day factors. Apply economic principles of
                    demand elasticity and peak-load pricing.
                    """},
                    {"role": "user", "content": f"""
                    Provide dynamic pricing recommendations for the following toll plaza situation:
                    {json.dumps(context, indent=2)}
                    
                    Return a JSON with these fields:
                    1. recommended_pricing: object with vehicle class codes as keys and recommended prices as values
                    2. expected_impact: object with revenue_change (%) and expected_traffic_reduction (%) 
                    3. pricing_strategy: string describing the recommended strategy
                    4. optimization_notes: array of key insights about this pricing strategy
                    
                    The pricing should optimize for both revenue and traffic flow, considering the
                    time of day, congestion level, and vehicle types.
                    """} 
                ]
            )
            
            # Extract and parse recommendation
            ai_recommendation = json.loads(response.choices[0].message.content)
            
            # Format the expected impact values
            expected_impact = ai_recommendation.get("expected_impact", {})
            if isinstance(expected_impact, dict):
                for key, value in expected_impact.items():
                    if isinstance(value, (int, float)):
                        expected_impact[key] = f"{value:.2f}%"
            
            # Create the complete recommendation to return
            recommendation = {
                "plaza": plaza,
                "hour": hour,
                "congestion_level": float(congestion_level),
                "current_pricing": current_pricing,
                "recommended_pricing": ai_recommendation.get("recommended_pricing", {}),
                "expected_impact": expected_impact,
                "pricing_strategy": ai_recommendation.get("pricing_strategy", ""),
                "optimization_notes": ai_recommendation.get("optimization_notes", []),
                "ai_enhanced": True,
                "recommendation_timestamp": datetime.now().isoformat()
            }
            
            return recommendation
            
        except Exception as e:
            print(f"Error getting AI pricing recommendations: {str(e)}")
            return None
    
    def _get_adjacent_hours_data(self, plaza, hour):
        """Get data for hours before and after the target hour for context."""
        adjacent_data = {}
        
        # Previous hour
        prev_hour = (hour - 1) % 24
        if plaza in self.historical_patterns and prev_hour in self.historical_patterns[plaza]:
            adjacent_data["previous_hour"] = {
                "hour": prev_hour,
                "avg_traffic": self.historical_patterns[plaza][prev_hour]['avg_traffic']
            }
        
        # Next hour
        next_hour = (hour + 1) % 24
        if plaza in self.historical_patterns and next_hour in self.historical_patterns[plaza]:
            adjacent_data["next_hour"] = {
                "hour": next_hour,
                "avg_traffic": self.historical_patterns[plaza][next_hour]['avg_traffic']
            }
            
        return adjacent_data
    
    def visualize_lane_recommendations(self, plaza: str, hour: int) -> Dict[str, Any]:
        """Create visualization data for the lane recommendations."""
        recommendations = self.get_lane_recommendations(plaza, hour)
        if "error" in recommendations:
            return {"error": recommendations["error"]}
        
        # Create enhanced visualization data
        viz_data = {
            "plaza": plaza,
            "hour": hour,
            "lane_allocation": {
                "lanes": [lane["lane"] for lane in recommendations["recommended_lanes"]],
                "roles": [lane["role"] for lane in recommendations["recommended_lanes"]],
                "volumes": [lane["expected_volume"] for lane in recommendations["recommended_lanes"]],
                "processing_times": [lane["processing_time"] for lane in recommendations["recommended_lanes"]]
            },
            "vehicle_distribution": recommendations["vehicle_distribution"],
            "ai_enhanced": recommendations.get("ai_enhanced", False)
        }
        
        # Add historical comparison
        if plaza in self.historical_patterns and hour in self.historical_patterns[plaza]:
            historical = self.historical_patterns[plaza][hour]
            viz_data["historical_comparison"] = {
                "avg_traffic": historical['avg_traffic'],
                "expected_traffic": recommendations["expected_traffic"],
                "traffic_difference_percent": ((recommendations["expected_traffic"] - historical['avg_traffic']) / 
                                              historical['avg_traffic'] * 100) if historical['avg_traffic'] > 0 else 0
            }
        
        return viz_data
    
    def get_optimization_insights(self, plaza: str) -> Dict[str, Any]:
        """Generate AI insights about optimization opportunities for a plaza."""
        if not self.client:
            return {"error": "OpenAI API key not configured for insights generation"}
            
        try:
            plaza_data = self.df[self.df['merchant_name'] == plaza]
            if plaza_data.empty:
                return {"error": f"No data available for plaza {plaza}"}
                
            # Create a summary of plaza operations
            summary = {
                "plaza": plaza,
                "total_transactions": len(plaza_data),
                "lanes": self.plaza_lanes.get(plaza, []),
                "avg_processing_time": float(plaza_data['inn_rr_time_sec'].mean()),
                "peak_hours": self._get_peak_hours(plaza_data),
                "vehicle_type_distribution": dict(plaza_data['vehicle_class_code'].value_counts(normalize=True)),
                "commercial_ratio": float(plaza_data['vehicle_comvehicle'].replace(
                    {'F': 0, 'T': 1}).mean()) if 'vehicle_comvehicle' in plaza_data.columns else 0.3
            }
            
            # Get lane efficiency data
            plaza_efficiency = self.lane_efficiency[self.lane_efficiency['plaza'] == plaza].copy()
            lane_data = plaza_efficiency.sort_values('efficiency_score').to_dict(orient='records')
            
            # Get AI insights
            response = self.client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": """
                    You are an AI traffic optimization analyst specializing in toll plaza operations.
                    Analyze the provided toll plaza data and identify optimization opportunities.
                    Focus on traffic flow improvements, lane efficiency, staffing optimization,
                    and revenue enhancement strategies. 
                    """},
                    {"role": "user", "content": f"""
                    Analyze this toll plaza data and provide optimization insights:
                    
                    Plaza Summary:
                    {json.dumps(summary, indent=2)}
                    
                    Lane Efficiency Data:
                    {json.dumps(lane_data, indent=2)}
                    
                    Return a JSON with these fields:
                    1. key_findings: array of most important observations
                    2. optimization_opportunities: array of specific actionable recommendations
                    3. projected_benefits: object with expected improvements in processing_time, revenue, and customer_satisfaction
                    4. implementation_difficulty: ranking of each optimization opportunity (1-5 scale, 5 being most difficult)
                    """} 
                ]
            )
            
            # Extract and parse insights
            insights = json.loads(response.choices[0].message.content)
            
            # Add metadata
            insights["plaza"] = plaza
            insights["analysis_timestamp"] = datetime.now().isoformat()
            insights["data_analyzed"] = {
                "transactions": len(plaza_data),
                "time_period": f"{plaza_data['initiated_time'].min()} to {plaza_data['initiated_time'].max()}"
            }
            
            return insights
            
        except Exception as e:
            print(f"Error generating AI insights: {str(e)}")
            return {"error": f"Failed to generate insights: {str(e)}"}
    
    def _get_peak_hours(self, plaza_data):
        """Identify peak hours for a plaza."""
        hourly_counts = plaza_data.groupby(plaza_data['initiated_time'].dt.hour).size()
        top_hours = hourly_counts.nlargest(3)
        return [{"hour": int(hour), "volume": int(count)} for hour, count in top_hours.items()]
    
    def simulate_optimization_impact(self, plaza: str, recommendations: Dict) -> Dict[str, Any]:
        """Simulate the impact of implementing lane and pricing recommendations."""
        if not self.client:
            return {"error": "OpenAI API key not configured for simulation"}
            
        try:
            # Extract lane and pricing recommendations
            lane_recommendations = recommendations.get("lane_recommendations", {})
            pricing_recommendations = recommendations.get("pricing_recommendations", {})
            
            # Create simulation context
            context = {
                "plaza": plaza,
                "current_operations": {
                    "avg_processing_time": float(self.df[self.df['merchant_name'] == plaza]['inn_rr_time_sec'].mean()),
                    "lanes": len(self.plaza_lanes.get(plaza, [])),
                    "pricing": pricing_recommendations.get("current_pricing", {})
                },
                "recommended_operations": {
                    "lane_allocation": lane_recommendations.get("recommended_lanes", []),
                    "pricing": pricing_recommendations.get("recommended_pricing", {})
                },
                "traffic_volume": lane_recommendations.get("expected_traffic", 0),
                "congestion_level": pricing_recommendations.get("congestion_level", 0)
            }
            
            # Get AI simulation
            response = self.client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": """
                    You are an AI traffic simulation expert specializing in toll plaza operations.
                    Simulate the impact of implementing the recommended optimizations.
                    Your simulation should estimate improvements in processing time, revenue,
                    customer satisfaction, and overall throughput based on traffic engineering principles.
                    """},
                    {"role": "user", "content": f"""
                    Simulate the impact of implementing these recommendations for {plaza} toll plaza:
                    {json.dumps(context, indent=2)}
                    
                    Return a JSON with these fields:
                    1. estimated_improvements: object with percentage improvements in various metrics
                    2. estimated_roi: return on investment calculation 
                    3. implementation_timeline: estimated time to realize full benefits
                    4. risk_factors: potential issues that could impact success
                    5. success_factors: key elements needed for successful implementation
                    """} 
                ]
            )
            
            # Extract and parse simulation
            simulation = json.loads(response.choices[0].message.content)
            
            # Add metadata
            simulation["plaza"] = plaza
            simulation["simulation_timestamp"] = datetime.now().isoformat()
            simulation["simulation_type"] = "ai_predictive_model"
            
            return simulation
            
        except Exception as e:
            print(f"Error simulating optimization impact: {str(e)}")
            return {"error": f"Failed to simulate impact: {str(e)}"}

# Example usage:
# Initialize enhanced lane optimization system
