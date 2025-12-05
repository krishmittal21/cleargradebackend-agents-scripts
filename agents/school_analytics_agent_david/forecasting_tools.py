"""
Improved Forecasting Tools with API Integration
Automatically fetches fee and expense data from TIAF API
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import warnings
import json
from datetime import datetime, timedelta
from scipy import stats

from tiaf_api_client import TIAFApiClient

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

try:
    import pmdarima as pm
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def convert_to_native_types(obj):
    """Recursively convert numpy/pandas types to native Python types."""
    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_native_types(obj.tolist())
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return convert_to_native_types(obj.to_dict())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


def parse_fee_report_data(api_response: Dict[str, Any]) -> List[float]:
    """
    Extract time series data from fee report API response.
    
    Actual API format:
    {
        "status": "success",
        "from": "2025-1-1",
        "to": "2025-12-03",
        "total_days": 197,
        "grand_total": 22887620,
        "data": [
            {
                "Date": "2025-02-11",
                "Admission Fee": 500,
                "Tuition Fee": 0,
                ...
                "Day Total": 500
            },
            ...
        ]
    }
    """
    # Check if API call was successful
    if not api_response.get("success"):
        # Also check for "status": "success"
        if api_response.get("status") != "success":
            logger.error(f"API response not successful: {api_response}")
            return []
    
    data = api_response.get("data", [])
    
    if not data:
        logger.error("No data found in API response")
        return []
    
    # Extract daily totals
    amounts = []
    for item in data:
        if isinstance(item, dict):
            # Try to get "Day Total" first (preferred for fee reports)
            amount = (
                item.get("Day Total") or
                item.get("day_total") or
                item.get("total") or
                item.get("amount") or
                item.get("collection") or
                0
            )
            amounts.append(float(amount))
        elif isinstance(item, (int, float)):
            amounts.append(float(item))
    
    logger.info(f"Parsed {len(amounts)} fee data points from API")
    return amounts


def parse_expense_report_data(api_response: Dict[str, Any]) -> List[float]:
    """
    Extract time series data from expense report API response.
    
    Actual API format (similar to fee report):
    {
        "status": "success",
        "from": "2025-1-1",
        "to": "2025-12-03",
        "total_days": 197,
        "grand_total": 5500000,
        "data": [
            {
                "Date": "2025-02-11",
                "Salary": 50000,
                "Utilities": 5000,
                ...
                "Day Total": 55000
            },
            ...
        ]
    }
    """
    # Check if API call was successful
    if not api_response.get("success"):
        # Also check for "status": "success"
        if api_response.get("status") != "success":
            logger.error(f"API response not successful: {api_response}")
            return []
    
    data = api_response.get("data", [])
    
    if not data:
        logger.error("No data found in API response")
        return []
    
    # Extract daily totals
    amounts = []
    for item in data:
        if isinstance(item, dict):
            # Try to get "Day Total" first (preferred for expense reports)
            amount = (
                item.get("Day Total") or
                item.get("day_total") or
                item.get("total") or
                item.get("amount") or
                item.get("expense") or
                0
            )
            amounts.append(float(amount))
        elif isinstance(item, (int, float)):
            amounts.append(float(item))
    
    logger.info(f"Parsed {len(amounts)} expense data points from API")
    return amounts


def aggregate_data(data: List[float], aggregation: str = 'weekly') -> Tuple[List[float], str]:
    """Aggregate high-frequency volatile data into smoother periods."""
    if aggregation == 'auto':
        if len(data) > 365:
            aggregation = 'monthly'
        elif len(data) > 90:
            aggregation = 'weekly'
        else:
            return data, 'daily'
    
    data_array = np.array(data)
    
    if aggregation == 'weekly':
        n_weeks = len(data_array) // 7
        if n_weeks < 4:
            return data, 'daily'
        aggregated = [
            float(np.sum(data_array[i*7:(i+1)*7])) 
            for i in range(n_weeks)
        ]
        return aggregated, 'weekly'
    
    elif aggregation == 'monthly':
        n_months = len(data_array) // 30
        if n_months < 3:
            return data, 'daily'
        aggregated = [
            float(np.sum(data_array[i*30:(i+1)*30])) 
            for i in range(n_months)
        ]
        return aggregated, 'monthly'
    
    return data, 'daily'


def calculate_robust_volatility(data: np.ndarray, method: str = 'iqr') -> float:
    """Calculate volatility that's robust to outliers and zeros."""
    non_zero_data = data[data > 0]
    
    if len(non_zero_data) < 2:
        return 0.2
    
    returns = np.diff(non_zero_data) / non_zero_data[:-1]
    returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0:
        return 0.2
    
    if method == 'iqr':
        q75, q25 = np.percentile(returns, [75, 25])
        iqr = q75 - q25
        volatility = iqr / 1.349
    elif method == 'mad':
        median = np.median(returns)
        mad = np.median(np.abs(returns - median))
        volatility = mad * 1.4826
    elif method == 'winsorized':
        lower, upper = np.percentile(returns, [5, 95])
        winsorized = np.clip(returns, lower, upper)
        volatility = np.std(winsorized)
    else:
        volatility = np.std(returns)
    
    return max(float(volatility), 0.05)


def smooth_data(data: List[float], method: str = 'moving_average', window: int = 7) -> List[float]:
    """Smooth high-volatility data to reduce noise."""
    if method == 'none' or len(data) < window:
        return data
    
    data_series = pd.Series(data)
    
    if method == 'moving_average':
        smoothed = data_series.rolling(window=window, min_periods=1).mean()
    elif method == 'exponential':
        smoothed = data_series.ewm(span=window, adjust=False).mean()
    else:
        return data
    
    return smoothed.tolist()


async def monte_carlo_simulation_tool(params_json: str) -> str:
    """
    Monte Carlo simulation with robust volatility estimation.
    
    Input format (JSON string):
    {
        "initial_value": 100000,
        "expected_return": 0.08,
        "volatility": 0.2,
        "time_horizon": 12,
        "num_simulations": 1000,
        "use_robust_volatility": true,
        "historical_data": [50000, 0, 0, 80000, ...]
    }
    """
    try:
        params = json.loads(params_json)
        
        initial_value = float(params.get("initial_value", 100000))
        expected_return = float(params.get("expected_return", 0.08))
        volatility = float(params.get("volatility", 0.2))
        time_horizon = int(params.get("time_horizon", 12))
        num_simulations = int(params.get("num_simulations", 1000))
        use_robust = params.get("use_robust_volatility", False)
        historical_data = params.get("historical_data", None)
        
        if initial_value <= 0:
            return json.dumps({"error": "Initial value must be positive", "success": False})
        if time_horizon <= 0:
            return json.dumps({"error": "Time horizon must be positive", "success": False})
        if num_simulations < 100:
            return json.dumps({"error": "Need at least 100 simulations", "success": False})
        
        if use_robust and historical_data and len(historical_data) > 10:
            volatility = calculate_robust_volatility(
                np.array(historical_data), 
                method='iqr'
            )
            logger.info(f"Using robust volatility: {volatility:.4f}")
        
        volatility = np.clip(volatility, 0.01, 2.0)
        
        logger.info(f"Monte Carlo: {num_simulations} paths, vol={volatility:.4f}")
        
        np.random.seed(42)
        dt = 1.0
        drift = (expected_return - 0.5 * volatility ** 2) * dt
        diffusion = volatility * np.sqrt(dt)
        
        Z = np.random.standard_normal(size=(num_simulations, time_horizon))
        returns = drift + diffusion * Z
        cumulative_returns = np.cumsum(returns, axis=1)
        price_paths = initial_value * np.exp(cumulative_returns)
        
        price_paths_full = np.column_stack([
            np.full(num_simulations, initial_value),
            price_paths
        ])
        
        final_values = price_paths[:, -1]
        
        mean_final = float(np.mean(final_values))
        median_final = float(np.median(final_values))
        std_final = float(np.std(final_values))
        min_final = float(np.min(final_values))
        max_final = float(np.max(final_values))
        
        percentiles = np.percentile(final_values, [5, 10, 25, 50, 75, 90, 95])
        
        prob_above_initial = float(np.sum(final_values > initial_value) / num_simulations * 100)
        expected_return_pct = float((mean_final - initial_value) / initial_value * 100)
        
        var_95 = float(initial_value - percentiles[0])
        var_90 = float(initial_value - percentiles[1])
        losses = final_values[final_values <= percentiles[0]]
        cvar_95 = float(initial_value - np.mean(losses)) if len(losses) > 0 else var_95
        
        mean_path = np.mean(price_paths_full, axis=0)
        lower_path = np.percentile(price_paths_full, 5, axis=0)
        upper_path = np.percentile(price_paths_full, 95, axis=0)
        
        results = {
            "simulation_parameters": {
                "initial_value": initial_value,
                "expected_return": expected_return,
                "volatility": round(volatility, 4),
                "volatility_method": "robust_iqr" if use_robust else "standard",
                "time_horizon": time_horizon,
                "num_simulations": num_simulations
            },
            "final_value_statistics": {
                "mean": round(mean_final, 2),
                "median": round(median_final, 2),
                "std_deviation": round(std_final, 2),
                "minimum": round(min_final, 2),
                "maximum": round(max_final, 2),
                "percentile_5": round(float(percentiles[0]), 2),
                "percentile_10": round(float(percentiles[1]), 2),
                "percentile_25": round(float(percentiles[2]), 2),
                "percentile_75": round(float(percentiles[4]), 2),
                "percentile_90": round(float(percentiles[5]), 2),
                "percentile_95": round(float(percentiles[6]), 2)
            },
            "probability_analysis": {
                "prob_above_initial": round(prob_above_initial, 2),
                "prob_below_initial": round(100 - prob_above_initial, 2),
                "expected_return_pct": round(expected_return_pct, 2)
            },
            "risk_metrics": {
                "value_at_risk_95": round(var_95, 2),
                "value_at_risk_90": round(var_90, 2),
                "conditional_var_95": round(cvar_95, 2),
                "max_drawdown": round(initial_value - min_final, 2),
                "max_gain": round(max_final - initial_value, 2),
                "risk_reward_ratio": round((mean_final - initial_value) / var_95, 2) if var_95 > 0 else 0
            },
            "forecast_paths": {
                "mean_path": [round(float(x), 2) for x in mean_path.tolist()],
                "lower_bound_95": [round(float(x), 2) for x in lower_path.tolist()],
                "upper_bound_95": [round(float(x), 2) for x in upper_path.tolist()]
            },
            "interpretation": generate_monte_carlo_interpretation(
                mean_final, initial_value, prob_above_initial, var_95, volatility
            ),
            "success": True
        }
        
        logger.info("Monte Carlo completed")
        return json.dumps(results, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {str(e)}", "success": False})
    except Exception as e:
        logger.error(f"Monte Carlo error: {e}", exc_info=True)
        return json.dumps({"error": str(e), "success": False})


async def arima_forecast_tool(params_json: str) -> str:
    """
    ARIMA forecasting with preprocessing for high-volatility data.
    
    Input format (JSON string):
    {
        "time_series_data": [1000, 0, 0, 1100, ...],
        "forecast_periods": 6,
        "seasonal": false,
        "aggregate_data": true,
        "aggregation_level": "auto",
        "smooth_data": true,
        "handle_zeros": true
    }
    """
    try:
        params = json.loads(params_json)
        
        time_series_data = params.get("time_series_data", [])
        forecast_periods = int(params.get("forecast_periods", 6))
        seasonal = params.get("seasonal", False)
        should_aggregate = params.get("aggregate_data", True)
        aggregation_level = params.get("aggregation_level", "auto")
        should_smooth = params.get("smooth_data", False)
        handle_zeros = params.get("handle_zeros", True)
        
        if not time_series_data or len(time_series_data) < 10:
            return json.dumps({
                "error": "Need at least 10 data points",
                "success": False
            })
        
        logger.info(f"ARIMA forecast: {len(time_series_data)} points")
        
        original_data = time_series_data.copy()
        processed_data = time_series_data.copy()
        preprocessing_steps = []
        
        if handle_zeros:
            zero_count = sum(1 for x in processed_data if x == 0)
            if zero_count > len(processed_data) * 0.3:
                preprocessing_steps.append(f"High zero-inflation: {zero_count}/{len(processed_data)} zeros")
                min_nonzero = min([x for x in processed_data if x > 0], default=1)
                epsilon = min_nonzero * 0.01
                processed_data = [max(x, epsilon) for x in processed_data]
                preprocessing_steps.append(f"Added epsilon={epsilon:.2f}")
        
        if should_aggregate:
            data_array = np.array(processed_data)
            cv = np.std(data_array) / np.mean(data_array) if np.mean(data_array) > 0 else 0
            
            if cv > 1.0:
                processed_data, period_used = aggregate_data(processed_data, aggregation_level)
                preprocessing_steps.append(f"Aggregated to {period_used} (CV={cv:.2f})")
                if period_used == 'weekly':
                    forecast_periods = max(1, forecast_periods // 7)
                elif period_used == 'monthly':
                    forecast_periods = max(1, forecast_periods // 30)
        
        if should_smooth:
            window = min(7, len(processed_data) // 4)
            processed_data = smooth_data(processed_data, method='moving_average', window=window)
            preprocessing_steps.append(f"Applied MA smoothing (window={window})")
        
        ts_data = pd.Series([float(x) for x in processed_data])
        ts_data = ts_data.dropna()
        
        if len(ts_data) < 10:
            return json.dumps({
                "error": "Insufficient data after preprocessing",
                "success": False
            })
        
        try:
            adf_result = adfuller(ts_data)
            is_stationary = adf_result[1] < 0.05
        except Exception:
            is_stationary = False
        
        try:
            if HAS_PMDARIMA:
                model = pm.auto_arima(
                    ts_data,
                    seasonal=seasonal,
                    m=12 if seasonal else 1,
                    start_p=0, start_q=0,
                    max_p=2, max_q=2,
                    d=None,
                    start_P=0, start_Q=0,
                    max_P=1, max_Q=1,
                    D=None,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True,
                    random_state=42,
                    method='lbfgs'
                )
                order = model.order
            else:
                d = 1 if not is_stationary else 0
                order = (1, d, 1)
                model = ARIMA(ts_data, order=order).fit()
        
        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            try:
                model = ExponentialSmoothing(
                    ts_data,
                    seasonal=None,
                    trend='add'
                ).fit()
                forecast_values = model.forecast(steps=forecast_periods)
                
                residuals = ts_data - model.fittedvalues
                std_err = np.std(residuals)
                conf_int_lower = forecast_values - 1.96 * std_err
                conf_int_upper = forecast_values + 1.96 * std_err
                
                order = (0, 0, 0)
                aic = model.aic
                in_sample_pred = model.fittedvalues
                
            except Exception as e2:
                return json.dumps({
                    "error": f"All methods failed: {str(e2)}",
                    "success": False
                })
        
        if 'forecast_values' not in locals():
            if HAS_PMDARIMA and hasattr(model, 'predict'):
                forecast_result = model.predict(n_periods=forecast_periods, return_conf_int=True)
                if isinstance(forecast_result, tuple):
                    forecast_values, conf_int = forecast_result
                    conf_int_lower = conf_int[:, 0]
                    conf_int_upper = conf_int[:, 1]
                else:
                    forecast_values = forecast_result
                    residuals = ts_data - model.predict_in_sample()
                    std_err = np.std(residuals)
                    conf_int_lower = forecast_values - 1.96 * std_err
                    conf_int_upper = forecast_values + 1.96 * std_err
            else:
                forecast_obj = model.get_forecast(steps=forecast_periods)
                forecast_values = forecast_obj.predicted_mean
                conf_int = forecast_obj.conf_int(alpha=0.05)
                conf_int_lower = conf_int.iloc[:, 0]
                conf_int_upper = conf_int.iloc[:, 1]
        
        if hasattr(model, 'aic'):
            aic = float(model.aic()) if callable(model.aic) else float(model.aic)
        else:
            aic = 0.0
        
        if 'in_sample_pred' not in locals():
            if HAS_PMDARIMA:
                in_sample_pred = model.predict_in_sample()
            else:
                in_sample_pred = model.fittedvalues
        
        residuals = ts_data - in_sample_pred
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals ** 2)))
        
        mape_values = []
        for actual, resid in zip(ts_data, residuals):
            if actual > 0:
                mape_values.append(abs(resid / actual))
        mape = float(np.mean(mape_values) * 100) if mape_values else 0.0
        
        results = {
            "model_parameters": {
                "p": int(order[0]) if isinstance(order, tuple) else 0,
                "d": int(order[1]) if isinstance(order, tuple) else 0,
                "q": int(order[2]) if isinstance(order, tuple) else 0,
                "seasonal": seasonal,
                "model_type": "ARIMA" if not seasonal else "SARIMA"
            },
            "preprocessing": {
                "steps_applied": preprocessing_steps,
                "original_data_points": len(original_data),
                "processed_data_points": len(ts_data)
            },
            "data_quality": {
                "zero_count": sum(1 for x in original_data if x == 0),
                "zero_percentage": round(sum(1 for x in original_data if x == 0) / len(original_data) * 100, 2),
                "coefficient_of_variation": round(float(np.std(ts_data) / np.mean(ts_data)), 2) if np.mean(ts_data) > 0 else 0,
                "is_stationary": bool(is_stationary)
            },
            "model_diagnostics": {
                "aic": round(aic, 2),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": round(mape, 2)
            },
            "forecast": {
                "values": [round(float(x), 2) for x in forecast_values],
                "lower_bound": [round(float(x), 2) for x in conf_int_lower],
                "upper_bound": [round(float(x), 2) for x in conf_int_upper],
                "confidence_level": 0.95
            },
            "interpretation": generate_arima_interpretation(
                ts_data.tolist(), forecast_values.tolist()
            ),
            "success": True
        }
        
        logger.info("ARIMA forecast completed")
        results = convert_to_native_types(results)
        return json.dumps(results, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {str(e)}", "success": False})
    except Exception as e:
        logger.error(f"ARIMA error: {e}", exc_info=True)
        return json.dumps({"error": str(e), "success": False})


async def fee_collection_forecast_tool(params_json: str) -> str:
    """
    Comprehensive fee collection forecasting - fetches data from API.
    
    Input format (JSON string):
    {
        "from_date": "2025-01-01",
        "to_date": "2025-12-31",
        "forecast_months": 6,
        "include_monte_carlo": true
    }
    """
    try:
        params = json.loads(params_json)
        
        from_date = params.get("from_date")
        to_date = params.get("to_date")
        forecast_months = int(params.get("forecast_months", 6))
        include_monte_carlo = params.get("include_monte_carlo", True)
        
        if not from_date or not to_date:
            return json.dumps({
                "error": "from_date and to_date are required (YYYY-MM-DD format)",
                "success": False
            })
        
        logger.info(f"Fee forecast: {from_date} to {to_date}")
        
        # Fetch data from API
        client = TIAFApiClient()
        api_response = await client.fee_report(from_date, to_date)
        
        logger.info(f"API response keys: {api_response.keys()}")
        logger.info(f"API response status: {api_response.get('status')} / {api_response.get('success')}")
        
        # Check both "success" and "status" fields
        is_success = api_response.get("success") or api_response.get("status") == "success"
        
        if not is_success:
            return json.dumps({
                "error": f"Failed to fetch fee data: {api_response.get('error', 'Unknown error')}",
                "success": False
            })
        
        # Parse the response
        historical_fees = parse_fee_report_data(api_response)
        
        logger.info(f"Parsed {len(historical_fees)} data points")
        if len(historical_fees) > 0:
            logger.info(f"First few values: {historical_fees[:5]}")
            logger.info(f"Last few values: {historical_fees[-5:]}")
        
        if len(historical_fees) < 10:
            return json.dumps({
                "error": f"Insufficient data: only {len(historical_fees)} data points found. Need at least 10 days with data.",
                "success": False,
                "debug_info": {
                    "api_status": api_response.get("status"),
                    "data_entries": len(api_response.get("data", [])),
                    "parsed_points": len(historical_fees)
                }
            })
        
        # Rest of the function remains the same...
        fees_array = np.array(historical_fees, dtype=float)
        zero_count = np.sum(fees_array == 0)
        zero_pct = (zero_count / len(fees_array)) * 100
        non_zero_fees = fees_array[fees_array > 0]
        
        mean_fees = float(np.mean(fees_array))
        median_fees = float(np.median(fees_array))
        std_fees = float(np.std(fees_array))
        cv = std_fees / mean_fees if mean_fees > 0 else 0
        
        should_aggregate = zero_pct > 30 or cv > 1.5
        
        # ARIMA Forecast
        arima_params = {
            "time_series_data": historical_fees,
            "forecast_periods": forecast_months,
            "seasonal": True,
            "aggregate_data": should_aggregate,
            "aggregation_level": "weekly",
            "smooth_data": cv > 1.0,
            "handle_zeros": True
        }
        arima_result_str = await arima_forecast_tool(json.dumps(arima_params))
        arima_result = json.loads(arima_result_str)
        
        # Monte Carlo Simulation
        monte_carlo_result = None
        if include_monte_carlo and len(non_zero_fees) >= 10:
            robust_vol = calculate_robust_volatility(fees_array, method='iqr')
            
            if len(non_zero_fees) > 1:
                returns = np.diff(non_zero_fees) / non_zero_fees[:-1]
                returns = returns[np.isfinite(returns)]
                expected_return = float(np.median(returns)) if len(returns) > 0 else 0.0
            else:
                expected_return = 0.0
            
            mc_params = {
                "initial_value": float(non_zero_fees[-1]) if len(non_zero_fees) > 0 else float(fees_array[-1]),
                "expected_return": expected_return,
                "volatility": robust_vol,
                "time_horizon": forecast_months,
                "num_simulations": 1000,
                "use_robust_volatility": True,
                "historical_data": historical_fees
            }
            mc_result_str = await monte_carlo_simulation_tool(json.dumps(mc_params))
            monte_carlo_result = json.loads(mc_result_str)
        
        if len(non_zero_fees) > 1:
            growth_rate = float((non_zero_fees[-1] / non_zero_fees[0]) - 1) * 100
        else:
            growth_rate = 0.0
        
        trend = "increasing" if growth_rate > 5 else "decreasing" if growth_rate < -5 else "stable"
        
        results = {
            "forecast_type": "Fee Collection Forecast",
            "data_source": {
                "from_date": from_date,
                "to_date": to_date,
                "api_status": "success"
            },
            "data_characteristics": {
                "data_points": len(historical_fees),
                "zero_count": int(zero_count),
                "zero_percentage": round(zero_pct, 2),
                "coefficient_of_variation": round(cv, 2),
                "volatility_level": "high" if cv > 1.0 else "moderate" if cv > 0.5 else "low"
            },
            "historical_summary": {
                "mean_fees": round(mean_fees, 2),
                "median_fees": round(median_fees, 2),
                "std_fees": round(std_fees, 2),
                "min_fees": round(float(np.min(fees_array)), 2),
                "max_fees": round(float(np.max(fees_array)), 2),
                "last_value": round(float(fees_array[-1]), 2),
                "trend": trend,
                "growth_rate_pct": round(growth_rate, 2)
            },
            "arima_forecast": arima_result,
            "monte_carlo_simulation": monte_carlo_result,
            "recommendations": generate_forecast_recommendations(
                arima_result, monte_carlo_result, historical_fees, zero_pct, cv, "fee"
            ),
            "success": True
        }
        
        logger.info("Fee forecast completed")
        results = convert_to_native_types(results)
        return json.dumps(results, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {str(e)}", "success": False})
    except Exception as e:
        logger.error(f"Fee forecast error: {e}", exc_info=True)
        return json.dumps({"error": str(e), "success": False})


async def expense_forecast_tool(params_json: str) -> str:
    """
    Comprehensive expense forecasting - fetches data from API.
    
    Input format (JSON string):
    {
        "from_date": "2025-01-01",
        "to_date": "2025-12-31",
        "forecast_months": 6,
        "include_monte_carlo": true
    }
    """
    try:
        params = json.loads(params_json)
        
        from_date = params.get("from_date")
        to_date = params.get("to_date")
        forecast_months = int(params.get("forecast_months", 6))
        include_monte_carlo = params.get("include_monte_carlo", True)
        
        if not from_date or not to_date:
            return json.dumps({
                "error": "from_date and to_date are required (YYYY-MM-DD format)",
                "success": False
            })
        
        logger.info(f"Expense forecast: {from_date} to {to_date}")
        
        # Fetch data from API
        client = TIAFApiClient()
        api_response = await client.expense_report(from_date, to_date)
        
        logger.info(f"API response keys: {api_response.keys()}")
        logger.info(f"API response status: {api_response.get('status')} / {api_response.get('success')}")
        
        # Check both "success" and "status" fields
        is_success = api_response.get("success") or api_response.get("status") == "success"
        
        if not is_success:
            return json.dumps({
                "error": f"Failed to fetch expense data: {api_response.get('error', 'Unknown error')}",
                "success": False
            })
        
        # Parse the response
        historical_expenses = parse_expense_report_data(api_response)
        
        logger.info(f"Parsed {len(historical_expenses)} data points")
        if len(historical_expenses) > 0:
            logger.info(f"First few values: {historical_expenses[:5]}")
            logger.info(f"Last few values: {historical_expenses[-5:]}")
        
        if len(historical_expenses) < 10:
            return json.dumps({
                "error": f"Insufficient data: only {len(historical_expenses)} data points. Need at least 10 days with data.",
                "success": False,
                "debug_info": {
                    "api_status": api_response.get("status"),
                    "data_entries": len(api_response.get("data", [])),
                    "parsed_points": len(historical_expenses)
                }
            })
        
        # Rest of the function remains the same...
        expenses_array = np.array(historical_expenses, dtype=float)
        zero_count = np.sum(expenses_array == 0)
        zero_pct = (zero_count / len(expenses_array)) * 100
        non_zero_expenses = expenses_array[expenses_array > 0]
        
        mean_expenses = float(np.mean(expenses_array))
        median_expenses = float(np.median(expenses_array))
        std_expenses = float(np.std(expenses_array))
        cv = std_expenses / mean_expenses if mean_expenses > 0 else 0
        
        should_aggregate = zero_pct > 30 or cv > 1.5
        
        # ARIMA Forecast
        arima_params = {
            "time_series_data": historical_expenses,
            "forecast_periods": forecast_months,
            "seasonal": True,
            "aggregate_data": should_aggregate,
            "aggregation_level": "weekly",
            "smooth_data": cv > 1.0,
            "handle_zeros": True
        }
        arima_result_str = await arima_forecast_tool(json.dumps(arima_params))
        arima_result = json.loads(arima_result_str)
        
        # Monte Carlo Simulation
        monte_carlo_result = None
        if include_monte_carlo and len(non_zero_expenses) >= 10:
            robust_vol = calculate_robust_volatility(expenses_array, method='iqr')
            
            if len(non_zero_expenses) > 1:
                returns = np.diff(non_zero_expenses) / non_zero_expenses[:-1]
                returns = returns[np.isfinite(returns)]
                expected_return = float(np.median(returns)) if len(returns) > 0 else 0.0
            else:
                expected_return = 0.0
            
            mc_params = {
                "initial_value": float(non_zero_expenses[-1]) if len(non_zero_expenses) > 0 else float(expenses_array[-1]),
                "expected_return": expected_return,
                "volatility": robust_vol,
                "time_horizon": forecast_months,
                "num_simulations": 1000,
                "use_robust_volatility": True,
                "historical_data": historical_expenses
            }
            mc_result_str = await monte_carlo_simulation_tool(json.dumps(mc_params))
            monte_carlo_result = json.loads(mc_result_str)
        
        if len(non_zero_expenses) > 1:
            growth_rate = float((non_zero_expenses[-1] / non_zero_expenses[0]) - 1) * 100
        else:
            growth_rate = 0.0
        
        trend = "increasing" if growth_rate > 5 else "decreasing" if growth_rate < -5 else "stable"
        
        results = {
            "forecast_type": "Expense Forecast",
            "data_source": {
                "from_date": from_date,
                "to_date": to_date,
                "api_status": "success"
            },
            "data_characteristics": {
                "data_points": len(historical_expenses),
                "zero_count": int(zero_count),
                "zero_percentage": round(zero_pct, 2),
                "coefficient_of_variation": round(cv, 2),
                "volatility_level": "high" if cv > 1.0 else "moderate" if cv > 0.5 else "low"
            },
            "historical_summary": {
                "mean_expenses": round(mean_expenses, 2),
                "median_expenses": round(median_expenses, 2),
                "std_expenses": round(std_expenses, 2),
                "min_expenses": round(float(np.min(expenses_array)), 2),
                "max_expenses": round(float(np.max(expenses_array)), 2),
                "last_value": round(float(expenses_array[-1]), 2),
                "trend": trend,
                "growth_rate_pct": round(growth_rate, 2)
            },
            "arima_forecast": arima_result,
            "monte_carlo_simulation": monte_carlo_result,
            "recommendations": generate_forecast_recommendations(
                arima_result, monte_carlo_result, historical_expenses, zero_pct, cv, "expense"
            ),
            "success": True
        }
        
        logger.info("Expense forecast completed")
        results = convert_to_native_types(results)
        return json.dumps(results, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {str(e)}", "success": False})
    except Exception as e:
        logger.error(f"Expense forecast error: {e}", exc_info=True)
        return json.dumps({"error": str(e), "success": False})


def generate_monte_carlo_interpretation(mean_final, initial_value, prob_above, var_95, volatility):
    """Generate interpretation for Monte Carlo results"""
    change_pct = ((mean_final - initial_value) / initial_value) * 100
    
    interpretation = []
    interpretation.append(
        f"Expected value: Rs {mean_final:,.2f} ({change_pct:+.2f}% from current)"
    )
    interpretation.append(
        f"Probability of increase: {prob_above:.1f}%"
    )
    interpretation.append(
        f"Volatility: {volatility*100:.1f}% ({'High' if volatility > 0.3 else 'Moderate' if volatility > 0.15 else 'Low'})"
    )
    interpretation.append(
        f"Potential loss (95% confidence): Rs {var_95:,.2f}"
    )
    
    if prob_above > 70:
        interpretation.append("Strong growth probability with manageable risk")
    elif prob_above < 30:
        interpretation.append("High downside risk - implement safeguards")
    else:
        interpretation.append("Balanced risk-reward scenario")
    
    return interpretation


def generate_arima_interpretation(historical, forecast):
    """Generate interpretation for ARIMA results"""
    interpretation = []
    
    if not historical or not forecast:
        return ["Unable to generate interpretation"]
    
    last_actual = historical[-1]
    mean_forecast = np.mean(forecast)
    change_pct = ((mean_forecast - last_actual) / last_actual) * 100 if last_actual > 0 else 0
    
    interpretation.append(
        f"Forecast indicates {change_pct:+.2f}% change from recent levels"
    )
    
    if change_pct > 15:
        interpretation.append("Strong growth trend expected")
    elif change_pct < -15:
        interpretation.append("Declining trend detected")
    else:
        interpretation.append("Relatively stable forecast")
    
    return interpretation


def generate_forecast_recommendations(arima_result, monte_carlo_result, historical_data, zero_pct, cv, forecast_type):
    """Generate actionable recommendations"""
    recommendations = []
    
    # Data quality recommendations
    if zero_pct > 50:
        recommendations.append(
            f"High zero-inflation ({zero_pct:.1f}%): Consider weekly/monthly aggregation"
        )
    
    if cv > 1.5:
        recommendations.append(
            f"Very high volatility (CV={cv:.2f}): " + 
            ("Implement payment schedules" if forecast_type == "fee" else "Review expense policies")
        )
    elif cv > 1.0:
        recommendations.append(
            f"High volatility (CV={cv:.2f}): Monitor patterns closely"
        )
    
    # ARIMA recommendations
    if arima_result and arima_result.get("success"):
        mape = arima_result.get("model_diagnostics", {}).get("mape", 100)
        
        if mape < 15:
            recommendations.append("Forecast accuracy is good despite volatility")
        elif mape > 30:
            recommendations.append("High uncertainty - use ranges, not point estimates")
        
        forecast_values = arima_result.get("forecast", {}).get("values", [])
        if forecast_values:
            last_actual = historical_data[-1] if historical_data else 0
            mean_forecast = np.mean(forecast_values)
            
            if mean_forecast > last_actual * 1.2:
                if forecast_type == "fee":
                    recommendations.append("Strong growth expected - prepare for increased capacity")
                else:
                    recommendations.append("Rising expenses expected - review cost controls")
            elif mean_forecast < last_actual * 0.8:
                if forecast_type == "fee":
                    recommendations.append("Declining trend - review collection strategies")
                else:
                    recommendations.append("Declining expenses - ensure service quality maintained")
    
    # Monte Carlo recommendations
    if monte_carlo_result and monte_carlo_result.get("success"):
        prob_increase = monte_carlo_result.get("probability_analysis", {}).get("prob_above_initial", 50)
        var_95 = monte_carlo_result.get("risk_metrics", {}).get("value_at_risk_95", 0)
        
        if prob_increase > 65:
            recommendations.append(f"Positive outlook ({prob_increase:.0f}% growth probability)")
        elif prob_increase < 35:
            recommendations.append(f"Risk alert: {100-prob_increase:.0f}% decline probability")
        
        if var_95 > 0:
            recommendations.append(
                f"Financial risk: Rs {var_95:,.2f} potential shortfall (maintain reserves)"
            )
    
    # Type-specific recommendations
    if forecast_type == "fee":
        recommendations.append(
            "Consider: automated reminders, installment plans, early payment incentives"
        )
    else:
        recommendations.append(
            "Consider: budget controls, vendor negotiations, expense tracking systems"
        )
    
    return recommendations