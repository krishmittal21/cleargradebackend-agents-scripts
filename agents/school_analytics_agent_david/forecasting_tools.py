"""
Forecasting and Simulation Tools for School Business Agent
Includes Monte Carlo simulation and ARIMA forecasting capabilities
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


async def monte_carlo_simulation_tool(params_json: str) -> str:
    """
    Perform Monte Carlo simulation for forecasting future values with uncertainty.
    Useful for revenue forecasting, fee collection prediction with risk analysis.
    
    Input format (JSON string):
    {
        "initial_value": 100000,
        "expected_return": 0.08,
        "volatility": 0.2,
        "time_horizon": 12,
        "num_simulations": 1000
    }
    
    Returns detailed simulation results with statistics and risk metrics.
    """
    try:
        params = json.loads(params_json)
        
        initial_value = float(params.get("initial_value"))
        expected_return = float(params.get("expected_return"))
        volatility = float(params.get("volatility"))
        time_horizon = int(params.get("time_horizon"))
        num_simulations = int(params.get("num_simulations", 1000))
        
        logger.info(f"Running Monte Carlo simulation with {num_simulations} paths")
        
        # Time step
        dt = 1
        steps = time_horizon
        
        # Calculate drift
        drift = (expected_return - 0.5 * volatility ** 2) * dt
        
        # Generate random shocks
        Z = np.random.normal(0, 1, size=(num_simulations, steps))
        
        # Build paths using vectorized GBM
        diffusion = volatility * np.sqrt(dt) * Z
        increments = drift + diffusion
        log_paths = np.cumsum(increments, axis=1)
        S_paths = initial_value * np.exp(log_paths)
        
        # Prepend initial value
        initial_col = np.full((num_simulations, 1), initial_value)
        simulation_results = np.hstack([initial_col, S_paths])
        
        # Extract final values
        final_values = simulation_results[:, -1]
        
        # Calculate statistics
        mean_final = np.mean(final_values)
        median_final = np.median(final_values)
        std_final = np.std(final_values)
        min_final = np.min(final_values)
        max_final = np.max(final_values)
        
        # Calculate percentiles (Value at Risk)
        percentile_5 = np.percentile(final_values, 5)
        percentile_25 = np.percentile(final_values, 25)
        percentile_75 = np.percentile(final_values, 75)
        percentile_95 = np.percentile(final_values, 95)
        
        # Probability analysis
        prob_above_initial = (final_values > initial_value).sum() / num_simulations * 100
        prob_below_initial = (final_values < initial_value).sum() / num_simulations * 100
        expected_final_return = (mean_final - initial_value) / initial_value * 100
        
        # Risk metrics
        var_95 = initial_value - percentile_5
        cvar_95 = initial_value - np.mean(final_values[final_values <= percentile_5])
        
        # Calculate mean path
        mean_path = np.mean(simulation_results, axis=0)
        percentile_5_path = np.percentile(simulation_results, 5, axis=0)
        percentile_95_path = np.percentile(simulation_results, 95, axis=0)
        
        results = {
            "simulation_parameters": {
                "initial_value": initial_value,
                "expected_return": expected_return,
                "volatility": volatility,
                "time_horizon": time_horizon,
                "num_simulations": num_simulations
            },
            "final_value_statistics": {
                "mean": round(mean_final, 2),
                "median": round(median_final, 2),
                "std_deviation": round(std_final, 2),
                "minimum": round(min_final, 2),
                "maximum": round(max_final, 2),
                "percentile_5": round(percentile_5, 2),
                "percentile_25": round(percentile_25, 2),
                "percentile_75": round(percentile_75, 2),
                "percentile_95": round(percentile_95, 2)
            },
            "probability_analysis": {
                "prob_above_initial": round(prob_above_initial, 2),
                "prob_below_initial": round(prob_below_initial, 2),
                "expected_return_pct": round(expected_final_return, 2)
            },
            "risk_metrics": {
                "value_at_risk_95": round(var_95, 2),
                "conditional_var_95": round(cvar_95, 2),
                "max_drawdown": round(initial_value - min_final, 2),
                "max_gain": round(max_final - initial_value, 2)
            },
            "interpretation": generate_monte_carlo_interpretation(
                mean_final, initial_value, prob_above_initial, var_95
            )
        }
        
        logger.info("Monte Carlo simulation completed successfully")
        return json.dumps(results, indent=2)
        
    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation: {e}")
        return json.dumps({"error": str(e), "success": False})


async def arima_forecast_tool(params_json: str) -> str:
    """
    Perform ARIMA time series forecasting for fee collection, enrollment, or revenue trends.
    
    Input format (JSON string):
    {
        "time_series_data": [1000, 1050, 1100, ...],
        "forecast_periods": 6,
        "auto_detect": true,
        "seasonal": false
    }
    
    Returns forecast values with confidence intervals and model diagnostics.
    """
    try:
        params = json.loads(params_json)
        
        time_series_data = params.get("time_series_data", [])
        forecast_periods = int(params.get("forecast_periods", 6))
        auto_detect = params.get("auto_detect", True)
        seasonal = params.get("seasonal", False)
        p = params.get("p")
        d = params.get("d")
        q = params.get("q")
        
        logger.info(f"Running ARIMA forecast for {forecast_periods} periods")
        
        # Convert to pandas Series
        ts_data = pd.Series(time_series_data)
        
        # Check if data is sufficient
        if len(ts_data) < 10:
            return json.dumps({
                "error": "Insufficient data. At least 10 observations required for ARIMA.",
                "success": False
            })
        
        # Stationarity test
        adf_result = adfuller(ts_data.dropna())
        is_stationary = adf_result[1] < 0.05
        
        # Auto-detect parameters if needed
        if auto_detect and (p is None or d is None or q is None):
            if not is_stationary:
                d = 1
            else:
                d = 0
            
            max_lag = min(10, len(ts_data) // 5)
            p = p if p is not None else min(2, max_lag)
            q = q if q is not None else min(2, max_lag)
        
        # Fit model
        logger.info(f"Fitting ARIMA({p},{d},{q}) model")
        model = ARIMA(ts_data, order=(p, d, q))
        fitted_model = model.fit()
        
        # Generate forecast
        forecast_obj = fitted_model.get_forecast(steps=forecast_periods)
        forecast_ci = forecast_obj.conf_int(alpha=0.05)
        forecast_result = forecast_obj.predicted_mean
        
        # Model diagnostics
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        # In-sample predictions
        in_sample_predictions = fitted_model.fittedvalues
        
        # Calculate accuracy metrics
        residuals = ts_data - in_sample_predictions
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            mape_arr = np.abs(residuals / np.where(ts_data != 0, ts_data, np.nan))
            mape = float(np.nanmean(mape_arr) * 100)
        
        results = {
            "model_parameters": {
                "p": int(p),
                "d": int(d),
                "q": int(q),
                "model_type": "ARIMA"
            },
            "stationarity_test": {
                "adf_statistic": round(adf_result[0], 4),
                "p_value": round(adf_result[1], 4),
                "is_stationary": bool(is_stationary)
            },
            "model_diagnostics": {
                "aic": round(aic, 2),
                "bic": round(bic, 2),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mape": round(mape, 2)
            },
            "forecast": {
                "values": [round(x, 2) for x in forecast_result.tolist()],
                "lower_bound": [round(x, 2) for x in forecast_ci.iloc[:, 0].tolist()],
                "upper_bound": [round(x, 2) for x in forecast_ci.iloc[:, 1].tolist()],
                "confidence_level": 0.95
            },
            "historical_data": {
                "actual": [round(x, 2) for x in ts_data.tolist()],
                "fitted": [round(x, 2) for x in in_sample_predictions.tolist()]
            },
            "interpretation": generate_arima_interpretation(
                ts_data.tolist(), forecast_result.tolist()
            ),
            "success": True
        }
        
        logger.info("ARIMA forecast completed successfully")
        return json.dumps(results, indent=2)
        
    except Exception as e:
        logger.error(f"Error in ARIMA forecast: {e}")
        return json.dumps({"error": str(e), "success": False})


async def fee_collection_forecast_tool(params_json: str) -> str:
    """
    Comprehensive fee collection forecasting combining ARIMA and Monte Carlo simulation.
    Analyzes historical fee data and provides future predictions with risk analysis.
    
    Input format (JSON string):
    {
        "historical_fees": [50000, 52000, 51500, ...],
        "forecast_months": 6,
        "include_monte_carlo": true
    }
    
    Returns comprehensive forecast with recommendations.
    """
    try:
        params = json.loads(params_json)
        
        historical_fees = params.get("historical_fees", [])
        forecast_months = int(params.get("forecast_months", 6))
        include_monte_carlo = params.get("include_monte_carlo", True)
        
        logger.info(f"Running fee collection forecast for {forecast_months} months")
        
        if len(historical_fees) < 10:
            return json.dumps({
                "error": "Need at least 10 months of historical fee data",
                "success": False
            })
        
        # Step 1: ARIMA Forecast
        arima_params = {
            "time_series_data": historical_fees,
            "forecast_periods": forecast_months,
            "auto_detect": True
        }
        arima_result_str = await arima_forecast_tool(json.dumps(arima_params))
        arima_result = json.loads(arima_result_str)
        
        if not arima_result.get("success"):
            return arima_result_str
        
        # Step 2: Monte Carlo Simulation (if requested)
        monte_carlo_result = None
        if include_monte_carlo:
            # Calculate historical statistics
            returns = np.diff(historical_fees) / np.array(historical_fees[:-1])
            expected_return = float(np.mean(returns))
            volatility = float(np.std(returns))
            
            mc_params = {
                "initial_value": historical_fees[-1],
                "expected_return": expected_return,
                "volatility": max(volatility, 0.01),  # Ensure non-zero
                "time_horizon": forecast_months,
                "num_simulations": 1000
            }
            mc_result_str = await monte_carlo_simulation_tool(json.dumps(mc_params))
            monte_carlo_result = json.loads(mc_result_str)
        
        # Combine results
        results = {
            "forecast_type": "Fee Collection Analysis",
            "historical_summary": {
                "data_points": len(historical_fees),
                "mean_fees": round(np.mean(historical_fees), 2),
                "std_fees": round(np.std(historical_fees), 2),
                "min_fees": round(np.min(historical_fees), 2),
                "max_fees": round(np.max(historical_fees), 2),
                "trend": "increasing" if historical_fees[-1] > historical_fees[0] else "decreasing",
                "growth_rate": round(((historical_fees[-1] / historical_fees[0]) - 1) * 100, 2)
            },
            "arima_forecast": arima_result,
            "monte_carlo_simulation": monte_carlo_result,
            "recommendations": generate_fee_forecast_recommendations(
                arima_result, monte_carlo_result, historical_fees
            ),
            "success": True
        }
        
        logger.info("Fee collection forecast completed successfully")
        return json.dumps(results, indent=2)
        
    except Exception as e:
        logger.error(f"Error in fee collection forecast: {e}")
        return json.dumps({"error": str(e), "success": False})


def generate_monte_carlo_interpretation(mean_final, initial_value, prob_above, var_95):
    """Generate interpretation for Monte Carlo results"""
    change_pct = ((mean_final - initial_value) / initial_value) * 100
    
    interpretation = []
    interpretation.append(f"Expected value after simulation: {mean_final:.2f} ({change_pct:+.2f}%)")
    interpretation.append(f"Probability of increase: {prob_above:.1f}%")
    interpretation.append(f"Maximum potential loss (95% confidence): {var_95:.2f}")
    
    if prob_above > 70:
        interpretation.append("⚠️ High probability of growth - consider expansion opportunities")
    elif prob_above < 30:
        interpretation.append("⚠️ High risk of decline - review strategies and implement safeguards")
    
    return interpretation


def generate_arima_interpretation(historical, forecast):
    """Generate interpretation for ARIMA results"""
    interpretation = []
    
    last_actual = historical[-1]
    mean_forecast = np.mean(forecast)
    change_pct = ((mean_forecast - last_actual) / last_actual) * 100
    
    interpretation.append(f"Forecast indicates {change_pct:+.2f}% change from current level")
    
    if change_pct > 10:
        interpretation.append("📈 Strong growth trend - prepare for increased capacity")
    elif change_pct < -10:
        interpretation.append("📉 Declining trend - investigate underlying causes")
    else:
        interpretation.append("➡️ Stable trend - maintain current operations")
    
    return interpretation


def generate_fee_forecast_recommendations(arima_result, monte_carlo_result, historical_data):
    """Generate actionable recommendations for fee collection forecast"""
    recommendations = []
    
    # ARIMA-based recommendations
    if arima_result.get("success"):
        forecast_values = arima_result["forecast"]["values"]
        mean_forecast = np.mean(forecast_values)
        last_actual = historical_data[-1]
        
        if mean_forecast > last_actual * 1.1:
            recommendations.append(
                "✅ Strong fee collection growth expected: Plan for increased cash flow and potential investments"
            )
        elif mean_forecast < last_actual * 0.9:
            recommendations.append(
                "⚠️ Declining fee collection forecast: Review fee payment policies and collection strategies"
            )
        else:
            recommendations.append(
                "➡️ Stable fee collection expected: Maintain current collection processes"
            )
        
        # Check forecast accuracy
        mape = arima_result["model_diagnostics"].get("mape", 0)
        if mape < 10:
            recommendations.append("✅ High forecast accuracy - reliable predictions")
        elif mape > 20:
            recommendations.append("⚠️ Lower forecast accuracy - use with caution and monitor closely")
    
    # Monte Carlo-based recommendations
    if monte_carlo_result and monte_carlo_result.get("success"):
        prob_increase = monte_carlo_result["probability_analysis"]["prob_above_initial"]
        var_95 = monte_carlo_result["risk_metrics"]["value_at_risk_95"]
        
        if prob_increase > 70:
            recommendations.append(
                f"✅ High confidence ({prob_increase:.1f}%) in fee collection growth"
            )
        elif prob_increase < 30:
            recommendations.append(
                f"⚠️ High risk ({100-prob_increase:.1f}%) of fee collection decline - implement contingency plans"
            )
        
        recommendations.append(
            f"💰 Financial Risk: Maximum potential shortfall of ₹{var_95:,.2f} (95% confidence)"
        )
    
    return recommendations