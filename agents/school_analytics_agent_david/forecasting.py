import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import statistics

from tiaf_api_client import TIAFApiClient

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Container for forecast results"""

    metric: str
    historical_data: List[Dict[str, Any]]
    predictions: List[Dict[str, Any]]
    method: str
    historical_pattern: Dict[str, Any]
    summary_stats: Dict[str, Any]
    confidence_interval: Optional[Tuple[float, float]] = None
    accuracy_metrics: Optional[Dict[str, float]] = None
    warning: Optional[str] = None


class FinancialForecaster:
    """Financial forecasting for irregular data patterns"""

    def __init__(self, api_client: Optional[TIAFApiClient] = None):
        self.api_client = api_client or TIAFApiClient()
        self.methods = [
            "monthly_average",
            "weekly_average",
            "historical_pattern",
        ]

    async def fetch_historical_data(
        self, metric: str, days: int = 180
    ) -> Tuple[List[Dict[str, Any]], datetime, datetime]:
        """Fetch historical data from API and return with date range"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        from_date = start_date.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")

        # Get historical data from API
        if metric == "revenue":
            result = await self.api_client.fee_report(from_date, to_date)
        elif metric == "expense":
            result = await self.api_client.expense_report(from_date, to_date)
        else:
            raise ValueError(f"Invalid metric: {metric}")

        if not result.get("success"):
            raise RuntimeError(
                f"Failed to fetch data: {result.get('error')}"
            )

        # Extract the actual data from nested response
        api_response = result.get("data", {})
        historical_data = (
            api_response.get("data", [])
            if isinstance(api_response, dict)
            else []
        )

        return historical_data, start_date, end_date

    def prepare_time_series(
        self, data: List[Dict[str, Any]], value_key: str
    ) -> List[Tuple[datetime, float]]:
        """Convert API data to time series format"""
        time_series = []
        for item in data:
            try:
                # Handle different date field names
                date_str = (
                    item.get("Date")
                    or item.get("date")
                    or item.get("tr_date", "")
                )
                if not date_str:
                    continue

                date = datetime.strptime(date_str, "%Y-%m-%d")

                # Handle different value field names
                value = (
                    item.get("Day Total")
                    or item.get(value_key)
                    or item.get("amount")
                    or 0
                )
                value = float(value)
                time_series.append((date, value))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid data point: {e}")
                continue

        return sorted(time_series, key=lambda x: x[0])

    def analyze_collection_pattern(
        self,
        time_series: List[Tuple[datetime, float]],
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """
        Analyze the fee collection pattern
        
        IMPORTANT: API only returns days WITH collections,
        so we need to calculate against calendar days
        """
        if not time_series:
            return {}

        # Calculate actual calendar days in the period
        calendar_days = (end_date - start_date).days + 1

        # Days with collections (from API)
        collection_days = len(time_series)
        zero_days = calendar_days - collection_days

        values = [v for _, v in time_series]
        total_collected = sum(values)

        # Real collection frequency against calendar days
        collection_frequency = (
            (collection_days / calendar_days * 100) if calendar_days > 0 else 0
        )

        # Average per collection day (not per calendar day)
        avg_on_collection_days = (
            statistics.mean(values) if values else 0
        )

        # Average per calendar day (including zeros)
        daily_average = total_collected / calendar_days if calendar_days > 0 else 0

        return {
            "calendar_days": calendar_days,
            "collection_days": collection_days,
            "zero_days": zero_days,
            "collection_frequency": collection_frequency,
            "avg_on_collection_days": avg_on_collection_days,
            "total_collected": total_collected,
            "daily_average": daily_average,
            "period_start": start_date.strftime("%Y-%m-%d"),
            "period_end": end_date.strftime("%Y-%m-%d"),
        }

    def aggregate_by_period(
        self, time_series: List[Tuple[datetime, float]], period: str = "week"
    ) -> List[Tuple[datetime, float]]:
        """Aggregate data by week or month"""
        if not time_series:
            return []

        aggregated = {}
        for date, value in time_series:
            if period == "week":
                key = date - timedelta(days=date.weekday())
            elif period == "month":
                key = date.replace(day=1)
            else:
                key = date

            if key not in aggregated:
                aggregated[key] = 0
            aggregated[key] += value

        return sorted(aggregated.items())

    def monthly_average_forecast(
        self,
        time_series: List[Tuple[datetime, float]],
        periods: int,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Forecast based on monthly aggregation"""
        monthly_data = self.aggregate_by_period(time_series, "month")

        if len(monthly_data) < 1:
            return []

        monthly_values = [v for _, v in monthly_data]
        
        # Use median for robustness
        if len(monthly_values) >= 2:
            median_monthly = statistics.median(monthly_values)
        else:
            median_monthly = monthly_values[0]

        last_date = time_series[-1][0] if time_series else end_date
        predictions = []

        # Calculate daily average based on calendar days
        calendar_days = (end_date - start_date).days + 1
        months_in_period = calendar_days / 30
        
        if months_in_period > 0:
            # Distribute evenly across calendar days
            daily_estimate = (
                (median_monthly * len(monthly_data) / months_in_period) / 30
            )
        else:
            daily_estimate = 0

        for i in range(1, periods + 1):
            future_date = last_date + timedelta(days=i)
            predictions.append(
                {
                    "date": future_date.strftime("%Y-%m-%d"),
                    "predicted_value": round(daily_estimate, 2),
                    "method": "monthly_average",
                }
            )

        return predictions

    def weekly_average_forecast(
        self,
        time_series: List[Tuple[datetime, float]],
        periods: int,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Forecast based on weekly aggregation"""
        weekly_data = self.aggregate_by_period(time_series, "week")

        if len(weekly_data) < 1:
            return []

        weekly_values = [v for _, v in weekly_data]
        
        if len(weekly_values) >= 2:
            median_weekly = statistics.median(weekly_values)
        else:
            median_weekly = weekly_values[0]

        last_date = time_series[-1][0] if time_series else end_date
        predictions = []

        # Calculate daily average
        calendar_days = (end_date - start_date).days + 1
        weeks_in_period = calendar_days / 7
        
        if weeks_in_period > 0:
            daily_estimate = (
                (median_weekly * len(weekly_data) / weeks_in_period) / 7
            )
        else:
            daily_estimate = 0

        for i in range(1, periods + 1):
            future_date = last_date + timedelta(days=i)
            predictions.append(
                {
                    "date": future_date.strftime("%Y-%m-%d"),
                    "predicted_value": round(daily_estimate, 2),
                    "method": "weekly_average",
                }
            )

        return predictions

    def historical_pattern_forecast(
        self,
        pattern: Dict[str, Any],
        last_date: datetime,
        periods: int,
    ) -> List[Dict[str, Any]]:
        """
        Forecast based on historical collection patterns
        
        Uses the daily average (which includes zero days in the denominator)
        """
        if pattern.get("calendar_days", 0) == 0:
            return []

        # Use daily average which already accounts for zero days
        daily_avg = pattern["daily_average"]

        predictions = []
        for i in range(1, periods + 1):
            future_date = last_date + timedelta(days=i)
            predictions.append(
                {
                    "date": future_date.strftime("%Y-%m-%d"),
                    "predicted_value": round(daily_avg, 2),
                    "method": "historical_pattern",
                }
            )

        return predictions

    def calculate_summary_stats(
        self, predictions: List[Dict[str, Any]], periods: int
    ) -> Dict[str, Any]:
        """Calculate summary statistics for predictions"""
        if not predictions:
            return {}

        total_predicted = sum(p["predicted_value"] for p in predictions)
        daily_avg = total_predicted / len(predictions)

        monthly_estimate = (total_predicted / periods) * 30
        annual_estimate = (total_predicted / periods) * 365

        return {
            "total_predicted": round(total_predicted, 2),
            "daily_average": round(daily_avg, 2),
            "monthly_estimate": round(monthly_estimate, 2),
            "annual_estimate": round(annual_estimate, 2),
            "forecast_days": periods,
        }

    async def generate_forecast(
        self,
        metric: str,
        periods: int = 30,
        method: str = "historical_pattern",
        historical_days: int = 180,
    ) -> ForecastResult:
        """Generate complete forecast"""
        # Validate inputs
        if metric not in ["revenue", "expense"]:
            raise ValueError("Metric must be 'revenue' or 'expense'")

        if not (7 <= periods <= 365):
            raise ValueError("Forecast periods must be between 7 and 365 days")

        # Fetch historical data with date range
        historical_data, start_date, end_date = (
            await self.fetch_historical_data(metric, historical_days)
        )

        if not historical_data:
            return ForecastResult(
                metric=metric,
                historical_data=[],
                predictions=[],
                method=method,
                historical_pattern={},
                summary_stats={},
                warning="No historical data available for forecasting",
            )

        # Prepare time series
        value_key = "amount" if metric == "revenue" else "expense_amount"
        time_series = self.prepare_time_series(historical_data, value_key)

        if len(time_series) < 1:
            return ForecastResult(
                metric=metric,
                historical_data=historical_data,
                predictions=[],
                method=method,
                historical_pattern={},
                summary_stats={},
                warning="Insufficient historical data for forecasting",
            )

        # Analyze pattern (with calendar days)
        pattern = self.analyze_collection_pattern(
            time_series, start_date, end_date
        )

        # Check if data is too sparse
        warning = None
        if pattern["collection_frequency"] < 10:
            warning = (
                f"Warning: Collections occur only on "
                f"{pattern['collection_frequency']:.1f}% of days "
                f"({pattern['collection_days']} out of "
                f"{pattern['calendar_days']} days). "
                f"Forecast reliability may vary."
            )

        # Get last date for predictions
        last_date = time_series[-1][0] if time_series else end_date

        # Generate forecast based on method
        if method == "monthly_average":
            predictions = self.monthly_average_forecast(
                time_series, periods, start_date, end_date
            )
        elif method == "weekly_average":
            predictions = self.weekly_average_forecast(
                time_series, periods, start_date, end_date
            )
        elif method == "historical_pattern":
            predictions = self.historical_pattern_forecast(
                pattern, last_date, periods
            )
        else:
            predictions = self.historical_pattern_forecast(
                pattern, last_date, periods
            )

        # Calculate summary stats
        summary_stats = self.calculate_summary_stats(predictions, periods)

        return ForecastResult(
            metric=metric,
            historical_data=historical_data,
            predictions=predictions,
            method=method,
            historical_pattern=pattern,
            summary_stats=summary_stats,
            warning=warning,
        )

    async def generate_business_forecast(
        self,
        metric: str,
        forecast_months: int = 6,
    ) -> Dict[str, Any]:
        """
        Generate a business-focused forecast with monthly projections.
        
        Returns clear, actionable output for questions like:
        "What revenue can I expect in the coming 6 months?"
        
        Args:
            metric: 'revenue' or 'expense'
            forecast_months: Number of months to forecast (1-12)
            
        Returns:
            Dictionary with monthly projections, totals, and context
        """
        # Validate inputs
        if metric not in ["revenue", "expense"]:
            return {
                "success": False,
                "error": "Metric must be 'revenue' or 'expense'"
            }
        
        if not (1 <= forecast_months <= 12):
            return {
                "success": False, 
                "error": "Forecast months must be between 1 and 12"
            }
        
        try:
            # Fetch 12 months of historical data for context
            historical_data, start_date, end_date = await self.fetch_historical_data(
                metric, days=365
            )
            
            if not historical_data:
                return {
                    "success": False,
                    "error": "No historical data available for forecasting"
                }
            
            # Prepare time series
            value_key = "amount" if metric == "revenue" else "expense_amount"
            time_series = self.prepare_time_series(historical_data, value_key)
            
            if len(time_series) < 10:
                return {
                    "success": False,
                    "error": "Insufficient historical data for forecasting (need at least 10 data points)"
                }
            
            # Aggregate by month
            monthly_data = self.aggregate_by_period(time_series, "month")
            
            if len(monthly_data) < 1:
                return {
                    "success": False,
                    "error": "Unable to aggregate data by month"
                }
            
            # Calculate monthly statistics
            monthly_values = [v for _, v in monthly_data]
            avg_monthly = statistics.mean(monthly_values) if monthly_values else 0
            
            # Use median for more robust estimate (less affected by outliers)
            median_monthly = statistics.median(monthly_values) if len(monthly_values) >= 2 else avg_monthly
            
            # Simple trend calculation (compare recent months to older months)
            trend_factor = 1.0
            trend_direction = "stable"
            if len(monthly_values) >= 4:
                recent_avg = statistics.mean(monthly_values[-2:])
                older_avg = statistics.mean(monthly_values[:-2])
                if older_avg > 0:
                    trend_factor = recent_avg / older_avg
                    if trend_factor > 1.05:
                        trend_direction = "increasing"
                    elif trend_factor < 0.95:
                        trend_direction = "decreasing"
            
            # Generate monthly projections
            monthly_projections = []
            base_date = datetime.now()
            
            for i in range(1, forecast_months + 1):
                # Calculate future month
                future_month = (base_date.month + i - 1) % 12 + 1
                future_year = base_date.year + ((base_date.month + i - 1) // 12)
                
                # Apply slight trend adjustment
                if trend_direction == "increasing":
                    projected_value = median_monthly * (1 + 0.02 * i)
                elif trend_direction == "decreasing":
                    projected_value = median_monthly * (1 - 0.02 * i)
                else:
                    projected_value = median_monthly
                
                month_name = datetime(future_year, future_month, 1).strftime("%B %Y")
                
                monthly_projections.append({
                    "month": month_name,
                    "projected_amount": round(projected_value, 2),
                    "month_number": i
                })
            
            # Calculate totals
            total_projected = sum(p["projected_amount"] for p in monthly_projections)
            
            # Historical context
            total_historical = sum(monthly_values)
            historical_months = len(monthly_values)
            
            # Calculate confidence level based on data quality
            confidence = "high" if len(monthly_values) >= 6 else "moderate" if len(monthly_values) >= 3 else "low"
            
            # Format result
            result = {
                "success": True,
                "metric": metric,
                "forecast_period": f"Next {forecast_months} months",
                "summary": {
                    "total_projected": round(total_projected, 2),
                    "monthly_average": round(total_projected / forecast_months, 2),
                    "trend": trend_direction,
                    "confidence": confidence
                },
                "monthly_projections": monthly_projections,
                "historical_context": {
                    "months_analyzed": historical_months,
                    "historical_total": round(total_historical, 2),
                    "historical_monthly_average": round(avg_monthly, 2),
                    "period": f"{start_date.strftime('%B %Y')} to {end_date.strftime('%B %Y')}"
                },
                "notes": self._generate_forecast_notes(metric, confidence, trend_direction)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating business forecast: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Failed to generate forecast: {str(e)}"
            }
    
    def _generate_forecast_notes(
        self,
        metric: str,
        confidence: str,
        trend: str
    ) -> str:
        """Generate helpful notes for the forecast."""
        metric_name = "revenue" if metric == "revenue" else "expenses"
        
        notes = []
        
        if confidence == "low":
            notes.append(f"Limited historical data available. Forecast may vary.")
        elif confidence == "moderate":
            notes.append(f"Based on {confidence} historical data coverage.")
        
        if trend == "increasing":
            notes.append(f"Recent {metric_name} show an upward trend.")
        elif trend == "decreasing":
            notes.append(f"Recent {metric_name} show a downward trend.")
        
        if metric == "revenue":
            notes.append("Fee collection patterns can vary based on academic calendar and payment cycles.")
        else:
            notes.append("Expense patterns may vary with seasonal factors and one-time costs.")
        
        return " ".join(notes)