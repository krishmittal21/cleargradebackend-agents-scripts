"""
API Client Template

TODO: Rename this file to match your API (e.g., your_api_client.py)
TODO: Update the class name and methods to match your API
"""

import os
import httpx
import json
from typing import Any, Dict, Optional


class ApiClient:
    """
    Template for external API client.
    
    TODO: Rename this class to match your API (e.g., YourApiClient)
    """
    
    def __init__(self, timeout: int = 30):
        # TODO: Update these with your actual API configuration
        self.base_url = os.environ.get("YOUR_API_URL", "https://api.example.com")
        self.api_key = os.environ.get("YOUR_API_KEY")
        self.timeout = timeout
        
        if not self.api_key:
            # Log a warning in production
            pass
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generic request method for API calls.
        
        TODO: Customize this method based on your API's authentication and response format
        """
        if not self.api_key:
            return {
                'success': False,
                'error': 'Configuration Error: Missing API key'
            }
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                if method.upper() == 'GET':
                    resp = await client.get(
                        f"{self.base_url}/{endpoint}",
                        params=params,
                        headers=headers
                    )
                elif method.upper() == 'POST':
                    resp = await client.post(
                        f"{self.base_url}/{endpoint}",
                        params=params,
                        json=json_data,
                        headers=headers
                    )
                else:
                    return {'success': False, 'error': f'Unsupported method: {method}'}
                
                resp.raise_for_status()
                
                try:
                    data = resp.json()
                    return {'success': True, 'data': data}
                except json.JSONDecodeError:
                    return {
                        'success': False,
                        'error': 'Invalid JSON response from server',
                        'raw': resp.text
                    }
                
            except httpx.HTTPStatusError as e:
                return {
                    'success': False,
                    'error': f'HTTP Error: {e.response.status_code}'
                }
            except httpx.RequestError as e:
                return {
                    'success': False,
                    'error': f'Connection Error: {str(e)}'
                }
    
    # TODO: Add your API methods below
    
    async def list_items(self, category: str) -> Dict[str, Any]:
        """
        Example method to list items.
        
        TODO: Implement your actual API endpoint
        """
        return await self._request('GET', 'items', params={'category': category})
    
    async def get_item(self, item_id: int) -> Dict[str, Any]:
        """
        Example method to get item details.
        
        TODO: Implement your actual API endpoint
        """
        return await self._request('GET', f'items/{item_id}')
    
    async def create_item(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Example method to create an item.
        
        TODO: Implement your actual API endpoint
        """
        return await self._request('POST', 'items', json_data=data)
    
    async def generate_report(
        self,
        from_date: str,
        to_date: str
    ) -> Dict[str, Any]:
        """
        Example method to generate a report.
        
        TODO: Implement your actual API endpoint
        """
        return await self._request(
            'GET',
            'reports',
            params={'from': from_date, 'to': to_date}
        )
