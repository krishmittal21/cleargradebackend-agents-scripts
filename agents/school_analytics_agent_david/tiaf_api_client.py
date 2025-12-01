import os
import httpx
import json
from typing import Any, Dict, Optional

class TIAFApiClient:
    def __init__(self, timeout: int = 30):
        self.base_url = "https://tiaf.coradius.in/api/api"
        self.secret_key = "f83b2f95ad404f45936f1237ef59204c"
        self.timeout = timeout
        
        if not self.secret_key:
            # Log a warning in production
            pass

    async def _request(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.secret_key:
            return {'success': False, 'error': 'Configuration Error: Missing TIAF_SECRET_KEY'}
        
        payload = params.copy()
        payload['key'] = self.secret_key
        payload['action'] = action
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(self.base_url, params=payload)
                resp.raise_for_status()
                
                try:
                    data = resp.json()
                    return {'success': True, 'data': data}
                except json.JSONDecodeError:
                    return {'success': False, 'error': 'Invalid JSON response from server', 'raw': resp.text}
                    
            except httpx.HTTPStatusError as e:
                return {'success': False, 'error': f'HTTP Error: {e.response.status_code}'}
            except httpx.RequestError as e:
                return {'success': False, 'error': f'Connection Error: {str(e)}'}

    async def student_list(self, class_name: str) -> Dict[str, Any]:
        return await self._request('student_list', {'class': class_name})

    async def student_view(self, student_id: int) -> Dict[str, Any]:
        return await self._request('student_view', {'id': student_id})

    async def fee_report(self, from_date: str, to_date: str) -> Dict[str, Any]:
        return await self._request('fee_report', {'from': from_date, 'to': to_date})

    async def expense_report(self, from_date: str, to_date: str) -> Dict[str, Any]:
        return await self._request('expense_report', {'from': from_date, 'to': to_date})