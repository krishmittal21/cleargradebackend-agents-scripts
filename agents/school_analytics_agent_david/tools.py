from typing import Type
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tiaf_api_client import TIAFApiClient

# Initialize client once
client = TIAFApiClient()

class ClassNameInput(BaseModel):
    class_name: str = Field(description="The exact name of the class (e.g., 'X SCI', 'Nursery', 'I').")

class StudentIdInput(BaseModel):
    student_id: int = Field(description="The unique numeric ID of the student.")

class DateRangeInput(BaseModel):
    from_date: str = Field(description="Start date in YYYY-MM-DD format.")
    to_date: str = Field(description="End date in YYYY-MM-DD format.")

@tool("list_students", args_schema=ClassNameInput)
async def list_students(class_name: str) -> str:
    """Fetch a list of students for a specific class."""
    result = await client.student_list(class_name)
    if not result['success']:
        return f"Error fetching students: {result.get('error')}"
    
    # Minimize token usage by summarizing if necessary, or returning raw data
    data = result.get('data', {})
    return str(data)

@tool("view_student_details", args_schema=StudentIdInput)
async def view_student_details(student_id: int) -> str:
    """Get detailed profile information for a specific student by ID."""
    result = await client.student_view(student_id)
    if not result['success']:
        return f"Error fetching student details: {result.get('error')}"
    return str(result.get('data', {}))

@tool("get_fee_report", args_schema=DateRangeInput)
async def get_fee_report(from_date: str, to_date: str) -> str:
    """Get the fees collected report for a specific date range."""
    result = await client.fee_report(from_date, to_date)
    if not result['success']:
        return f"Error fetching fees: {result.get('error')}"
    return str(result.get('data', {}))

@tool("get_expense_report", args_schema=DateRangeInput)
async def get_expense_report(from_date: str, to_date: str) -> str:
    """Get the school expenditure report for a specific date range."""
    result = await client.expense_report(from_date, to_date)
    if not result['success']:
        return f"Error fetching expenses: {result.get('error')}"
    return str(result.get('data', {}))

def get_tools():
    return [list_students, view_student_details, get_fee_report, get_expense_report]