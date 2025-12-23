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

class TeacherIdInput(BaseModel):
    teacher_id: int = Field(description="The unique numeric ID of the teacher.")

class AttendanceInput(BaseModel):
    class_name: str = Field(description="The exact name of the class (e.g., 'IV', 'X SCI').")
    section: str = Field(description="The section name (e.g., 'A', 'TULIPS', 'ORCHIDS').")
    date: str = Field(description="Date in YYYY-MM-DD format.")

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

@tool("list_admissions")
async def list_admissions() -> str:
    """Get all student admissions including cancelled ones."""
    result = await client.student_admission()
    if not result['success']:
        return f"Error fetching admissions: {result.get('error')}"
    return str(result.get('data', {}))

@tool("list_teachers")
async def list_teachers() -> str:
    """Get a list of all teachers in the school."""
    result = await client.teacher_list()
    if not result['success']:
        return f"Error fetching teachers: {result.get('error')}"
    return str(result.get('data', {}))

@tool("view_teacher_details", args_schema=TeacherIdInput)
async def view_teacher_details(teacher_id: int) -> str:
    """Get detailed profile information for a specific teacher by ID."""
    result = await client.teacher_view(teacher_id)
    if not result['success']:
        return f"Error fetching teacher details: {result.get('error')}"
    return str(result.get('data', {}))

@tool("get_enquiries", args_schema=DateRangeInput)
async def get_enquiries(from_date: str, to_date: str) -> str:
    """Get admission enquiries for a specific date range."""
    result = await client.enquiries(from_date, to_date)
    if not result['success']:
        return f"Error fetching enquiries: {result.get('error')}"
    return str(result.get('data', {}))

@tool("get_student_attendance", args_schema=AttendanceInput)
async def get_student_attendance(class_name: str, section: str, date: str) -> str:
    """Get student attendance for a specific class, section and date."""
    result = await client.student_attendance(class_name, section, date)
    if not result['success']:
        return f"Error fetching attendance: {result.get('error')}"
    return str(result.get('data', {}))

def get_tools():
    return [
        list_students, 
        view_student_details, 
        get_fee_report, 
        get_expense_report,
        list_admissions,
        list_teachers,
        view_teacher_details,
        get_enquiries,
        get_student_attendance
    ]