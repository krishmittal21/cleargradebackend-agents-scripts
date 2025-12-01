import logging
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, EmailStr, Field
from google.cloud import firestore
from google.api_core.exceptions import GoogleAPICallError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ MODELS ============

class UserBase(BaseModel):
    email: EmailStr
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    ip_address: Optional[str] = None
    phone: Optional[str] = None
    organization: Optional[str] = None


class UserCreate(UserBase):
    user_id: Optional[str] = Field(
        None,
        description="Optional Firebase Auth UID to use as document ID"
    )


class UserUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    phone: Optional[str] = None
    organization: Optional[str] = None


class User(UserBase):
    user_id: str
    date_created: str
    date_edited: str

    class Config:
        from_attributes = True


# ============ SERVICE ============

class UsersService:
    def __init__(self, project_id: Optional[str] = None):
        self.db = firestore.AsyncClient(project=project_id)
        self.collection_name = "users"

    def _get_iso_timestamp(self) -> str:
        """Get current timestamp in ISO 8601 format with Z."""
        return datetime.now(timezone.utc).isoformat(
            timespec='milliseconds'
        ).replace('+00:00', 'Z')

    def _user_doc_to_model(
        self,
        user_id: str,
        data: Dict[str, Any]
    ) -> User:
        """Convert Firestore document to User model."""
        return User(
            user_id=user_id,
            email=data.get("email"),
            first_name=data.get("first_name"),
            last_name=data.get("last_name"),
            ip_address=data.get("ip_address"),
            phone=data.get("phone"),
            organization=data.get("organization"),
            date_created=data.get("date_created"),
            date_edited=data.get("date_edited"),
        )

    async def create_user(
        self,
        user_data: UserCreate,
        ip_address: Optional[str] = None
    ) -> User:
        """Create a new user."""
        try:
            # Check if email already exists
            existing = self.db.collection(
                self.collection_name
            ).where("email", "==", user_data.email).limit(1).stream()

            has_existing = False
            async for doc in existing:
                has_existing = True
                break

            if has_existing:
                raise ValueError(f"Email {user_data.email} already exists")

            # If user_id provided, check if it already exists
            if user_data.user_id:
                existing_doc = await self.db.collection(
                    self.collection_name
                ).document(user_data.user_id).get()

                if existing_doc.exists:
                    raise ValueError(
                        f"User ID {user_data.user_id} already exists"
                    )

            now = self._get_iso_timestamp()

            user_doc = {
                "email": user_data.email,
                "first_name": user_data.first_name,
                "last_name": user_data.last_name,
                "ip_address": ip_address or user_data.ip_address,
                "phone": user_data.phone,
                "organization": user_data.organization,
                "date_created": now,
                "date_edited": now,
            }

            # Use provided user_id or let Firestore generate one
            if user_data.user_id:
                user_id = user_data.user_id
                await self.db.collection(
                    self.collection_name
                ).document(user_id).set(user_doc)
                logger.info(
                    f"User created with provided ID: {user_id} "
                    f"({user_data.email})"
                )
            else:
                # add() returns (timestamp, document_reference)
                _, doc_ref = await self.db.collection(
                    self.collection_name
                ).add(user_doc)
                user_id = doc_ref.id
                logger.info(
                    f"User created with auto-generated ID: {user_id} "
                    f"({user_data.email})"
                )

            return self._user_doc_to_model(user_id, user_doc)

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except GoogleAPICallError as e:
            logger.error(f"Firestore error creating user: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error creating user: {type(e).__name__}: {e}"
            )
            raise

    async def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        try:
            doc = await self.db.collection(
                self.collection_name
            ).document(user_id).get()

            if not doc.exists:
                logger.warning(f"User not found: {user_id}")
                return None

            return self._user_doc_to_model(user_id, doc.to_dict())

        except GoogleAPICallError as e:
            logger.error(f"Firestore error getting user: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error getting user: {type(e).__name__}: {e}"
            )
            raise

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        try:
            docs = self.db.collection(
                self.collection_name
            ).where("email", "==", email).limit(1).stream()

            async for doc in docs:
                return self._user_doc_to_model(doc.id, doc.to_dict())

            logger.warning(f"User not found by email: {email}")
            return None

        except GoogleAPICallError as e:
            logger.error(f"Firestore error getting user by email: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error getting user by email: "
                f"{type(e).__name__}: {e}"
            )
            raise

    async def update_user(
        self,
        user_id: str,
        user_data: UserUpdate
    ) -> Optional[User]:
        """Update user by ID."""
        try:
            doc = await self.db.collection(
                self.collection_name
            ).document(user_id).get()

            if not doc.exists:
                logger.warning(f"User not found: {user_id}")
                return None

            update_data = user_data.model_dump(exclude_none=True)
            update_data["date_edited"] = self._get_iso_timestamp()

            await self.db.collection(
                self.collection_name
            ).document(user_id).update(update_data)

            logger.info(f"User updated: {user_id}")
            return await self.get_user(user_id)

        except GoogleAPICallError as e:
            logger.error(f"Firestore error updating user: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error updating user: "
                f"{type(e).__name__}: {e}"
            )
            raise

    async def delete_user(self, user_id: str) -> bool:
        """Delete user by ID."""
        try:
            doc = await self.db.collection(
                self.collection_name
            ).document(user_id).get()

            if not doc.exists:
                logger.warning(f"User not found: {user_id}")
                return False

            await self.db.collection(
                self.collection_name
            ).document(user_id).delete()

            logger.info(f"User deleted: {user_id}")
            return True

        except GoogleAPICallError as e:
            logger.error(f"Firestore error deleting user: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error deleting user: "
                f"{type(e).__name__}: {e}"
            )
            raise

    async def list_users(self, limit: int = 100) -> List[User]:
        """List all users."""
        try:
            docs = self.db.collection(
                self.collection_name
            ).order_by(
                "date_created",
                direction=firestore.Query.DESCENDING
            ).limit(limit).stream()

            users = []
            async for doc in docs:
                users.append(self._user_doc_to_model(doc.id, doc.to_dict()))

            return users

        except GoogleAPICallError as e:
            logger.error(f"Firestore error listing users: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error listing users: "
                f"{type(e).__name__}: {e}"
            )
            raise

    async def get_user_chat_history(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get all chat sessions for a user."""
        try:
            user = await self.get_user(user_id)
            if not user:
                logger.warning(f"User not found: {user_id}")
                return []

            docs = self.db.collection(
                "conversations"
            ).where(
                "user_id",
                "==",
                user_id
            ).order_by(
                "date_created",
                direction=firestore.Query.DESCENDING
            ).limit(limit).stream()

            sessions = []
            async for doc in docs:
                data = doc.to_dict()
                sessions.append({
                    "session_id": doc.id,
                    "user_id": user_id,
                    "message_count": len(data.get("messages", [])),
                    "date_created": data.get("date_created"),
                    "date_edited": data.get("date_edited"),
                })

            return sessions

        except GoogleAPICallError as e:
            logger.error(f"Error getting user chat history: {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error getting user chat history: "
                f"{type(e).__name__}: {e}"
            )
            raise


# ============ APP SETUP ============

project_id = (
    os.environ.get("FIRESTORE_PROJECT_ID")
    or os.environ.get("GCP_PROJECT_ID")
    or "clearmarks"
)

users_service = UsersService(project_id=project_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting Users Service (Project: {project_id})")
    yield
    logger.info("Shutting down Users Service")


app = FastAPI(
    title="Users Service",
    description="User management with Firestore",
    version="1.0",
    lifespan=lifespan,
)


# ============ ENDPOINTS ============

@app.post("/users", response_model=User)
async def create_user(
    user_data: UserCreate,
    x_forwarded_for: Optional[str] = Header(None)
) -> User:
    """Create a new user."""
    try:
        ip_address = (
            x_forwarded_for.split(",")[0].strip()
            if x_forwarded_for
            else None
        )
        user = await users_service.create_user(user_data, ip_address)
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except GoogleAPICallError as e:
        logger.error(f"Firestore API error creating user: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error creating user: {type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail="User creation failed")


@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str) -> User:
    """Get user by ID."""
    try:
        user = await users_service.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException:
        raise
    except GoogleAPICallError as e:
        logger.error(f"Firestore API error getting user: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error getting user: {type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to get user")


@app.get("/users/email/{email}", response_model=User)
async def get_user_by_email(email: str) -> User:
    """Get user by email."""
    try:
        user = await users_service.get_user_by_email(email)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException:
        raise
    except GoogleAPICallError as e:
        logger.error(f"Firestore API error getting user by email: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error getting user by email: "
            f"{type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to get user")


@app.put("/users/{user_id}", response_model=User)
async def update_user(
    user_id: str,
    user_data: UserUpdate
) -> User:
    """Update user by ID."""
    try:
        user = await users_service.update_user(user_id, user_data)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException:
        raise
    except GoogleAPICallError as e:
        logger.error(f"Firestore API error updating user: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error updating user: {type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail="User update failed")


@app.delete("/users/{user_id}")
async def delete_user(user_id: str) -> Dict[str, str]:
    """Delete user by ID."""
    try:
        success = await users_service.delete_user(user_id)
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        return {"status": "deleted", "user_id": user_id}
    except HTTPException:
        raise
    except GoogleAPICallError as e:
        logger.error(f"Firestore API error deleting user: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error deleting user: {type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail="User deletion failed")


@app.get("/users", response_model=List[User])
async def list_users(limit: int = 100) -> List[User]:
    """List all users."""
    try:
        if limit > 1000:
            limit = 1000
        users = await users_service.list_users(limit)
        return users
    except GoogleAPICallError as e:
        logger.error(f"Firestore API error listing users: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error listing users: {type(e).__name__}: {e}"
        )
        raise HTTPException(status_code=500, detail="Failed to list users")


@app.get("/users/{user_id}/chat-history")
async def get_user_chat_history(user_id: str) -> Dict[str, Any]:
    """Get all chat sessions for a user."""
    try:
        sessions = await users_service.get_user_chat_history(user_id)
        return {
            "user_id": user_id,
            "session_count": len(sessions),
            "sessions": sessions,
        }
    except GoogleAPICallError as e:
        logger.error(f"Firestore API error getting chat history: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database service unavailable"
        )
    except Exception as e:
        logger.error(
            f"Unexpected error getting user chat history: "
            f"{type(e).__name__}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to get chat history"
        )


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check."""
    return {"status": "healthy"}


@app.get("/ready")
async def ready() -> Dict[str, str]:
    """Readiness check."""
    try:
        await users_service.list_users(limit=1)
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)