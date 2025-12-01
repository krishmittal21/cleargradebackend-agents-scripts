import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from google.cloud import firestore
from google.api_core.exceptions import GoogleAPICallError
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.chat_history import BaseChatMessageHistory

logger = logging.getLogger(__name__)

MESSAGE_TYPE_MAP = {
    HumanMessage: "human",
    AIMessage: "ai",
}

ROLE_TO_MESSAGE = {
    "human": HumanMessage,
    "ai": AIMessage,
}


class AsyncFirestoreChatMessageHistory(BaseChatMessageHistory):
    """Async Firestore-backed chat message history."""

    def __init__(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        collection_name: str = "conversations",
        ttl_days: int = 30,
        max_messages: int = 500,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.collection_name = collection_name
        self.ttl_days = ttl_days
        self.max_messages = max_messages

        self._db: Optional[firestore.AsyncClient] = None
        self._messages: List[BaseMessage] = []
        self._lock = asyncio.Lock()
        self._init_db(project_id)

    def _init_db(self, project_id: Optional[str]) -> None:
        """Initialize Firestore async client."""
        try:
            self._db = (
                firestore.AsyncClient(project=project_id)
                if project_id
                else firestore.AsyncClient()
            )
        except GoogleAPICallError as e:
            logger.error(f"Failed to initialize Firestore: {e}")
            self._db = None

    def _doc_ref(self) -> Optional[firestore.AsyncDocumentReference]:
        """Get document reference."""
        if not self._db:
            return None
        return self._db.collection(self.collection_name).document(
            self.session_id
        )

    async def _load(self) -> None:
        """Load messages from Firestore."""
        if not self._db:
            self._messages = []
            return

        try:
            doc = await self._doc_ref().get()
            if doc.exists:
                data = doc.to_dict() or {}
                items = data.get("messages", [])

                msgs: List[BaseMessage] = []
                for msg_data in items:
                    role = msg_data.get("role")
                    content = msg_data.get("content", "")

                    if role in ROLE_TO_MESSAGE:
                        msgs.append(ROLE_TO_MESSAGE[role](content=content))

                self._messages = msgs
            else:
                self._messages = []
        except GoogleAPICallError as e:
            logger.error(f"Error loading messages: {e}")
            self._messages = []

    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages."""
        return list(self._messages)

    async def aadd_message(self, message: BaseMessage) -> None:
        """Add a message asynchronously."""
        async with self._lock:
            self._messages.append(message)
            await self._persist_one(message)
            await self._cleanup_if_needed()

    async def aadd_user_message(self, message: str) -> None:
        """Add a user message."""
        await self.aadd_message(HumanMessage(content=message))

    async def aadd_ai_message(self, message: str) -> None:
        """Add an AI message."""
        await self.aadd_message(AIMessage(content=message))

    async def aclear(self) -> None:
        """Clear all messages."""
        async with self._lock:
            self._messages = []
            if self._db:
                try:
                    await self._doc_ref().set(
                        {
                            "messages": [],
                            "user_id": self.user_id,
                            "clearedAt": datetime.utcnow().isoformat(),
                            "expiresAt": self._calculate_ttl(),
                        }
                    )
                except GoogleAPICallError as e:
                    logger.error(f"Error clearing messages: {e}")

    async def _persist_one(self, message: BaseMessage) -> None:
        """Persist a single message."""
        if not self._db:
            return

        try:
            doc_ref = self._doc_ref()
            doc = await doc_ref.get()

            data = doc.to_dict() if doc.exists else {}
            items = data.get("messages", [])

            role = MESSAGE_TYPE_MAP.get(type(message), "unknown")
            items.append(
                {
                    "role": role,
                    "content": message.content,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            await doc_ref.set(
                {
                    "messages": items,
                    "user_id": self.user_id,
                    "date_created": data.get(
                        "date_created",
                        datetime.utcnow().isoformat()
                    ),
                    "updatedAt": datetime.utcnow().isoformat(),
                    "expiresAt": self._calculate_ttl(),
                },
                merge=False,
            )
        except GoogleAPICallError as e:
            logger.error(f"Error persisting message: {e}")

    async def _cleanup_if_needed(self) -> None:
        """Remove old messages if count exceeds limit."""
        if len(self._messages) <= self.max_messages:
            return

        self._messages = self._messages[-self.max_messages :]

        if self._db:
            try:
                await self._doc_ref().set(
                    {
                        "messages": [
                            {
                                "role": MESSAGE_TYPE_MAP.get(
                                    type(m), "unknown"
                                ),
                                "content": m.content,
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                            for m in self._messages
                        ],
                        "user_id": self.user_id,
                        "updatedAt": datetime.utcnow().isoformat(),
                        "expiresAt": self._calculate_ttl(),
                    },
                    merge=False,
                )
            except GoogleAPICallError as e:
                logger.error(f"Error cleaning up messages: {e}")

    def _calculate_ttl(self) -> str:
        """Calculate TTL timestamp."""
        ttl = datetime.utcnow() + timedelta(days=self.ttl_days)
        return ttl.isoformat()

    def add_message(self, message: BaseMessage) -> None:
        logger.warning(
            "Sync add_message called. Use aadd_message instead."
        )
        raise RuntimeError(
            "Use async method: await history.aadd_message(message)"
        )

    def add_user_message(self, message: str) -> None:
        logger.warning(
            "Sync add_user_message called. Use aadd_user_message instead."
        )
        raise RuntimeError(
            "Use async method: await history.aadd_user_message(message)"
        )

    def add_ai_message(self, message: str) -> None:
        logger.warning(
            "Sync add_ai_message called. Use aadd_ai_message instead."
        )
        raise RuntimeError(
            "Use async method: await history.aadd_ai_message(message)"
        )

    def clear(self) -> None:
        logger.warning("Sync clear called. Use aclear instead.")
        raise RuntimeError("Use async method: await history.aclear()")


@asynccontextmanager
async def get_chat_history(
    session_id: str,
    user_id: Optional[str] = None,
    project_id: Optional[str] = None,
    collection_name: str = "conversations",
):
    pid = (
        project_id
        or os.environ.get("FIRESTORE_PROJECT_ID")
        or os.environ.get("GCP_PROJECT_ID")
        or "clearmarks"
    )
    history = AsyncFirestoreChatMessageHistory(
        session_id=session_id,
        user_id=user_id,
        project_id=pid,
        collection_name=collection_name,
    )
    await history._load()
    try:
        yield history
    finally:
        pass