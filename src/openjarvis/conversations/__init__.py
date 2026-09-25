"""Durable, local, append-only conversation history (``conversations.db``)."""

from openjarvis.conversations.recorder import ConversationRecorder
from openjarvis.conversations.store import (
    MAX_METADATA_BYTES,
    OWNER_USER_ID,
    ROLES,
    SCHEMA_VERSION,
    Conversation,
    ConversationMessage,
    ConversationStore,
    ConversationStoreError,
)

__all__ = [
    "MAX_METADATA_BYTES",
    "OWNER_USER_ID",
    "ROLES",
    "SCHEMA_VERSION",
    "Conversation",
    "ConversationMessage",
    "ConversationRecorder",
    "ConversationStore",
    "ConversationStoreError",
]
