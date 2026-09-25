"""Durable, local, append-only conversation history (``conversations.db``)."""

from openjarvis.conversations.history import (
    CONTEXT_MAX_MESSAGES,
    CONTEXT_MAX_TOKENS,
    RESUME_MAX_MESSAGES,
    RESUME_MAX_TOKENS,
    ResumeContext,
    build_resume_context,
    normalize_context_messages,
)
from openjarvis.conversations.recorder import ConversationRecorder, ResumedConversation
from openjarvis.conversations.store import (
    MAX_METADATA_BYTES,
    OWNER_USER_ID,
    ROLES,
    SCHEMA_VERSION,
    Conversation,
    ConversationMessage,
    ConversationNotFound,
    ConversationStore,
    ConversationStoreError,
)

__all__ = [
    "CONTEXT_MAX_MESSAGES",
    "CONTEXT_MAX_TOKENS",
    "MAX_METADATA_BYTES",
    "OWNER_USER_ID",
    "RESUME_MAX_MESSAGES",
    "RESUME_MAX_TOKENS",
    "ROLES",
    "SCHEMA_VERSION",
    "Conversation",
    "ConversationMessage",
    "ConversationNotFound",
    "ConversationRecorder",
    "ConversationStore",
    "ConversationStoreError",
    "ResumeContext",
    "ResumedConversation",
    "build_resume_context",
    "normalize_context_messages",
]
