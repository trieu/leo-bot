

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from uuid import UUID, uuid4

from typing import List, Optional
from pydantic import BaseModel, HttpUrl
from datetime import datetime, timezone
from enum import Enum

from leoai.db_utils import get_async_pg_conn
  
MAX_DOC_TEXT_LENGTH = 100000
    
# ============================================================
# ENUM Models (from SQL ENUM types)
# ============================================================

class KnowledgeSourceType(str, Enum):
    """Defines the type of knowledge source."""
    BOOK_SUMMARY = "book_summary"
    REPORT_ANALYTICS = "report_analytics"
    MARKDOWN_UPLOAD = "uploaded_document"
    WEB_PAGE = "web_page"
    OTHER = "other"


class ProcessingStatus(str, Enum):
    """Tracks the state of a document in the processing pipeline."""
    PENDING = "pending"
    PROCESSING = "processing"
    ACTIVE = "active"
    FAILED = "failed"
    ARCHIVED = "archived"


# ============================================================
# Table Models (from SQL tables)
# ============================================================

class KnowledgeSource(BaseModel):
    """
    Data model for a knowledge source document.
    Corresponds to the 'knowledge_sources' table.
    """
    id: UUID = Field(default_factory=uuid4)
    user_id: str
    tenant_id: str
    source_type: KnowledgeSourceType
    name: str = Field(..., description="e.g., 'Q3 Financial Report.pdf'")
    uri: Optional[str] = Field(None, description="Path to the original file in blob storage")
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extra info like author, source URL, etc.")
    created_at: Optional[datetime] = Field(default_factory=datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(default_factory=datetime.now(timezone.utc))

    class Config:
        # This allows the model to be created from database records
        from_attributes = True


class KnowledgeChunk(BaseModel):
    """
    Data model for a single text chunk and its embedding.
    Corresponds to the 'knowledge_chunks' table.
    """
    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    content: str = Field(..., description="The actual text chunk for embedding")
    embedding: List[float] = Field(..., description="The vector embedding for the content")
    chunk_sequence: Optional[int] = Field(None, description="The order of this chunk within the original document")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extra info like page number, section headers, etc.")
    created_at: Optional[datetime] = Field(default_factory=datetime.now(timezone.utc))

    class Config:
        from_attributes = True


# ---------------------------------------------

import logging
import json
from typing import Any, Dict, Optional, List
from uuid import UUID
from datetime import datetime

# Assume these are imported from your project structure
# from leoai.db_utils import get_async_pg_conn
# from your_models_file import KnowledgeSource, KnowledgeSourceType, ProcessingStatus

# ---------------------------------------------

logger = logging.getLogger(__name__)

class KnowledgeManager:
    """
    Manages CRUD operations for knowledge sources in the database.
    """

    async def _save_knowledge_source(
        self,
        source: KnowledgeSource
    ) -> Optional[KnowledgeSource]:
        """
        Saves a new knowledge source record to the database.

        Args:
            source: A Pydantic KnowledgeSource object to be saved.

        Returns:
            The saved KnowledgeSource object with the database-generated ID and timestamps, or None on failure.
        """
        try:
            async with await get_async_pg_conn() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        INSERT INTO knowledge_sources (
                            id, user_id, tenant_id, source_type, name, uri, status, metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id, created_at, updated_at;
                    """, (
                        source.id,
                        source.user_id,
                        source.tenant_id,
                        source.source_type.value, # Use .value for enums
                        source.name,
                        source.uri,
                        source.status.value,
                        json.dumps(source.metadata) if source.metadata else None
                    ))

                    # Fetch the returned new values from the DB
                    result = await cur.fetchone()
                    if result:
                        source.id, source.created_at, source.updated_at = result
                        await conn.commit()
                        logger.info(f"💾 Knowledge source saved successfully. ID: {source.id}")
                        return source
                    else:
                        await conn.rollback()
                        logger.error("❌ Failed to save knowledge source: No ID returned.")
                        return None

        except Exception as e:
            logger.error(f"❌ Database error while saving knowledge source: {e}")
            # In a real scenario, the connection context manager might handle rollback.
            # Explicit rollback here is for clarity if needed.
            if 'conn' in locals() and conn.is_usable():
                await conn.rollback()
            return None

    async def _list_knowledge_sources(
        self,
        user_id: str,
        tenant_id: str,
        source_type: Optional[KnowledgeSourceType] = None,
        status: Optional[ProcessingStatus] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[KnowledgeSource]:
        """
        Lists knowledge sources for a user and tenant, with optional filters.

        Args:
            user_id: The user's ID.
            tenant_id: The tenant's ID.
            source_type: Optional filter by source type.
            status: Optional filter by processing status.
            limit: The maximum number of records to return.
            offset: The number of records to skip for pagination.

        Returns:
            A list of KnowledgeSource objects.
        """
        try:
            params = [user_id, tenant_id]
            # Start with a base query
            query = "SELECT id, user_id, tenant_id, source_type, name, uri, status, metadata, created_at, updated_at FROM knowledge_sources WHERE user_id = %s AND tenant_id = %s"

            # Dynamically add filters
            if source_type:
                query += " AND source_type = %s"
                params.append(source_type.value)
            if status:
                query += " AND status = %s"
                params.append(status.value)

            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s;"
            params.extend([limit, offset])

            async with await get_async_pg_conn() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(query, tuple(params))
                    rows = await cur.fetchall()

            sources = [KnowledgeSource(
                id=row[0], user_id=row[1], tenant_id=row[2], source_type=row[3],
                name=row[4], uri=row[5], status=row[6], metadata=row[7],
                created_at=row[8], updated_at=row[9]
            ) for row in rows]
            
            logger.info(f"🔍 Found {len(sources)} knowledge sources for user={user_id}")
            return sources

        except Exception as e:
            logger.error(f"❌ Failed to list knowledge sources: {e}")
            return []

    async def _get_knowledge_source_details(
        self,
        source_id: UUID
    ) -> Optional[KnowledgeSource]:
        """
        Retrieves the full details for a single knowledge source by its ID.

        Args:
            source_id: The unique identifier of the knowledge source.

        Returns:
            A KnowledgeSource object if found, otherwise None.
        """
        try:
            async with await get_async_pg_conn() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("""
                        SELECT id, user_id, tenant_id, source_type, name, uri, status, metadata, created_at, updated_at
                        FROM knowledge_sources
                        WHERE id = %s;
                    """, (source_id,))

                    row = await cur.fetchone()

            if not row:
                logger.warning(f"⚠️ Knowledge source with ID={source_id} not found.")
                return None

            source = KnowledgeSource(
                id=row[0], user_id=row[1], tenant_id=row[2], source_type=row[3],
                name=row[4], uri=row[5], status=row[6], metadata=row[7],
                created_at=row[8], updated_at=row[9]
            )
            logger.info(f"🧠 Retrieved details for knowledge source ID={source_id}")
            return source

        except Exception as e:
            logger.error(f"❌ Failed to get knowledge source details: {e}")
            return None
        
    
   