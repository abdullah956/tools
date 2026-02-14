"""Pinecone indexing service."""

import logging
import os

from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)


class PineconeService:
    """Service for indexing embeddings to Pinecone."""

    def __init__(self):
        """Initialize PineconeService with API credentials."""
        self.api_key = os.environ.get("PINECONE_API_KEY")
        self.environment = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = os.environ.get("PINECONE_TOOL_INDEX", "scraping-tool-index")

        if not self.api_key:
            logger.warning("PINECONE_API_KEY not found in environment")

        if not self.index_name:
            logger.error(
                "PINECONE_TOOL_INDEX not found in environment, using default 'scraping-tool-index'"
            )
            self.index_name = "scraping-tool-index"

        self.pc = Pinecone(api_key=self.api_key) if self.api_key else None
        self.index = None

        if self.pc:
            self._initialize_index()

    def _initialize_index(self):
        """Initializes or creates the Pinecone index."""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()

            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")

                # Use a valid AWS region for serverless (default to us-east-1)
                # Common valid regions: us-east-1, us-west-2, eu-west-1
                valid_region = "us-east-1"
                if self.environment in [
                    "us-east-1",
                    "us-west-2",
                    "eu-west-1",
                    "ap-southeast-1",
                ]:
                    valid_region = self.environment
                else:
                    logger.warning(
                        f"Invalid region '{self.environment}', using default: {valid_region}"
                    )

                # Create index with appropriate dimensions (1536 for text-embedding-3-small)
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=valid_region),
                )
                logger.info(
                    f"Created index: {self.index_name} in region {valid_region}"
                )
            else:
                logger.info(f"Index {self.index_name} already exists, connecting to it")

            # Get index instance
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")

        except Exception as e:
            logger.exception(f"Error initializing Pinecone index: {e}")
            self.index = None

    def index_site(self, site):
        """
        Indexes all chunks from a site to Pinecone with minimal metadata.

        Each chunk gets the same UUID (from CombinedText) but different vector IDs.
        Minimal schema: ID, category, description, master_category, text, title, website

        Args:
            site: ToolSite model instance

        Returns:
            bool: Success status
        """
        if not self.index:
            logger.error("Pinecone index not initialized")
            return False

        try:
            combined_text_record = site.combined_text_record
            if not combined_text_record:
                logger.warning(f"No combined text record found for site {site.id}")
                return False

            # Generate chunks in memory
            from .chunking import TextChunker

            chunker = TextChunker()
            chunks_data = chunker.chunk(combined_text_record.combined_text)

            if not chunks_data:
                logger.warning(
                    f"No chunks generated for combined text {combined_text_record.id}"
                )
                return False

            from .llm import EmbeddingService

            embedding_service = EmbeddingService()
            vectors_to_upsert = []

            # Same UUID for all chunks from this tool
            tool_uuid = str(combined_text_record.id)

            for idx, chunk_text in enumerate(chunks_data):
                # Prepare metadata for embedding
                metadata_fields = {
                    "title": site.title or "",
                    "category": site.category or "",
                    "master_category": site.master_category or "",
                    "description": site.description[:500] if site.description else "",
                    "website": site.website or "",
                }

                # Combine metadata with chunk text for embedding
                # This ensures embeddings capture both content and context
                metadata_text = " | ".join(
                    [f"{k}: {v}" for k, v in metadata_fields.items() if v]
                )
                text_to_embed = f"{metadata_text}\n\n{chunk_text}"

                # Generate embedding with metadata included
                embedding = embedding_service.generate(text_to_embed)
                if not embedding:
                    logger.warning(
                        f"Failed to generate embedding for chunk {idx} of tool {tool_uuid}"
                    )
                    continue

                # Vector ID: UUID + chunk index for uniqueness in Pinecone
                vector_id = f"{tool_uuid}_{idx}"

                # MINIMAL metadata schema - same as second picture
                metadata = {
                    "ID": tool_uuid,  # Same UUID for all chunks from this tool
                    "category": site.category or "",
                    "description": site.description[:500] if site.description else "",
                    "master_category": site.master_category or "",
                    "text": chunk_text,  # Full chunk text
                    "title": site.title or "",
                    "website": site.website or "",
                }

                vectors_to_upsert.append(
                    {"id": vector_id, "values": embedding, "metadata": metadata}
                )

            if vectors_to_upsert:
                success = self.upsert_vectors(vectors_to_upsert)
                if success:
                    logger.info(
                        f"Upserted {len(vectors_to_upsert)} chunks for tool {tool_uuid}"
                    )
                return success
            else:
                logger.warning(f"No vectors to upsert for tool {tool_uuid}")
                return False

        except Exception as e:
            logger.exception(f"Error indexing site to Pinecone: {e}")
            return False

    def upsert_vectors(self, vectors):
        """
        Upsert vectors directly to Pinecone.

        Args:
            vectors: List of vector dictionaries with id, values, metadata

        Returns:
            bool: Success status
        """
        if not self.index:
            logger.error("Pinecone index not initialized")
            return False

        try:
            # Use default namespace for all vectors
            namespace = ""  # Empty string = default namespace in Pinecone

            # Upsert in batches of 100
            batch_size = 100
            total_upserted = 0

            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                total_upserted += len(batch)
                logger.debug(
                    f"Upserted batch {i // batch_size + 1}: {len(batch)} vectors"
                )

            logger.info(f"Successfully upserted {total_upserted} vectors to Pinecone")
            return True

        except Exception as e:
            logger.exception(f"Error upserting vectors: {e}")
            return False

    def search(self, query_embedding, namespace="", top_k=10, filter=None):
        """
        Searches Pinecone index for similar vectors.

        Args:
            query_embedding (list): Query vector
            namespace (str): Namespace to search
            top_k (int): Number of results
            filter (dict): Metadata filters

        Returns:
            dict: Search results
        """
        if not self.index:
            logger.error("Pinecone index not initialized")
            return {"matches": []}

        try:
            results = self.index.query(
                vector=query_embedding,
                namespace=namespace,
                top_k=top_k,
                filter=filter,
                include_metadata=True,
            )

            logger.info(f"Found {len(results['matches'])} matches")
            return results

        except Exception as e:
            logger.exception(f"Error searching Pinecone: {e}")
            return {"matches": []}

    def delete_site(self, site):
        """
        Deletes all vectors for a site from Pinecone.

        Args:
            site: ToolSite model instance

        Returns:
            bool: Success status
        """
        if not self.index:
            return False

        try:
            # Use default namespace (empty string)
            namespace = ""

            # Delete by prefix (all vectors starting with site_id::)
            # Vector IDs are formatted as "site_id::chunk_index"
            self.index.delete(
                delete_all=False,
                namespace=namespace,
                filter={
                    "chunk_id": {"$exists": True}
                },  # Delete all with chunk_id metadata
            )

            logger.info(f"Deleted vectors for site {site.id} from Pinecone")
            return True

        except Exception as e:
            logger.exception(f"Error deleting site from Pinecone: {e}")
            return False

    def clear_all_vectors(self, namespace=None):
        """
        Clears all vectors from Pinecone index.

        Args:
            namespace (str): Specific namespace to clear, or None to clear all namespaces

        Returns:
            bool: Success status
        """
        if not self.index:
            logger.error("Pinecone index not initialized")
            return False

        try:
            if namespace:
                # Clear specific namespace
                self.index.delete(delete_all=True, namespace=namespace)
                logger.info(f"Cleared all vectors from namespace '{namespace}'")
            else:
                # Clear all namespaces - need to get stats first
                stats = self.index.describe_index_stats()
                namespaces = stats.get("namespaces", {}).keys()

                for ns in namespaces:
                    self.index.delete(delete_all=True, namespace=ns)
                    logger.info(f"Cleared namespace '{ns}'")

                # Also clear default namespace
                self.index.delete(delete_all=True, namespace="")
                logger.info("Cleared default namespace")

            logger.info("Successfully cleared Pinecone vectors")
            return True

        except Exception as e:
            logger.exception(f"Error clearing Pinecone vectors: {e}")
            return False
