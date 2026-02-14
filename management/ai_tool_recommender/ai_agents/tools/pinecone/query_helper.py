"""Pinecone query utilities and helpers."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PineconeQueryHelper:
    """Helper class for Pinecone queries."""

    @staticmethod
    def prepare_query_embedding(query: str, embeddings) -> Optional[List[float]]:
        """Prepare query embedding for Pinecone search.

        Args:
            query: Search query text
            embeddings: Embeddings model

        Returns:
            Query embedding vector or None if failed
        """
        try:
            query_embedding = embeddings.embed_query(query)
            query_embedding = np.array(query_embedding, dtype=np.float32).tolist()
            logger.info(
                f"Generated query embedding with dimension: {len(query_embedding)}"
            )
            return query_embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return None

    @staticmethod
    def process_search_results(results, namespace: str = None) -> List[Dict[str, Any]]:
        """Process Pinecone search results.

        Args:
            results: Pinecone query results
            namespace: Namespace used for the query

        Returns:
            List of processed tool data
        """
        tools = []

        for match in results.matches:
            metadata = match.metadata
            if metadata:
                # Ensure source is set
                if "Source" not in metadata:
                    metadata["Source"] = "Pinecone Vector Database"

                # Add namespace info if available
                if namespace:
                    metadata["Namespace"] = namespace

                # Add similarity score
                metadata["Similarity Score"] = match.score

                tools.append(metadata)
                logger.info(f"Added Pinecone tool: {metadata.get('Title', 'Unknown')}")

        logger.info(f"Processed {len(tools)} tools from Pinecone results")
        return tools

    @staticmethod
    def filter_results_by_relevance(
        tools: List[Dict[str, Any]], min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Filter results by similarity score.

        Args:
            tools: List of tools with similarity scores
            min_score: Minimum similarity score threshold

        Returns:
            Filtered list of tools
        """
        filtered_tools = [
            tool for tool in tools if tool.get("Similarity Score", 0) >= min_score
        ]

        logger.info(
            f"Filtered {len(tools)} tools to {len(filtered_tools)} by relevance (min_score: {min_score})"
        )
        return filtered_tools

    @staticmethod
    def sort_results_by_score(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort results by similarity score (descending).

        Args:
            tools: List of tools with similarity scores

        Returns:
            Sorted list of tools
        """
        sorted_tools = sorted(
            tools, key=lambda x: x.get("Similarity Score", 0), reverse=True
        )

        logger.info(f"Sorted {len(tools)} tools by similarity score")
        return sorted_tools
