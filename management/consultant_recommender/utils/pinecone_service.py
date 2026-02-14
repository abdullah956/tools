"""Pinecone service for consultant search."""

import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec

from .embeddings import get_embedding

logger = logging.getLogger(__name__)

# Define the consultant JSON format template as a constant (from FastAPI)
CONSULTANT_JSON_TEMPLATE = """{{
    'name': match.metadata.get('name'),
    'expertise': match.metadata.get('expertise'),
    'experience': match.metadata.get('experience'),
    'website': match.metadata.get('website'),
    'phone': match.metadata.get('phone'),
    'gmail': match.metadata.get('gmail'),
    'apps_included': match.metadata.get('apps_included'),
    'language': match.metadata.get('language'),
    'country': match.metadata.get('country'),
    'company_name': match.metadata.get('company_name'),
    'type_of_services': match.metadata.get('type_of_services'),
    'countries_with_office_locations': match.metadata.get(
        'countries_with_office_locations'
    ),
    'about': match.metadata.get('about'),
    'date': match.metadata.get('date'),
    'time': match.metadata.get('time')
}}"""


class ConsultantPineconeService:
    """Service for searching consultants using Pinecone vector database and LLM filtering."""

    def __init__(self):
        """Initialize the Pinecone service for consultants."""
        self.pinecone_client = None
        self.index = None
        self.index_name = "consultants-index"
        self._initialize_services()

    def _initialize_services(self):
        """Initialize Pinecone client and index."""
        try:
            # Initialize Pinecone client
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                logger.error("PINECONE_API_KEY not found in environment variables")
                return

            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
            logger.info("Pinecone client initialized successfully")

            # Connect to consultants index
            self._connect_to_index()

        except Exception as e:
            logger.error(f"Error initializing Pinecone services: {e}")

    def _connect_to_index(self):
        """Connect to Pinecone consultants index."""
        try:
            if not self.pinecone_client:
                logger.error("Pinecone client not initialized")
                return

            # Check if index exists
            indexes = self.pinecone_client.list_indexes()
            if self.index_name not in [idx.name for idx in indexes]:
                logger.warning(
                    f"Pinecone index '{self.index_name}' not found. Creating it..."
                )
                self.pinecone_client.create_index(
                    name=self.index_name,
                    dimension=1536,  # OpenAI ada-002 embedding size
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                logger.info(f"Created Pinecone index: {self.index_name}")

            # Connect to index
            self.index = self.pinecone_client.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")

        except Exception as e:
            logger.error(f"Error connecting to Pinecone index: {e}")

    async def search_consultants(
        self,
        query: str,
        user_work_description: str,
        top_k: int = 10,
        namespace: str = "consultants",
    ) -> Dict[str, Any]:
        """Search for consultants using Pinecone vector search and LLM filtering.

        This matches the FastAPI implementation exactly.

        Args:
            query: The search query (user's tools/requirements)
            user_work_description: Description of user's work/project
            top_k: Number of consultants to retrieve from Pinecone
            namespace: Pinecone namespace to search in

        Returns:
            Dictionary with status, best_consultants list, and metadata
        """
        try:
            if not self.index:
                raise ValueError("Pinecone index not initialized")

            # 1. Generate embedding for query
            logger.info(f"Generating embedding for query: {query[:50]}...")
            query_embedding = get_embedding(query)
            query_embedding = np.array(query_embedding, dtype=np.float32).tolist()

            # 2. Query Pinecone
            logger.info(f"Querying Pinecone for top {top_k} consultants...")
            response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_values=True,
                include_metadata=True,
            )

            if not response.matches:
                logger.warning("No consultants found in Pinecone")
                return {
                    "status": "no_results",
                    "message": "No consultants found.",
                    "best_consultants": [],
                }

            # 3. Extract consultant metadata
            consultants = [match.metadata for match in response.matches]
            logger.info(f"Found {len(consultants)} consultants from Pinecone")

            # 4. Use LLM to filter and rank consultants (FastAPI logic)
            best_consultants = await self._filter_consultants_with_llm(
                consultants, query, user_work_description
            )

            return {
                "status": "success",
                "best_consultants": best_consultants,
                "total_found": len(consultants),
                "filtered_count": len(best_consultants),
            }

        except Exception as e:
            logger.error(f"Error searching consultants: {e}")
            return {
                "status": "error",
                "error": str(e),
                "best_consultants": [],
            }

    async def _filter_consultants_with_llm(
        self, consultants: List[Dict], query: str, user_work_description: str
    ) -> List[Dict]:
        """Filter and rank consultants using LLM (matches FastAPI implementation).

        Args:
            consultants: List of consultant metadata from Pinecone
            query: User's tools/requirements
            user_work_description: Description of user's work

        Returns:
            List of best consultants filtered by LLM
        """
        try:
            # Build prompt (exact same as FastAPI)
            prompt_text = (
                "You are the best consultant recommender. "
                "If Need translate in english Experience in english. "
                "The best choice depends on the specific needs of "
                "the business seeking consulting services. "
                "Given the following consultants with their "
                "work descriptions, select only the best "
                "ones based on expertise and experience. "
                "Return only the selected consultants as "
                "a JSON array, with each consultant as a JSON object "
                f"in the format:\n\n{CONSULTANT_JSON_TEMPLATE}\n\n"
                "Do not include any additional text or description. "
                "Consultants: {consultants}. "
                "User's work description: {user_work_description}. "
                "User's tools: {query}"
            )

            prompt_template = ChatPromptTemplate.from_template(prompt_text)

            # Initialize LLM (same model as FastAPI)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=4000)
            chain = prompt_template | llm

            logger.info("Invoking LLM to filter consultants...")
            logger.debug(f"Query: {query}")
            logger.debug(f"User work description: {user_work_description}")
            logger.debug(f"Number of consultants: {len(consultants)}")

            # Invoke LLM
            result = chain.invoke(
                {
                    "consultants": consultants,
                    "user_work_description": user_work_description,
                    "query": query,
                }
            )

            # Parse LLM response
            response_content = (
                result.content.replace("```json", "").replace("```", "").strip()
            )

            try:
                best_consultants_list = json.loads(response_content)
                logger.info(f"LLM filtered to {len(best_consultants_list)} consultants")
                return best_consultants_list
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError parsing LLM response: {e}")
                logger.error(f"Response content: {response_content}")
                # Return original consultants if parsing fails
                return consultants

        except Exception as e:
            logger.error(f"Error filtering consultants with LLM: {e}")
            # Return original consultants if LLM filtering fails
            return consultants


# Create a singleton instance
consultant_pinecone_service = ConsultantPineconeService()
