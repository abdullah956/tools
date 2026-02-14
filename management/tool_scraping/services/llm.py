"""LLM and embedding services for metadata extraction and embedding generation."""

import json
import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM-based metadata extraction."""

    def __init__(self):
        """Initialize LLMService with OpenAI client."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = "gpt-4o-mini"  # Cost-effective model for extraction

    def extract_metadata(self, text, existing_data=None):
        """
        Extracts ~200 metadata fields from text using LLM.

        Args:
            text (str): Combined site text
            existing_data (dict): Existing data from CSV

        Returns:
            dict: Dictionary of extracted metadata fields
        """
        if not self.client:
            logger.error("OpenAI client not initialized")
            return existing_data or {}

        try:
            # Truncate text if too long (keep first 10000 chars)
            text_sample = text[:10000] if len(text) > 10000 else text

            # Build prompt for metadata extraction
            prompt = self._build_extraction_prompt(text_sample, existing_data)

            logger.info(f"Extracting metadata using {self.model}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI tool metadata extraction expert. "
                            "Extract detailed, accurate metadata from website content."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                response_format={"type": "json_object"},
            )

            result = response.choices[0].message.content
            metadata = json.loads(result)

            logger.info(f"Extracted {len(metadata)} metadata fields")
            return metadata

        except Exception as e:
            logger.exception(f"Error extracting metadata: {e}")
            return existing_data or {}

    def _build_extraction_prompt(self, text, existing_data):
        """Builds the extraction prompt for the LLM."""
        existing_json = json.dumps(existing_data, indent=2) if existing_data else "{}"

        return f"""Extract comprehensive metadata about this AI tool from the following text.

Existing data from CSV:
{existing_json}

Website content:
{text}

Extract and return a JSON object with the following fields (provide empty string if not found):

Core Fields:
- title: Tool name
- description: Detailed description
- category: Primary category
- master_category: High-level category
- price: Pricing information
- pricing_model: Pricing model (free/freemium/paid/subscription)
- use_cases: Array of use cases
- target_audience: Target users
- key_features: Array of main features

Company Info:
- company_name: Company/creator name
- company_website: Company website
- founded_year: Year founded
- headquarters_location: Location

Technical Details:
- api_available: boolean
- integrations: Array of integrations
- platforms: Array of supported platforms (web/ios/android/desktop)
- languages_supported: Array of supported languages

Social & Support:
- twitter_url: Twitter/X profile
- linkedin_url: LinkedIn profile
- github_url: GitHub repository
- youtube_url: YouTube channel
- instagram_url: Instagram
- tiktok_url: TikTok
- support_email: Support email
- documentation_url: Documentation URL

Additional Fields (extract if found):
- trust_score: Estimated trust score (0-100)
- user_reviews_summary: Summary of user feedback
- competitors: Array of competing tools
- unique_selling_points: Array of USPs
- limitations: Known limitations
- security_features: Security capabilities
- compliance_certifications: Compliance certifications
- last_updated: Last update date

Prefer scraped values over existing CSV data if they appear more accurate or detailed.
Return pure JSON without markdown formatting."""


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self):
        """Initialize EmbeddingService with OpenAI client."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = "text-embedding-3-small"  # 1536 dimensions

    def generate(self, text):
        """
        Generates embedding vector for text.

        Args:
            text (str): Text to embed

        Returns:
            list: Embedding vector (1536 dimensions) as JSON array
        """
        if not self.client:
            logger.error("OpenAI client not initialized")
            return None

        try:
            # Truncate if too long (OpenAI has 8191 token limit)
            text = text[:8000] if len(text) > 8000 else text

            response = self.client.embeddings.create(model=self.model, input=text)

            embedding = response.data[0].embedding
            logger.debug(f"Generated {len(embedding)}-dimensional embedding")

            return embedding  # Returns as list/array

        except Exception as e:
            logger.exception(f"Error generating embedding: {e}")
            return None

    def generate_batch(self, texts):
        """
        Generates embeddings for multiple texts in a batch.

        Args:
            texts (list): List of texts to embed

        Returns:
            list: List of embedding vectors
        """
        if not self.client:
            return [None] * len(texts)

        try:
            # Truncate all texts
            texts = [t[:8000] if len(t) > 8000 else t for t in texts]

            response = self.client.embeddings.create(model=self.model, input=texts)

            embeddings = [item.embedding for item in response.data]
            logger.info(f"Generated {len(embeddings)} embeddings")

            return embeddings

        except Exception as e:
            logger.exception(f"Error generating batch embeddings: {e}")
            return [None] * len(texts)
