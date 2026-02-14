"""Scalable prompt template system for AI tool validation and enhancement."""

import logging
from enum import Enum
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of prompts available in the system."""

    CATEGORY_GENERATION = "category_generation"
    DESCRIPTION_GENERATION = "description_generation"
    FEATURES_GENERATION = "features_generation"
    TAGS_GENERATION = "tags_generation"
    PRICING_ANALYSIS = "pricing_analysis"
    TOOL_VALIDATION = "tool_validation"
    DATA_ENHANCEMENT = "data_enhancement"


class PromptTemplate:
    """Base class for prompt templates."""

    def __init__(
        self, prompt_type: PromptType, template: str, system_context: str = ""
    ):
        """Initialize prompt template with type, template, and system context."""
        self.prompt_type = prompt_type
        self.template = template
        self.system_context = system_context

    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing required variable {e} for prompt {self.prompt_type}")
            raise


class PromptSystem:
    """Centralized prompt management system."""

    def __init__(self):
        """Initialize prompt system with all templates."""
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[PromptType, PromptTemplate]:
        """Initialize all prompt templates."""
        return {
            PromptType.CATEGORY_GENERATION: PromptTemplate(
                PromptType.CATEGORY_GENERATION,
                """
                SYSTEM: You are an AI tool categorization expert. Your task is to determine the most appropriate category for AI tools based on their information.

                CATEGORIES: {valid_categories}

                TASK: Categorize this AI tool:
                Title: {title}
                Description: {description}
                Website: {website}

                INSTRUCTIONS:
                1. Analyze the tool's primary function and purpose
                2. Select ONE category from the provided list
                3. Return ONLY the category name, nothing else

                CATEGORY:
                """,
                "AI tool categorization expert",
            ),
            PromptType.DESCRIPTION_GENERATION: PromptTemplate(
                PromptType.DESCRIPTION_GENERATION,
                """
                SYSTEM: You are a technical writer specializing in AI tool descriptions. Create clear, professional descriptions.

                TASK: Write a brief description (1-2 sentences) for this AI tool:
                Title: {title}
                Website: {website}
                Features: {features}

                REQUIREMENTS:
                - Keep it concise and professional
                - Focus on what the tool does and its main purpose
                - Use clear, accessible language

                DESCRIPTION:
                """,
                "Technical writer for AI tools",
            ),
            PromptType.FEATURES_GENERATION: PromptTemplate(
                PromptType.FEATURES_GENERATION,
                """
                SYSTEM: You are an AI tool analyst. Extract and list key features from tool information.

                TASK: List 3-5 key features for this AI tool:
                Title: {title}
                Description: {description}
                Category: {category}

                REQUIREMENTS:
                - List main features this tool offers
                - Return as comma-separated list
                - Focus on unique capabilities

                FEATURES:
                """,
                "AI tool analyst",
            ),
            PromptType.TAGS_GENERATION: PromptTemplate(
                PromptType.TAGS_GENERATION,
                """
                SYSTEM: You are a keyword specialist for AI tools. Generate relevant tags and keywords.

                TASK: Generate relevant keywords/tags for this tool:
                Title: {title}
                Description: {description}
                Category: {category}

                REQUIREMENTS:
                - Return 3-5 relevant keywords as comma-separated list
                - Include the category and related terms
                - Focus on searchable terms

                TAGS:
                """,
                "AI tool keyword specialist",
            ),
            PromptType.PRICING_ANALYSIS: PromptTemplate(
                PromptType.PRICING_ANALYSIS,
                """
                SYSTEM: You are a pricing analyst for software tools. Analyze pricing information from content.

                TASK: Analyze pricing information from this content:
                Content: {content}
                URL: {url}

                REQUIREMENTS:
                - Extract pricing details if mentioned
                - Identify pricing model (free, freemium, paid, contact sales)
                - Return structured pricing information

                PRICING ANALYSIS:
                """,
                "Software pricing analyst",
            ),
            PromptType.TOOL_VALIDATION: PromptTemplate(
                PromptType.TOOL_VALIDATION,
                """
                SYSTEM: You are a tool validation expert. Verify and enhance tool data completeness.

                TASK: Validate and enhance this tool data:
                {tool_data}

                VALIDATION CRITERIA:
                - Required fields: Title, Description, Category, Website, Price From
                - Optional fields: Features, Tags, Social Media Links, Price To
                - Data quality and completeness

                VALIDATION RESULT:
                """,
                "Tool validation expert",
            ),
            PromptType.DATA_ENHANCEMENT: PromptTemplate(
                PromptType.DATA_ENHANCEMENT,
                """
                SYSTEM: You are a data enhancement specialist. Improve incomplete tool data using available information.

                TASK: Enhance this incomplete tool data:
                {tool_data}

                MISSING FIELDS: {missing_fields}

                ENHANCEMENT RULES:
                - Use available information to fill gaps
                - Maintain data consistency
                - Provide reasonable defaults for missing data
                - Ensure all required fields are present

                ENHANCED DATA:
                """,
                "Data enhancement specialist",
            ),
        }

    def get_template(self, prompt_type: PromptType) -> PromptTemplate:
        """Get a prompt template by type."""
        if prompt_type not in self.templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        return self.templates[prompt_type]

    def format_prompt(self, prompt_type: PromptType, **kwargs) -> str:
        """Format a prompt template with variables."""
        template = self.get_template(prompt_type)
        return template.format(**kwargs)

    def get_system_context(self, prompt_type: PromptType) -> str:
        """Get system context for a prompt type."""
        template = self.get_template(prompt_type)
        return template.system_context


class PromptProcessor:
    """Processes prompts using the LLM service."""

    def __init__(self, llm_service):
        """Initialize prompt processor with LLM service."""
        self.llm_service = llm_service
        self.prompt_system = PromptSystem()

    async def process_prompt(self, prompt_type: PromptType, **kwargs) -> str:
        """Process a prompt and return the LLM response."""
        try:
            prompt = self.prompt_system.format_prompt(prompt_type, **kwargs)
            response = await self.llm_service.generate_response(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error processing prompt {prompt_type}: {e}")
            raise

    async def process_json_prompt(
        self, prompt_type: PromptType, **kwargs
    ) -> Dict[str, Any]:
        """Process a prompt and return parsed JSON response."""
        try:
            response = await self.process_prompt(prompt_type, **kwargs)
            return await self.llm_service.parse_json_response(response)
        except Exception as e:
            logger.error(f"Error processing JSON prompt {prompt_type}: {e}")
            raise


class ToolDataEnhancer:
    """Enhanced tool data processor using the prompt system."""

    def __init__(self, prompt_processor: PromptProcessor):
        """Initialize tool data enhancer with prompt processor."""
        self.processor = prompt_processor

        # Valid categories for categorization
        self.valid_categories = [
            "Video Editing",
            "Content Creation",
            "Design & Visualization",
            "Productivity",
            "Marketing",
            "Analytics",
            "Development",
            "Communication",
            "Education",
            "E-commerce",
            "Healthcare",
            "Finance",
            "Gaming",
            "Social Media",
            "AI Assistant",
            "Data Analysis",
            "Automation",
            "Other",
        ]

    async def enhance_tool_data(self, tool_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance tool data using the prompt system."""
        enhanced_tool = tool_data.copy()

        # Check and enhance required fields
        required_fields = {
            "Category": self._enhance_category,
            "Description": self._enhance_description,
            "Price From": self._enhance_pricing,
        }

        for field, enhancer in required_fields.items():
            if not enhanced_tool.get(field) or enhanced_tool.get(field) == "":
                try:
                    enhanced_tool[field] = await enhancer(enhanced_tool)
                except Exception as e:
                    logger.error(f"Error enhancing {field}: {e}")
                    enhanced_tool[field] = self._get_default_value(field)

        # Enhance optional fields
        optional_fields = {
            "Features": self._enhance_features,
            "Tags (Keywords)": self._enhance_tags,
        }

        for field, enhancer in optional_fields.items():
            if not enhanced_tool.get(field) or enhanced_tool.get(field) == "":
                try:
                    enhanced_tool[field] = await enhancer(enhanced_tool)
                except Exception as e:
                    logger.error(f"Error enhancing {field}: {e}")
                    enhanced_tool[field] = self._get_default_value(field)

        return enhanced_tool

    async def _enhance_category(self, tool_data: Dict[str, Any]) -> str:
        """Enhance category using prompt system."""
        return await self.processor.process_prompt(
            PromptType.CATEGORY_GENERATION,
            valid_categories=", ".join(self.valid_categories),
            title=tool_data.get("Title", ""),
            description=tool_data.get("Description", ""),
            website=tool_data.get("Website", ""),
        )

    async def _enhance_description(self, tool_data: Dict[str, Any]) -> str:
        """Enhance description using prompt system."""
        return await self.processor.process_prompt(
            PromptType.DESCRIPTION_GENERATION,
            title=tool_data.get("Title", ""),
            website=tool_data.get("Website", ""),
            features=tool_data.get("Features", ""),
        )

    async def _enhance_features(self, tool_data: Dict[str, Any]) -> str:
        """Enhance features using prompt system."""
        return await self.processor.process_prompt(
            PromptType.FEATURES_GENERATION,
            title=tool_data.get("Title", ""),
            description=tool_data.get("Description", ""),
            category=tool_data.get("Category", ""),
        )

    async def _enhance_tags(self, tool_data: Dict[str, Any]) -> str:
        """Enhance tags using prompt system."""
        return await self.processor.process_prompt(
            PromptType.TAGS_GENERATION,
            title=tool_data.get("Title", ""),
            description=tool_data.get("Description", ""),
            category=tool_data.get("Category", ""),
        )

    async def _enhance_pricing(self, tool_data: Dict[str, Any]) -> str:
        """Enhance pricing using prompt system."""
        # For now, return default pricing info
        # This could be enhanced to analyze website content
        return "Contact for pricing"

    def _get_default_value(self, field: str) -> str:
        """Get default value for a field."""
        defaults = {
            "Category": "AI Tool",
            "Description": "AI-powered tool for various tasks",
            "Features": "AI-powered functionality",
            "Tags (Keywords)": "AI, automation",
            "Price From": "Contact for pricing",
        }
        return defaults.get(field, "")


# Global instances
prompt_system = PromptSystem()
