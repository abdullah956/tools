"""Scalable tool data validation service using centralized prompt system."""

import logging
from typing import Any, Dict, List, Optional

from ai_tool_recommender.ai_agents.core.llm import shared_llm
from ai_tool_recommender.ai_agents.core.prompt_system import (
    PromptProcessor,
    ToolDataEnhancer,
)

logger = logging.getLogger(__name__)


class ToolDataValidator:
    """Scalable validator for tool data using centralized prompt system."""

    REQUIRED_FIELDS = {
        "Title": "Tool name/title",
        "Description": "What the tool does",
        "Category": "Tool category/type",
        "Website": "Official website URL",
        "Price From": "Starting price",
    }

    OPTIONAL_FIELDS = {
        "Features": "Key features list",
        "Tags (Keywords)": "Relevant keywords",
        "Twitter": "Twitter profile URL",
        "Facebook": "Facebook page URL",
        "Linkedin": "LinkedIn company page URL",
        "Instagram": "Instagram profile URL",
        "Price To": "Maximum price",
    }

    def __init__(self):
        """Initialize the validator with prompt system."""
        self.prompt_processor = PromptProcessor(shared_llm)
        self.data_enhancer = ToolDataEnhancer(self.prompt_processor)

    async def validate_tool_data(
        self, tool: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Validate and enhance tool data using the scalable prompt system.

        Args:
            tool: Tool data dictionary

        Returns:
            Enhanced tool data with all required fields filled, or None if tool should be rejected
        """
        try:
            # First check if tool has valid title - reject garbage immediately
            title = tool.get("Title", "").strip()
            garbage_names = [
                "unknown tool",
                "untitled",
                "unknown",
                "",
                "tool",
                "ai tool",
                "software",
            ]

            if not title or title.lower() in garbage_names:
                logger.warning(f"❌ REJECTING tool with garbage title: '{title}'")
                return None

            # Check for minimum viable website
            website = tool.get("Website", "").strip()
            if not website or website in ["", "N/A", "Unknown", "None"]:
                logger.warning(f"❌ REJECTING tool '{title}' - no valid website")
                return None

            # Use the centralized data enhancer
            enhanced_tool = await self.data_enhancer.enhance_tool_data(tool)

            # Final validation - ensure enhanced tool still has valid data
            enhanced_title = enhanced_tool.get("Title", "").strip()
            if not enhanced_title or enhanced_title.lower() in garbage_names:
                logger.warning(
                    f"❌ REJECTING enhanced tool with invalid title: '{enhanced_title}'"
                )
                return None

            # Log validation results
            missing_fields = self._get_missing_fields(enhanced_tool)
            if missing_fields:
                logger.warning(
                    f"Tool {enhanced_tool.get('Title', 'Unknown')} still missing: {missing_fields}"
                )

            logger.info(f"✅ Validated tool: '{enhanced_title}'")
            return enhanced_tool

        except Exception as e:
            logger.error(f"Error validating tool data: {e}")
            # Try fallback, but it might return None if tool is garbage
            fallback_tool = self._create_fallback_tool(tool)
            if fallback_tool is None:
                logger.warning("❌ Fallback also rejected tool - returning None")
            return fallback_tool

    async def validate_tools_batch(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Validate a batch of tools using parallel processing for maximum speed."""
        import asyncio

        # Process tools in parallel for speed
        tasks = [self.validate_tool_data(tool) for tool in tools]
        validated_tools = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions and filter out None results
        final_tools = []
        for i, result in enumerate(validated_tools):
            if isinstance(result, Exception):
                logger.error(
                    f"Error validating tool {tools[i].get('Title', 'Unknown')}: {result}"
                )
                fallback_tool = self._create_fallback_tool(tools[i])
                if fallback_tool is not None:  # Only add if fallback is valid
                    final_tools.append(fallback_tool)
            elif result is not None:  # Only add non-None results
                final_tools.append(result)
            # Skip None results (rejected tools)

        return final_tools

    def get_validation_report(self, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a validation report for tools.

        Args:
            tools: List of tool data dictionaries

        Returns:
            Validation report with statistics
        """
        total_tools = len(tools)
        missing_fields = {}

        for field in self.REQUIRED_FIELDS.keys():
            missing_count = sum(
                1 for tool in tools if not tool.get(field) or tool.get(field) == ""
            )
            missing_fields[field] = missing_count

        optional_missing = {}
        for field in self.OPTIONAL_FIELDS.keys():
            missing_count = sum(
                1 for tool in tools if not tool.get(field) or tool.get(field) == ""
            )
            optional_missing[field] = missing_count

        return {
            "total_tools": total_tools,
            "required_fields_missing": missing_fields,
            "optional_fields_missing": optional_missing,
            "validation_status": (
                "passed"
                if all(count == 0 for count in missing_fields.values())
                else "failed"
            ),
            "recommendations": self._get_recommendations(
                missing_fields, optional_missing
            ),
            "system_info": {
                "prompt_system_version": "2.0",
                "enhancement_method": "centralized_prompt_system",
                "scalability": "high",
            },
        }

    def _get_missing_fields(self, tool: Dict[str, Any]) -> List[str]:
        """Get list of missing required fields."""
        missing = []
        for field in self.REQUIRED_FIELDS.keys():
            if not tool.get(field) or tool.get(field) == "":
                missing.append(field)
        return missing

    def _create_fallback_tool(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Create a tool with minimal required fields as fallback - but only if it has valid data."""
        title = tool.get("Title", "").strip()

        # REJECT tools with garbage names - don't create fallback for them
        garbage_names = [
            "unknown tool",
            "untitled",
            "unknown",
            "",
            "tool",
            "ai tool",
            "software",
        ]
        if not title or title.lower() in garbage_names:
            logger.warning(
                f"❌ REJECTING tool with garbage title: '{title}' - no fallback created"
            )
            return None  # Return None to indicate this tool should be rejected

        # Only create fallback if we have a legitimate tool name
        fallback_tool = tool.copy()
        fallback_tool["Title"] = title
        fallback_tool["Description"] = tool.get(
            "Description", f"{title} - AI-powered tool"
        )
        fallback_tool["Category"] = tool.get("Category", "AI Tool")
        fallback_tool["Website"] = tool.get("Website", "")
        fallback_tool["Price From"] = tool.get("Price From", "Contact for pricing")

        logger.info(f"✅ Created fallback tool for: '{title}'")
        return fallback_tool

    def _get_recommendations(
        self, required_missing: Dict[str, int], optional_missing: Dict[str, int]
    ) -> List[str]:
        """Generate recommendations based on missing fields."""
        recommendations = []

        for field, count in required_missing.items():
            if count > 0:
                recommendations.append(
                    f"Fix {count} tools missing required field: {field}"
                )

        for field, count in optional_missing.items():
            if count > 0:
                recommendations.append(
                    f"Consider enhancing {count} tools with optional field: {field}"
                )

        return recommendations


class ValidationMetrics:
    """Metrics and analytics for validation system."""

    def __init__(self):
        """Initialize validation metrics with empty statistics."""
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "field_enhancements": {},
            "prompt_usage": {},
        }

    def record_validation(
        self, success: bool, enhanced_fields: List[str], prompt_types_used: List[str]
    ):
        """Record validation metrics."""
        self.validation_stats["total_validations"] += 1

        if success:
            self.validation_stats["successful_validations"] += 1
        else:
            self.validation_stats["failed_validations"] += 1

        # Record field enhancements
        for field in enhanced_fields:
            self.validation_stats["field_enhancements"][field] = (
                self.validation_stats["field_enhancements"].get(field, 0) + 1
            )

        # Record prompt usage
        for prompt_type in prompt_types_used:
            self.validation_stats["prompt_usage"][prompt_type] = (
                self.validation_stats["prompt_usage"].get(prompt_type, 0) + 1
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        total = self.validation_stats["total_validations"]
        success_rate = (
            self.validation_stats["successful_validations"] / total * 100
            if total > 0
            else 0
        )

        return {
            **self.validation_stats,
            "success_rate": f"{success_rate:.2f}%",
            "system_efficiency": (
                "high"
                if success_rate > 90
                else "medium"
                if success_rate > 70
                else "low"
            ),
        }


# Global instances
tool_validator = ToolDataValidator()
validation_metrics = ValidationMetrics()
