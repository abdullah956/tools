"""Service for generating workflow implementation guides."""

import json
import logging
import time
from typing import Any, Dict, Optional

from asgiref.sync import sync_to_async

from ..ai_agents.core.llm import get_shared_llm
from ..models import WorkflowImplementationGuide

logger = logging.getLogger(__name__)


class WorkflowImplementationService:
    """Service to generate comprehensive implementation guides for workflows."""

    def __init__(self):
        """Initialize the service."""
        # Use shared LLM to avoid executor shutdown issues
        self.llm = get_shared_llm()

    async def generate_or_update_implementation_guide(
        self, workflow_id: str, user, workflow_data: Dict[str, Any], workflow_query: str
    ) -> Dict[str, Any]:
        """
        Generate or update implementation guide for a workflow.

        Args:
            workflow_id: UUID of the workflow
            user: User object
            workflow_data: Complete workflow data with nodes and edges
            workflow_query: Original user query for the workflow

        Returns:
            Dict containing the implementation guide and metadata
        """
        start_time = time.time()

        try:
            # Check if implementation guide already exists
            existing_guide = await self._get_existing_guide(workflow_id, user)

            # Generate the implementation guide content
            implementation_content = await self._generate_implementation_content(
                workflow_data, workflow_query
            )

            generation_time_ms = (time.time() - start_time) * 1000
            tools_count = (
                len(workflow_data.get("nodes", [])) - 1
            )  # Exclude trigger node

            if existing_guide:
                # Update existing guide
                existing_guide.implementation_guide = implementation_content
                existing_guide.workflow_snapshot = workflow_data
                existing_guide.tools_count = tools_count
                existing_guide.generation_time_ms = generation_time_ms
                existing_guide.status = "completed"
                existing_guide.error_message = ""

                await sync_to_async(existing_guide.save)()

                logger.info(
                    f"✅ Updated implementation guide for workflow {workflow_id}"
                )

                return {
                    "status": "updated",
                    "implementation_guide": implementation_content,
                    "guide_id": str(existing_guide.id),
                    "tools_count": tools_count,
                    "generation_time_ms": generation_time_ms,
                }
            else:
                # Create new guide
                new_guide = await sync_to_async(
                    WorkflowImplementationGuide.objects.create
                )(
                    workflow_id=workflow_id,
                    user=user,
                    implementation_guide=implementation_content,
                    workflow_snapshot=workflow_data,
                    tools_count=tools_count,
                    generation_time_ms=generation_time_ms,
                    status="completed",
                )

                logger.info(
                    f"✅ Created new implementation guide for workflow {workflow_id}"
                )

                return {
                    "status": "created",
                    "implementation_guide": implementation_content,
                    "guide_id": str(new_guide.id),
                    "tools_count": tools_count,
                    "generation_time_ms": generation_time_ms,
                }

        except Exception as e:
            logger.error(f"❌ Error generating implementation guide: {e}")

            # Update or create guide with error status
            try:
                existing_guide = await self._get_existing_guide(workflow_id, user)
                if existing_guide:
                    existing_guide.status = "error"
                    existing_guide.error_message = str(e)
                    await sync_to_async(existing_guide.save)()
                else:
                    await sync_to_async(WorkflowImplementationGuide.objects.create)(
                        workflow_id=workflow_id,
                        user=user,
                        implementation_guide="",
                        workflow_snapshot=workflow_data,
                        tools_count=0,
                        status="error",
                        error_message=str(e),
                    )
            except Exception as save_error:
                logger.error(f"❌ Error saving error state: {save_error}")

            return {
                "status": "error",
                "error": str(e),
                "implementation_guide": None,
            }

    async def _get_existing_guide(
        self, workflow_id: str, user
    ) -> Optional[WorkflowImplementationGuide]:
        """Get existing implementation guide for workflow and user."""
        try:
            return await sync_to_async(WorkflowImplementationGuide.objects.get)(
                workflow_id=workflow_id, user=user
            )
        except WorkflowImplementationGuide.DoesNotExist:
            return None

    async def _generate_implementation_content(
        self, workflow_data: Dict[str, Any], workflow_query: str
    ) -> str:
        """Generate the implementation guide content using LLM."""
        prompt_template = """
You are an expert automation consultant who creates detailed, step-by-step implementation guides for AI and automation workflows.

The user has created a workflow to solve this challenge: "{query}"

Here is the complete workflow structure with all tools and connections:
{workflow_json}

Your task is to create a comprehensive, professional implementation guide that will help the user successfully implement this workflow from start to finish.

## Requirements:

### 1. Structure your guide with these sections:
- **# Workflow Implementation Guide**
- **## Overview**
- **## Prerequisites**
- **## Step-by-Step Implementation**
- **## Tool Configuration Details**
- **## Integration & Connections**
- **## Testing & Validation**
- **## Troubleshooting**
- **## Next Steps**

### 2. For each tool in the workflow:
- Explain what the tool does in simple terms
- Provide detailed setup instructions
- Include specific configuration steps
- Explain how to connect it to other tools
- Mention any API keys, accounts, or subscriptions needed
- Include screenshots or UI guidance where helpful

### 3. Implementation Details:
- Break down complex steps into smaller, actionable tasks
- Provide exact settings and configurations
- Include code snippets or configuration examples where applicable
- Explain the data flow between tools
- Address common setup challenges

### 4. Professional Quality:
- Use clear, professional language
- Include estimated time for each major step
- Provide troubleshooting tips for common issues
- Suggest best practices and optimization tips
- Include links to official documentation where relevant

### 5. Formatting:
- Use proper Markdown formatting
- Include numbered lists for sequential steps
- Use bullet points for options or features
- Use code blocks for technical configurations
- Use tables for comparison or settings

### 6. Practical Focus:
- Focus on actionable steps the user can take immediately
- Prioritize the most important configurations first
- Explain the "why" behind each step
- Provide alternative approaches when possible

Generate a comprehensive implementation guide that would allow someone to successfully implement this entire workflow, even if they're not highly technical.

Begin your implementation guide:
"""

        # Format the workflow data for the prompt
        formatted_workflow = json.dumps(workflow_data, indent=2)

        prompt = prompt_template.format(
            query=workflow_query, workflow_json=formatted_workflow
        )

        # Generate the implementation guide using shared LLM service
        # Handle executor shutdown errors gracefully
        try:
            implementation_content = await self.llm.generate_response(prompt)
            return implementation_content
        except RuntimeError as e:
            if (
                "cannot schedule new futures after shutdown" in str(e)
                or "shutdown" in str(e).lower()
            ):
                logger.warning(
                    "Executor shutdown detected, retrying with fresh LLM instance"
                )
                # Retry with a fresh LLM instance
                try:
                    from ..ai_agents.core.llm import SharedLLMService

                    fresh_llm = SharedLLMService()
                    implementation_content = await fresh_llm.generate_response(prompt)
                    return implementation_content
                except Exception as retry_error:
                    logger.error(f"Retry also failed: {retry_error}", exc_info=True)
                    raise RuntimeError(
                        "Failed to generate implementation guide due to executor shutdown. "
                        "Please try again in a moment."
                    ) from retry_error
            else:
                raise

    async def get_implementation_guide(
        self, workflow_id: str, user
    ) -> Optional[Dict[str, Any]]:
        """Get existing implementation guide for a workflow."""
        try:
            guide = await sync_to_async(WorkflowImplementationGuide.objects.get)(
                workflow_id=workflow_id, user=user
            )

            return {
                "id": str(guide.id),
                "workflow_id": str(guide.workflow_id),
                "implementation_guide": guide.implementation_guide,
                "status": guide.status,
                "tools_count": guide.tools_count,
                "generation_time_ms": guide.generation_time_ms,
                "created_at": guide.created_at.isoformat(),
                "updated_at": guide.updated_at.isoformat(),
                "error_message": guide.error_message,
            }
        except WorkflowImplementationGuide.DoesNotExist:
            return None
