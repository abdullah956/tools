"""Query Decomposer Agent - Analyzes refined queries and generates targeted tool search queries."""

import logging
from typing import Any, Dict

from ai_tool_recommender.agents.base_agent import BaseAgent
from ai_tool_recommender.ai_agents.core.llm import get_shared_llm

logger = logging.getLogger(__name__)


class QueryDecomposerAgent(BaseAgent):
    """
    Agent that analyzes refined queries and decomposes them into targeted tool search queries.

    This agent:
    - Understands the complete workflow context from refined queries
    - Identifies specific tool categories and functionalities needed
    - Generates multiple targeted search queries for parallel processing
    - Ensures comprehensive coverage of the user's workflow requirements
    """

    def __init__(self):
        """Initialize the query decomposer agent."""
        super().__init__()
        self.llm = get_shared_llm()

    def get_agent_name(self) -> str:
        """Get agent name."""
        return "query_decomposer"

    async def can_handle(
        self, user_message: str, conversation, current_state: str, **kwargs
    ) -> bool:
        """This agent is called programmatically, not through user messages."""
        return False

    async def process_message(
        self,
        user_message: str,
        conversation,
        workflow_id: str,
        request_user,
        **kwargs,
    ) -> Dict[str, Any]:
        """This agent is called programmatically."""
        return self.format_response(
            message="Query decomposer is called programmatically",
            suggestions=[],
        )

    async def decompose_refined_query(
        self, refined_query: str, original_query: str = ""
    ) -> Dict[str, Any]:
        """
        Decompose a refined query into targeted tool search queries.

        Args:
            refined_query: The comprehensive refined query from questionnaire
            original_query: Optional original user query for context

        Returns:
            Dictionary containing:
            - workflow_understanding: LLM's understanding of the workflow
            - tool_categories: List of tool categories needed
            - search_queries: List of targeted search queries for parallel processing
            - expected_tools_count: Expected number of tools needed
            - workflow_steps: Identified workflow steps
        """
        try:
            logger.info(f"🧠 Decomposing refined query: {refined_query[:100]}...")

            decomposition_prompt = f"""
You are a Workflow Analysis Expert. Your task is to analyze a refined user query and decompose it into targeted tool search queries for building an AI workflow.

REFINED QUERY: "{refined_query}"
{f'ORIGINAL QUERY: "{original_query}"' if original_query else ''}

Your analysis should:
1. UNDERSTAND THE COMPLETE WORKFLOW and the specific problem the user is solving.
2. EXTRACT MENTIONED TOOLS: Look for tools mentioned in "Current Situation" or "Specific Tools Mentioned" (e.g., Salesforce, Excel, QuickBooks) and create targeted search queries for them or their integrations.
3. IDENTIFY FUNCTIONAL PROBLEMS: Extract specific functional requirements (e.g., "automated data reconciliation", "revenue forecasting formulas") and create descriptive search queries for tools that solve these specific tasks.
4. GENERATE TARGETED SEARCH QUERIES: Create unique queries for each tool category and specific functionality needed.

CRITICAL REQUIREMENTS:
- Each search query should target a SPECIFIC tool name (if mentioned) or a SPECIFIC functionality.
- Queries should be optimized for vector similarity search (descriptive, not just keywords).
- NEVER use the entire Refined Query as a search string.
- Generate 4-7 search queries covering both the tools mentioned and the problems described.
- Each query should be 1-2 sentences.

Return ONLY valid JSON in this exact format:
{{
    "workflow_understanding": "Clear description of what workflow the user wants to build and the problem they're solving",
    "workflow_steps": [
        "Step 1: Description of first workflow step",
        "Step 2: Description of second workflow step"
    ],
    "tool_categories": [
        {{
            "category": "Email Marketing Automation",
            "purpose": "Send automated email campaigns to leads",
            "priority": "high",
            "step_number": 1
        }},
        {{
            "category": "CRM Integration",
            "purpose": "Manage and track customer relationships",
            "priority": "medium",
            "step_number": 2
        }}
    ],
    "search_queries": [
        {{
            "query": "Email marketing automation platform with campaign management and lead nurturing capabilities",
            "target_category": "Email Marketing Automation",
            "expected_results": 5,
            "priority": "high",
            "workflow_step": 1
        }},
        {{
            "query": "Customer relationship management CRM software with contact management and sales pipeline tracking",
            "target_category": "CRM Integration",
            "expected_results": 4,
            "priority": "medium",
            "workflow_step": 2
        }}
    ],
    "expected_tools_count": 8,
    "workflow_complexity": "medium",
    "success_criteria": "What defines success for this workflow"
}}

EXAMPLES OF GOOD SEARCH QUERIES:
- "Project management software with task tracking team collaboration and deadline management"
- "Social media scheduling platform with multi-platform posting analytics and content calendar"
- "Customer support chatbot with natural language processing and ticket routing capabilities"

AVOID:
- Generic queries like "email tool" or "CRM"
- Single-word searches
- Overly broad queries that would match too many irrelevant tools

Generate queries that will find SPECIFIC, RELEVANT tools for the user's exact workflow needs.
"""

            response = await self.llm.generate_response(decomposition_prompt)
            result = await self.llm.parse_json_response(response)

            if not result or "search_queries" not in result:
                logger.error("❌ LLM didn't return valid decomposition format")
                return self._create_fallback_decomposition(refined_query)

            # Validate and enhance the result
            search_queries = result.get("search_queries", [])
            if not search_queries:
                logger.warning("⚠️ No search queries generated, creating fallback")
                return self._create_fallback_decomposition(refined_query)

            # Deduplicate search queries
            unique_queries = []
            seen_queries = set()

            for query_obj in search_queries:
                query_text = query_obj.get("query", "").lower().strip()
                if query_text and query_text not in seen_queries:
                    seen_queries.add(query_text)
                    unique_queries.append(query_obj)
                else:
                    logger.info(f"🔄 Skipping duplicate query: {query_text[:50]}...")

            search_queries = unique_queries

            # Ensure we have reasonable number of queries (3-7)
            if len(search_queries) > 7:
                search_queries = search_queries[:7]
                logger.info(f"🔧 Trimmed to {len(search_queries)} search queries")
            elif len(search_queries) < 3:
                logger.warning(
                    f"⚠️ Only {len(search_queries)} queries generated, may need more coverage"
                )

            result["search_queries"] = search_queries
            result["total_queries"] = len(search_queries)

            logger.info(f"✅ Generated {len(search_queries)} targeted search queries")
            logger.info(
                f"📊 Workflow complexity: {result.get('workflow_complexity', 'unknown')}"
            )
            logger.info(
                f"🎯 Expected tools: {result.get('expected_tools_count', 'unknown')}"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Error decomposing query: {e}", exc_info=True)
            return self._create_fallback_decomposition(refined_query)

    def _create_fallback_decomposition(self, refined_query: str) -> Dict[str, Any]:
        """Create a fallback decomposition if LLM fails."""
        logger.info("🔧 Creating fallback decomposition")

        # Simple keyword-based fallback
        query_lower = refined_query.lower()

        # Identify common workflow patterns
        search_queries = []

        if any(word in query_lower for word in ["email", "marketing", "campaign"]):
            search_queries.append(
                {
                    "query": "Email marketing automation platform with campaign management capabilities",
                    "target_category": "Email Marketing",
                    "expected_results": 5,
                    "priority": "high",
                    "workflow_step": 1,
                }
            )

        if any(word in query_lower for word in ["crm", "customer", "sales", "lead"]):
            search_queries.append(
                {
                    "query": "Customer relationship management CRM software with contact management",
                    "target_category": "CRM",
                    "expected_results": 4,
                    "priority": "high",
                    "workflow_step": 2,
                }
            )

        if any(word in query_lower for word in ["social", "media", "post", "content"]):
            search_queries.append(
                {
                    "query": "Social media management platform with scheduling and analytics",
                    "target_category": "Social Media",
                    "expected_results": 4,
                    "priority": "medium",
                    "workflow_step": 3,
                }
            )

        if any(word in query_lower for word in ["project", "task", "manage", "team"]):
            search_queries.append(
                {
                    "query": "Project management software with task tracking and team collaboration",
                    "target_category": "Project Management",
                    "expected_results": 4,
                    "priority": "medium",
                    "workflow_step": 4,
                }
            )

        # If no specific patterns found, create generic search
        if not search_queries:
            search_queries.append(
                {
                    "query": f"AI automation tools and software platforms for {refined_query[:100]}",
                    "target_category": "General AI Tools",
                    "expected_results": 5,
                    "priority": "high",
                    "workflow_step": 1,
                }
            )

        return {
            "workflow_understanding": f"Workflow based on: {refined_query[:200]}",
            "workflow_steps": [
                "Analyze requirements",
                "Implement solution",
                "Monitor results",
            ],
            "tool_categories": [
                {
                    "category": query["target_category"],
                    "purpose": "Support workflow requirements",
                    "priority": query["priority"],
                    "step_number": query["workflow_step"],
                }
                for query in search_queries
            ],
            "search_queries": search_queries,
            "expected_tools_count": sum(q["expected_results"] for q in search_queries),
            "workflow_complexity": "medium",
            "success_criteria": "Successfully implement the requested workflow",
            "total_queries": len(search_queries),
            "fallback_used": True,
        }

    async def validate_decomposition(self, decomposition: Dict[str, Any]) -> bool:
        """
        Validate that the decomposition is comprehensive and actionable.

        Args:
            decomposition: The decomposition result to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            required_fields = [
                "workflow_understanding",
                "search_queries",
                "expected_tools_count",
            ]

            # Check required fields
            for field in required_fields:
                if field not in decomposition:
                    logger.error(f"❌ Missing required field: {field}")
                    return False

            # Validate search queries
            search_queries = decomposition.get("search_queries", [])
            if not search_queries or len(search_queries) < 1:
                logger.error("❌ No search queries found")
                return False

            # Validate each search query
            for i, query in enumerate(search_queries):
                required_query_fields = ["query", "target_category", "expected_results"]
                for field in required_query_fields:
                    if field not in query:
                        logger.error(f"❌ Search query {i+1} missing field: {field}")
                        return False

                # Check query length (should be descriptive)
                if len(query["query"]) < 20:
                    logger.warning(
                        f"⚠️ Search query {i+1} might be too short: {query['query']}"
                    )

            logger.info("✅ Decomposition validation passed")
            return True

        except Exception as e:
            logger.error(f"❌ Error validating decomposition: {e}")
            return False
