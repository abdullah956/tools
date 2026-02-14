"""Tool Selector Agent - Intelligently selects the best tools from search results."""

import logging
from typing import Any, Dict, List

from ai_tool_recommender.agents.base_agent import BaseAgent
from ai_tool_recommender.ai_agents.core.llm import get_shared_llm
from ai_tool_recommender.ai_agents.core.validation import ToolDataFormatter

logger = logging.getLogger(__name__)


class ToolSelectorAgent(BaseAgent):
    """
    Agent that intelligently selects the best tools from parallel search results.

    This agent:
    - Analyzes batches of tools from each search query
    - Understands workflow context and tool relationships
    - Selects complementary tools that work well together
    - Ensures comprehensive coverage of workflow requirements
    - Provides reasoning for tool selections
    """

    def __init__(self):
        """Initialize the tool selector agent."""
        super().__init__()
        self.llm = get_shared_llm()

    def get_agent_name(self) -> str:
        """Get agent name."""
        return "tool_selector"

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
            message="Tool selector is called programmatically",
            suggestions=[],
        )

    async def select_optimal_tools(
        self,
        search_results: Dict[str, Dict[str, Any]],
        workflow_understanding: str,
        refined_query: str,
        max_tools: int = 8,
    ) -> Dict[str, Any]:
        """
        Select the optimal tools from parallel search results.

        Args:
            search_results: Results from ParallelToolSearchService
            workflow_understanding: LLM's understanding of the workflow
            refined_query: Original refined query for context
            max_tools: Maximum number of tools to select

        Returns:
            Dictionary containing:
            - selected_tools: List of selected tools with reasoning
            - selection_reasoning: Overall reasoning for selections
            - workflow_coverage: Analysis of workflow step coverage
            - tool_relationships: How tools work together
            - confidence_score: Confidence in the selection
        """
        try:
            logger.info(
                f"🎯 Selecting optimal tools from {len(search_results)} search categories"
            )

            # Analyze each category and pre-select best tools
            category_selections = await self._analyze_categories_parallel(
                search_results, workflow_understanding, refined_query
            )

            # Perform final selection considering tool relationships
            final_selection = await self._perform_final_selection(
                category_selections, workflow_understanding, refined_query, max_tools
            )

            # Generate comprehensive analysis
            analysis = await self._generate_selection_analysis(
                final_selection, search_results, workflow_understanding
            )

            return {
                "selected_tools": final_selection["tools"],
                "selection_reasoning": final_selection["reasoning"],
                "workflow_coverage": analysis["workflow_coverage"],
                "tool_relationships": analysis["tool_relationships"],
                "confidence_score": analysis["confidence_score"],
                "category_breakdown": category_selections,
                "total_tools_selected": len(final_selection["tools"]),
                "status": "success",
            }

        except Exception as e:
            logger.error(f"❌ Error in tool selection: {e}", exc_info=True)
            return {
                "selected_tools": [],
                "selection_reasoning": f"Error in tool selection: {str(e)}",
                "workflow_coverage": {},
                "tool_relationships": {},
                "confidence_score": 0,
                "status": "error",
                "error": str(e),
            }

    async def _analyze_categories_parallel(
        self,
        search_results: Dict[str, Dict[str, Any]],
        workflow_understanding: str,
        refined_query: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze each tool category and pre-select best tools."""
        import asyncio

        logger.info(f"🔍 Analyzing {len(search_results)} tool categories in parallel")

        # Create tasks for parallel category analysis
        tasks = []
        for category_id, category_data in search_results.items():
            task = asyncio.create_task(
                self._analyze_single_category(
                    category_id, category_data, workflow_understanding, refined_query
                ),
                name=f"analyze_category_{category_id}",
            )
            tasks.append((category_id, task))

        # Wait for all analyses to complete
        category_selections = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in tasks], return_exceptions=True
        )

        # Process results
        for (category_id, _), result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"❌ Category analysis failed for {category_id}: {result}")
                category_selections[category_id] = {
                    "selected_tools": [],
                    "reasoning": f"Analysis failed: {str(result)}",
                    "confidence": 0,
                }
            else:
                category_selections[category_id] = result
                logger.info(
                    f"✅ {category_id}: Selected {len(result['selected_tools'])} tools "
                    f"(confidence: {result['confidence']})"
                )

        return category_selections

    async def _analyze_single_category(
        self,
        category_id: str,
        category_data: Dict[str, Any],
        workflow_understanding: str,
        refined_query: str,
    ) -> Dict[str, Any]:
        """Analyze a single category and select best tools."""
        try:
            tools = category_data.get("tools", [])
            query_info = category_data.get("query_info", {})

            if not tools:
                return {
                    "selected_tools": [],
                    "reasoning": f"No tools found for {category_id}",
                    "confidence": 0,
                }

            logger.info(f"🔍 Analyzing {len(tools)} tools in category: {category_id}")

            # Prepare tools data for LLM analysis
            tools_data = ToolDataFormatter.prepare_tools_data_for_prompt(tools)

            analysis_prompt = f"""
You are an AI Tool Selection Expert. Analyze these tools for a specific workflow category and select the BEST ones.

WORKFLOW CONTEXT: {workflow_understanding}
REFINED QUERY: {refined_query}
CATEGORY: {category_id}
CATEGORY PURPOSE: {query_info.get('purpose', 'Not specified')}
CATEGORY PRIORITY: {query_info.get('priority', 'medium')}

TOOLS TO ANALYZE:
{tools_data}

Your task:
1. UNDERSTAND how this category fits into the overall workflow
2. EVALUATE each tool's relevance, quality, and capabilities
3. SELECT the 2-3 BEST tools that would work optimally for this workflow step
4. CONSIDER tool popularity, features, integration capabilities, and user reviews
5. PRIORITIZE tools that complement the overall workflow

SELECTION CRITERIA:
- Direct relevance to the workflow step
- Quality and reliability indicators
- Feature completeness
- Integration capabilities with other tools
- User adoption and reviews (if available)
- Pricing appropriateness

Return ONLY valid JSON in this exact format:
{{
    "selected_tools": [
        {{
            "tool_index": 1,
            "tool_name": "Tool Name",
            "selection_reason": "Specific reason why this tool is optimal for the workflow",
            "workflow_fit": "How this tool fits into the overall workflow",
            "key_features": ["Feature 1", "Feature 2", "Feature 3"],
            "confidence": 0.95
        }}
    ],
    "reasoning": "Overall reasoning for selections in this category",
    "category_importance": "high/medium/low",
    "workflow_step": "Which step of the workflow this category addresses",
    "confidence": 0.85
}}

CRITICAL: Only select tools that are TRULY relevant and would add value to the workflow.
Better to select fewer high-quality tools than many mediocre ones.
Tool indices should match the numbering in the tools list (1-based).
"""

            response = await self.llm.generate_response(analysis_prompt)
            result = await self.llm.parse_json_response(response)

            if not result or "selected_tools" not in result:
                logger.error(f"❌ Invalid LLM response for category {category_id}")
                return self._create_fallback_category_selection(tools, category_id)

            # Validate and enhance selected tools with full data
            selected_tools = []
            for selection in result.get("selected_tools", []):
                tool_index = selection.get("tool_index", 1) - 1  # Convert to 0-based
                if 0 <= tool_index < len(tools):
                    tool_data = tools[tool_index].copy()
                    tool_data["selection_reason"] = selection.get(
                        "selection_reason", ""
                    )
                    tool_data["workflow_fit"] = selection.get("workflow_fit", "")
                    tool_data["key_features"] = selection.get("key_features", [])
                    tool_data["selection_confidence"] = selection.get("confidence", 0.8)
                    selected_tools.append(tool_data)

            result["selected_tools"] = selected_tools
            logger.info(
                f"✅ Category {category_id}: Selected {len(selected_tools)} tools"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Error analyzing category {category_id}: {e}")
            return self._create_fallback_category_selection(tools, category_id)

    def _create_fallback_category_selection(
        self, tools: List[Dict[str, Any]], category_id: str
    ) -> Dict[str, Any]:
        """Create fallback selection if LLM analysis fails."""
        # Select top 2 tools by similarity score
        sorted_tools = sorted(
            tools,
            key=lambda x: x.get("Similarity Score", x.get("Relevance Score", 0)),
            reverse=True,
        )

        selected_tools = []
        for i, tool in enumerate(sorted_tools[:2]):
            tool_data = tool.copy()
            tool_data[
                "selection_reason"
            ] = f"High similarity score ({tool.get('Similarity Score', 0):.2f})"
            tool_data["workflow_fit"] = f"Relevant to {category_id} requirements"
            tool_data["selection_confidence"] = 0.6
            selected_tools.append(tool_data)

        return {
            "selected_tools": selected_tools,
            "reasoning": f"Fallback selection for {category_id} based on similarity scores",
            "category_importance": "medium",
            "confidence": 0.6,
        }

    async def _perform_final_selection(
        self,
        category_selections: Dict[str, Dict[str, Any]],
        workflow_understanding: str,
        refined_query: str,
        max_tools: int,
    ) -> Dict[str, Any]:
        """Perform final tool selection considering relationships and workflow fit."""
        try:
            # Collect all pre-selected tools
            all_selected_tools = []
            for category_id, selection in category_selections.items():
                for tool in selection.get("selected_tools", []):
                    tool["source_category"] = category_id
                    all_selected_tools.append(tool)

            if len(all_selected_tools) <= max_tools:
                # All tools fit within limit
                return {
                    "tools": all_selected_tools,
                    "reasoning": f"Selected all {len(all_selected_tools)} pre-selected tools as they fit within the {max_tools} tool limit",
                }

            logger.info(
                f"🎯 Final selection: {len(all_selected_tools)} tools → {max_tools} tools"
            )

            # Need to reduce tools - use LLM for intelligent selection
            tools_data = ToolDataFormatter.prepare_tools_data_for_prompt(
                all_selected_tools
            )

            # Build the prompt using string concatenation to avoid bandit SQL injection warning
            final_selection_prompt = (
                "You are a Workflow Optimization Expert. Select the BEST "  # nosec B608
                + str(max_tools)  # nosec B608
                + " tools from these pre-selected candidates to create an optimal workflow.\n\n"  # nosec B608
                "WORKFLOW CONTEXT: " + workflow_understanding + "\n"  # nosec B608
                "REFINED QUERY: " + refined_query + "\n\n"  # nosec B608
                "PRE-SELECTED TOOLS:\n" + tools_data + "\n\n"  # nosec B608
                "Your task:\n"  # nosec B608
                "1. SELECT exactly "  # nosec B608
                + str(max_tools)  # nosec B608
                + " tools that work BEST together\n"  # nosec B608
                "2. ENSURE comprehensive workflow coverage\n"
                "3. PRIORITIZE tools with high selection confidence\n"
                "4. AVOID redundant functionality\n"
                "5. CREATE a logical workflow sequence\n\n"
                "SELECTION CRITERIA:\n"
                "- Workflow coverage and completeness\n"
                "- Tool complementarity and integration potential\n"
                "- Selection confidence from category analysis\n"
                "- Avoiding functional overlap\n"
                "- Creating logical workflow steps\n\n"
                "Return ONLY valid JSON in this exact format:\n"
                "{\n"
                '    "selected_tool_indices": [1, 3, 5, 7, 9, 12, 15, 18],\n'
                '    "reasoning": "Detailed explanation of why these specific tools were selected and how they work together",\n'
                '    "workflow_sequence": [\n'
                "        {\n"
                '            "step": 1,\n'
                '            "tool_index": 1,\n'
                '            "tool_name": "Tool Name",\n'
                '            "purpose": "What this tool does in the workflow"\n'
                "        }\n"
                "    ],\n"
                '    "coverage_analysis": "Analysis of how well the selected tools cover the workflow requirements"\n'
                "}\n\n"
                "CRITICAL: Select exactly "  # nosec B608
                + str(max_tools)  # nosec B608
                + " tools. Tool indices should match the numbering in the tools list (1-based)."  # nosec B608
            )  # nosec B608

            response = await self.llm.generate_response(final_selection_prompt)
            result = await self.llm.parse_json_response(response)

            if not result or "selected_tool_indices" not in result:
                logger.error("❌ Invalid LLM response for final selection")
                return self._create_fallback_final_selection(
                    all_selected_tools, max_tools
                )

            # Extract selected tools and ensure strict uniqueness
            selected_indices = result.get("selected_tool_indices", [])
            final_tools = []
            seen_identifiers = set()

            for index in selected_indices:
                tool_index = index - 1  # Convert to 0-based
                if 0 <= tool_index < len(all_selected_tools):
                    tool = all_selected_tools[tool_index]

                    # Create comprehensive identifiers for strict deduplication
                    identifiers = self._create_tool_identifiers_comprehensive(tool)

                    # Check if any identifier already exists
                    is_duplicate = False
                    matched_identifier = None
                    for identifier in identifiers:
                        if identifier in seen_identifiers:
                            is_duplicate = True
                            matched_identifier = identifier
                            break

                    # Only add if not duplicate
                    if not is_duplicate:
                        # Add all identifiers to seen set
                        seen_identifiers.update(identifiers)
                        final_tools.append(tool)
                    else:
                        logger.info(
                            f"🔄 Skipping duplicate tool: {tool.get('Title', 'Unknown')} (matched: {matched_identifier})"
                        )

            # Ensure we have the right number of tools
            if len(final_tools) != max_tools:
                logger.warning(f"⚠️ Expected {max_tools} tools, got {len(final_tools)}")
                # Adjust if needed
                if len(final_tools) < max_tools:
                    # Add more tools if we have fewer
                    remaining_tools = [
                        tool
                        for i, tool in enumerate(all_selected_tools)
                        if (i + 1) not in selected_indices
                    ]
                    final_tools.extend(remaining_tools[: max_tools - len(final_tools)])
                elif len(final_tools) > max_tools:
                    # Trim if we have too many
                    final_tools = final_tools[:max_tools]

            return {
                "tools": final_tools,
                "reasoning": result.get(
                    "reasoning", "Final selection based on workflow optimization"
                ),
                "workflow_sequence": result.get("workflow_sequence", []),
                "coverage_analysis": result.get("coverage_analysis", ""),
            }

        except Exception as e:
            logger.error(f"❌ Error in final selection: {e}")
            return self._create_fallback_final_selection(all_selected_tools, max_tools)

    def _create_fallback_final_selection(
        self, all_tools: List[Dict[str, Any]], max_tools: int
    ) -> Dict[str, Any]:
        """Create fallback final selection if LLM fails."""
        # Sort by selection confidence and similarity score
        sorted_tools = sorted(
            all_tools,
            key=lambda x: (
                x.get("selection_confidence", 0),
                x.get("Similarity Score", x.get("Relevance Score", 0)),
            ),
            reverse=True,
        )

        selected_tools = sorted_tools[:max_tools]

        return {
            "tools": selected_tools,
            "reasoning": f"Fallback selection of top {len(selected_tools)} tools by confidence and relevance scores",
        }

    async def _generate_selection_analysis(
        self,
        final_selection: Dict[str, Any],
        search_results: Dict[str, Dict[str, Any]],
        workflow_understanding: str,
    ) -> Dict[str, Any]:
        """Generate comprehensive analysis of the tool selection."""
        try:
            selected_tools = final_selection.get("tools", [])

            # Calculate workflow coverage
            categories_covered = {
                tool.get("source_category", "unknown") for tool in selected_tools
            }
            total_categories = len(search_results)
            coverage_percentage = (
                (len(categories_covered) / total_categories * 100)
                if total_categories > 0
                else 0
            )

            # Calculate confidence score
            confidence_scores = [
                tool.get("selection_confidence", 0.5) for tool in selected_tools
            ]
            avg_confidence = (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores
                else 0
            )

            # Analyze tool relationships
            tool_relationships = self._analyze_tool_relationships(selected_tools)

            return {
                "workflow_coverage": {
                    "categories_covered": list(categories_covered),
                    "total_categories": total_categories,
                    "coverage_percentage": round(coverage_percentage, 1),
                    "coverage_quality": "excellent"
                    if coverage_percentage >= 80
                    else "good"
                    if coverage_percentage >= 60
                    else "needs_improvement",
                },
                "tool_relationships": tool_relationships,
                "confidence_score": round(avg_confidence, 2),
                "selection_quality": "high"
                if avg_confidence >= 0.8
                else "medium"
                if avg_confidence >= 0.6
                else "low",
            }

        except Exception as e:
            logger.error(f"❌ Error generating selection analysis: {e}")
            return {
                "workflow_coverage": {},
                "tool_relationships": {},
                "confidence_score": 0.5,
            }

    def _analyze_tool_relationships(
        self, selected_tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze relationships between selected tools."""
        try:
            # Group tools by category
            category_groups = {}
            for tool in selected_tools:
                category = tool.get("source_category", "unknown")
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append(tool.get("Title", "Unknown"))

            # Identify potential integrations (simplified)
            integration_potential = []
            tool_names = [tool.get("Title", "").lower() for tool in selected_tools]

            # Common integration patterns
            if any("zapier" in name for name in tool_names):
                integration_potential.append(
                    "Zapier can connect multiple tools in the workflow"
                )

            if any("slack" in name for name in tool_names) and len(selected_tools) > 1:
                integration_potential.append(
                    "Slack can serve as central communication hub"
                )

            return {
                "category_distribution": category_groups,
                "integration_opportunities": integration_potential,
                "workflow_completeness": len(category_groups)
                >= 3,  # At least 3 different categories
                "tool_diversity": len(
                    {tool.get("Category", "") for tool in selected_tools}
                ),
            }

        except Exception as e:
            logger.error(f"❌ Error analyzing tool relationships: {e}")
            return {}

    def _create_tool_identifiers_comprehensive(self, tool: Dict[str, Any]) -> List[str]:
        """Create comprehensive identifiers for strict tool deduplication."""
        identifiers = []

        try:
            title = tool.get("Title", "").strip()
            website = tool.get("Website", "").strip()

            # 1. Website-based identifiers (most reliable)
            if website:
                clean_website = (
                    website.lower()
                    .replace("https://", "")
                    .replace("http://", "")
                    .replace("www.", "")
                    .rstrip("/")
                )
                if clean_website and len(clean_website) > 5:
                    identifiers.append(f"website:{clean_website}")

                    # Domain-only identifier
                    try:
                        from urllib.parse import urlparse

                        domain = urlparse(website).netloc.lower()
                        if domain:
                            identifiers.append(f"domain:{domain}")
                    except Exception:
                        pass

            # 2. Title-based identifiers
            if title and len(title) > 3:
                # Exact title
                identifiers.append(f"title:{title.lower()}")

                # Normalized title (handle AI variations)
                normalized = (
                    title.lower()
                    .replace(" ai", "")
                    .replace("ai ", "")
                    .replace(".", "")
                    .replace("-", "")
                    .replace("_", "")
                    .replace(" ", "")
                )
                if len(normalized) > 3:
                    identifiers.append(f"normalized:{normalized}")

                # Core name (remove common suffixes/prefixes)
                core_name = (
                    title.lower()
                    .replace(" app", "")
                    .replace(" tool", "")
                    .replace(" platform", "")
                    .replace(" software", "")
                    .replace(" suite", "")
                    .replace("the ", "")
                    .strip()
                )
                if len(core_name) > 3 and core_name != title.lower():
                    identifiers.append(f"core:{core_name}")

            # 3. Fallback identifier
            if not identifiers:
                identifiers.append(f"fallback:{title.lower()}" if title else "unknown")

            return identifiers

        except Exception as e:
            logger.error(f"Error creating comprehensive tool identifiers: {e}")
            return [f"error:{tool.get('Title', 'unknown')}"]
