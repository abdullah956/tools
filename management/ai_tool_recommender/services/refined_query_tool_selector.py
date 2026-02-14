"""Refined Query-Based Tool Selector - Intelligently selects optimal tools based on comprehensive analysis."""

import logging
from typing import Any, Dict, List

from ai_tool_recommender.ai_agents.core.llm import get_shared_llm

logger = logging.getLogger(__name__)


class RefinedQueryToolSelector:
    """
    Service that selects optimal tools based on refined query analysis.

    Unlike generic tool selection, this service deeply understands:
    - The user's problem context
    - Technical requirements
    - Integration needs
    - Workflow stage requirements
    - User proficiency level
    """

    def __init__(self):
        """Initialize the refined query tool selector."""
        self.llm = get_shared_llm()

    async def select_optimal_tools(
        self,
        candidate_tools: List[Dict[str, Any]],
        refined_query: str,
        workflow_understanding: str = "",
        max_tools: int = 10,
    ) -> Dict[str, Any]:
        """
        Select optimal tools based on refined query analysis.

        Args:
            candidate_tools: List of tools found by search
            refined_query: Comprehensive problem analysis
            workflow_understanding: Understanding of workflow from query decomposition
            max_tools: Maximum number of tools to select

        Returns:
            Dict with selected_tools and selection reasoning
        """
        try:
            logger.info(
                f"🎯 Selecting {max_tools} optimal tools from {len(candidate_tools)} candidates"
            )

            if not candidate_tools:
                return {
                    "status": "error",
                    "message": "No candidate tools provided",
                    "selected_tools": [],
                }

            # Prepare tool information for LLM analysis
            tools_info = self._prepare_tools_for_analysis(candidate_tools)

            prompt = f"""
You are an expert system architect selecting the PERFECT tools to solve a client's problem.
You have a deep understanding of their problem and a list of candidate tools.
Your job is to select the {max_tools} BEST tools that will create the most effective solution.

PROBLEM ANALYSIS (Comprehensive Understanding):
{refined_query}

WORKFLOW UNDERSTANDING:
{workflow_understanding}

CANDIDATE TOOLS ({len(candidate_tools)} total):
{tools_info}

YOUR SELECTION CRITERIA:

1. **PROBLEM FIT**: How well does the tool solve the specific problem described?
   - Does it address the core pain points mentioned?
   - Does it support the workflow they described?
   - Does it match their technical level?

2. **FEATURE MATCH**: Does it have the required features?
   - Check against the requirements in the analysis
   - Verify it has must-have capabilities
   - Consider nice-to-have features

3. **INTEGRATION FIT**: Will it integrate with their existing tools?
   - Check if it mentions integration with tools they currently use
   - Verify API/integration capabilities if needed
   - Consider data flow requirements

4. **WORKFLOW STAGE**: Does it fit the workflow stage needed?
   - Input/Collection tools for data gathering
   - Processing/Analysis tools for transformation
   - Action/Automation tools for execution
   - Monitoring/Reporting tools for tracking
   - Integration tools for connecting systems

5. **DIVERSITY**: Select tools that cover different aspects
   - Don't select multiple tools that do the same thing
   - Ensure coverage across all workflow stages
   - Balance between different capabilities

6. **QUALITY INDICATORS**: Prefer high-quality tools
   - Tools with detailed descriptions (more information = better match)
   - Tools with websites (verified, real tools)
   - Tools from reliable sources (Pinecone > Internet)
   - Tools with specific feature lists

7. **AVOID DUPLICATES**: Never select similar tools
   - Check tool names for similarity
   - Check descriptions for overlapping functionality
   - Check websites to avoid same tool with different names

SELECTION PROCESS:

Step 1: Filter out low-quality tools (no website, vague descriptions, duplicates)

Step 2: Map each requirement from the analysis to candidate tools

Step 3: Score each tool based on how many requirements it fulfills

Step 4: Select {max_tools} highest-scoring tools ensuring diversity

Step 5: Verify no duplicates or overlapping functionality

RETURN FORMAT (ONLY JSON):
{{
  "selected_tools": [
    {{
      "tool_id": "ID from candidate tools",
      "title": "Tool name",
      "selection_reason": "SPECIFIC explanation of why this tool is perfect for their problem. Reference actual pain points and requirements from the analysis. Explain which workflow stage it covers.",
      "workflow_stage": "input|processing|action|monitoring|integration",
      "addresses_pain_points": ["Specific pain point 1", "Specific pain point 2"],
      "key_capabilities": ["Capability 1 they need", "Capability 2 they need"],
      "integration_notes": "How it integrates with their existing tools if applicable",
      "priority": "critical|high|medium|low"
    }}
  ],
  "selection_reasoning": "Overall explanation of the selection strategy and how these tools work together",
  "workflow_coverage": {{
    "input": "How input/collection needs are covered",
    "processing": "How processing/analysis needs are covered",
    "action": "How action/automation needs are covered",
    "monitoring": "How monitoring/reporting needs are covered",
    "integration": "How integration needs are covered"
  }},
  "confidence_score": 0.0-1.0,
  "missing_capabilities": ["Any requirements that couldn't be fulfilled"]
}}

CRITICAL RULES:
- Select EXACTLY {max_tools} tools (or fewer if not enough quality candidates)
- NO DUPLICATES: Each tool must be unique
- DIVERSE FUNCTIONALITY: Cover different aspects of the solution
- QUALITY OVER QUANTITY: Better to select fewer high-quality tools than include low-quality ones
- SPECIFIC REASONING: Every selection reason must reference the actual problem analysis
- PRIORITIZE COMPLETENESS: Ensure all critical workflow stages are covered

Make intelligent decisions that show deep understanding of the problem and careful consideration of each tool's fit.
"""

            response = await self.llm.generate_response(prompt)
            result = await self.llm.parse_json_response(response)

            if not result or "selected_tools" not in result:
                logger.error("❌ Failed to parse tool selection from LLM")
                raise ValueError("Invalid response from LLM")

            selected_tool_ids = {
                t.get("tool_id") for t in result.get("selected_tools", [])
            }

            # Map back to full tool objects
            final_tools = []
            for tool in candidate_tools:
                tool_id = tool.get("ID") or tool.get("id") or tool.get("Title")
                if tool_id in selected_tool_ids:
                    # Find the selection info
                    selection_info = next(
                        (
                            t
                            for t in result.get("selected_tools", [])
                            if t.get("tool_id") == tool_id
                        ),
                        {},
                    )

                    # Enrich tool with selection reasoning
                    enriched_tool = tool.copy()
                    enriched_tool["_selection_reason"] = selection_info.get(
                        "selection_reason", ""
                    )
                    enriched_tool["_workflow_stage"] = selection_info.get(
                        "workflow_stage", ""
                    )
                    enriched_tool["_priority"] = selection_info.get(
                        "priority", "medium"
                    )
                    final_tools.append(enriched_tool)

            logger.info(f"✅ Selected {len(final_tools)} optimal tools")
            logger.info(f"   Confidence score: {result.get('confidence_score', 0)}")

            return {
                "status": "success",
                "selected_tools": final_tools,
                "selection_reasoning": result.get("selection_reasoning", ""),
                "workflow_coverage": result.get("workflow_coverage", {}),
                "confidence_score": result.get("confidence_score", 0.0),
                "missing_capabilities": result.get("missing_capabilities", []),
                "total_selected": len(final_tools),
            }

        except Exception as e:
            logger.error(f"❌ Error in refined query tool selection: {e}", exc_info=True)
            # Fallback: Return top tools by similarity score
            return self._fallback_selection(candidate_tools, max_tools)

    def _prepare_tools_for_analysis(self, tools: List[Dict[str, Any]]) -> str:
        """
        Prepare tools information for LLM analysis.

        Args:
            tools: List of tool dictionaries

        Returns:
            Formatted string with tool information
        """
        try:
            tools_text = []

            for i, tool in enumerate(tools, 1):
                title = tool.get("Title") or tool.get("title") or "Unknown Tool"
                description = (
                    tool.get("Description")
                    or tool.get("description")
                    or "No description"
                )
                features = tool.get("Features") or tool.get("features") or []
                tags = tool.get("Tags (Keywords)") or tool.get("tags") or []
                website = tool.get("Website") or tool.get("website") or ""
                source = tool.get("Source") or "Unknown"
                tool_id = tool.get("ID") or tool.get("id") or title

                # Format features
                features_str = ""
                if features:
                    if isinstance(features, list):
                        features_str = ", ".join(features[:5])  # Top 5 features
                    else:
                        features_str = str(features)[:200]

                # Format tags
                tags_str = ""
                if tags:
                    if isinstance(tags, list):
                        tags_str = ", ".join(tags[:5])
                    else:
                        tags_str = str(tags)[:100]

                tool_info = f"""
{i}. **{title}** (ID: {tool_id})
   Description: {description[:300]}{"..." if len(description) > 300 else ""}
   Features: {features_str}
   Tags: {tags_str}
   Website: {website}
   Source: {source}
"""
                tools_text.append(tool_info)

            return "\n".join(tools_text)

        except Exception as e:
            logger.error(f"Error preparing tools for analysis: {e}")
            return "Error formatting tools"

    def _fallback_selection(
        self, candidate_tools: List[Dict[str, Any]], max_tools: int
    ) -> Dict[str, Any]:
        """
        Fallback selection when intelligent selection fails.

        Args:
            candidate_tools: List of candidate tools
            max_tools: Maximum tools to select

        Returns:
            Dict with selected tools
        """
        try:
            logger.warning("⚠️ Using fallback tool selection (top by similarity)")

            # Sort by similarity score if available
            sorted_tools = sorted(
                candidate_tools,
                key=lambda t: float(t.get("Similarity Score", 0)),
                reverse=True,
            )

            # Select top tools
            selected = sorted_tools[:max_tools]

            return {
                "status": "success",
                "selected_tools": selected,
                "selection_reasoning": "Fallback selection based on similarity scores",
                "workflow_coverage": {},
                "confidence_score": 0.5,
                "missing_capabilities": [],
                "total_selected": len(selected),
            }

        except Exception as e:
            logger.error(f"❌ Fallback selection failed: {e}")
            return {
                "status": "error",
                "selected_tools": [],
                "selection_reasoning": "Selection failed",
                "workflow_coverage": {},
                "confidence_score": 0.0,
                "missing_capabilities": [],
                "total_selected": 0,
            }

    async def validate_tool_coverage(
        self,
        selected_tools: List[Dict[str, Any]],
        refined_query: str,
    ) -> Dict[str, Any]:
        """
        Validate that selected tools adequately cover all requirements.

        Args:
            selected_tools: List of selected tools
            refined_query: Comprehensive problem analysis

        Returns:
            Dict with coverage analysis and gaps
        """
        try:
            logger.info("🔍 Validating tool coverage against requirements")

            tools_summary = "\n".join(
                [
                    f"- {tool.get('Title', 'Unknown')}: {tool.get('Description', '')[:100]}"
                    for tool in selected_tools
                ]
            )

            prompt = f"""
Analyze whether these selected tools adequately cover all requirements from the problem analysis.

PROBLEM ANALYSIS:
{refined_query}

SELECTED TOOLS:
{tools_summary}

Return JSON:
{{
  "coverage_score": 0.0-1.0,
  "covered_requirements": ["Requirement 1", "Requirement 2"],
  "gaps": ["Missing capability 1", "Missing capability 2"],
  "recommendations": ["Consider adding tool for X", "Y is not covered"]
}}
"""

            response = await self.llm.generate_response(prompt)
            result = await self.llm.parse_json_response(response)

            if result:
                logger.info(f"   Coverage score: {result.get('coverage_score', 0)}")
                logger.info(f"   Gaps found: {len(result.get('gaps', []))}")

            return result or {"coverage_score": 0.5, "gaps": [], "recommendations": []}

        except Exception as e:
            logger.error(f"❌ Error validating coverage: {e}")
            return {"coverage_score": 0.5, "gaps": [], "recommendations": []}
