"""Intelligent Search Query Generator - Generates detailed search queries from refined query."""

import logging
from typing import Any, Dict

from ai_tool_recommender.ai_agents.core.llm import get_shared_llm

logger = logging.getLogger(__name__)


class IntelligentQueryGenerator:
    """Service to generate intelligent search queries from refined query analysis."""

    def __init__(self):
        """Initialize the intelligent query generator."""
        self.llm = get_shared_llm()

    async def generate_search_queries(
        self,
        refined_query: str,
        original_query: str = "",
        max_queries: int = 8,
    ) -> Dict[str, Any]:
        """
        Generate intelligent, paragraph-level search queries from refined query.

        These search queries explain tool functionality WITHOUT mentioning tool names,
        allowing Pinecone to find tools based on description similarity.

        Args:
            refined_query: The comprehensive refined query analysis
            original_query: Original user query for context
            max_queries: Maximum number of search queries to generate

        Returns:
            Dict with search_queries list and metadata
        """
        try:
            logger.info(
                f"🎯 Generating {max_queries} intelligent search queries from refined query"
            )

            prompt = f"""
You are an AI tool discovery expert. Your task is to analyze a comprehensive problem analysis and generate highly descriptive search queries that will find the PERFECT tools without mentioning specific tool names.

ORIGINAL USER REQUEST:
"{original_query}"

COMPREHENSIVE PROBLEM ANALYSIS:
{refined_query}

YOUR TASK:
Generate {max_queries} detailed, paragraph-level search queries that describe the FUNCTIONALITY, FEATURES, and CAPABILITIES needed to solve this problem.

CRITICAL RULES FOR SEARCH QUERY GENERATION:

1. **USE TOOL NAMES IF SPECIFIED**: If the analysis mentions specific tools the user already uses (e.g., "Alexa", "iPhone", "Slack"), YOU MUST INCLUDE THEM in the queries to find compatible integrations.

2. **DESCRIBE FUNCTIONALITY**: For new tools needed, explain what the tool should DO, not what it's called
   ❌ BAD: "Find Cursor or VS Code alternatives"
   ✅ GOOD: "An AI-integrated code editor with intelligent autocomplete that understands context across your entire codebase, provides real-time suggestions, can refactor code automatically, includes a prompt bar to plan and search features, has agentic systems that can write code based on natural language queries, and helps developers code more efficiently with advanced AI assistance"

3. **PARAGRAPH FORMAT**: Each search query should be 2-4 sentences describing the tool's purpose and key features
   - First sentence: Primary purpose and main problem it solves
   - Second sentence: Key features and capabilities
   - Third sentence: Integration needs or workflow position
   - Fourth sentence: (Optional) Technical requirements or user experience

4. **BE EXTREMELY DETAILED**: Include:
   - What domain/category (CRM, automation, analytics, etc.)
   - What it does (manage contacts, automate emails, analyze data, etc.)
   - Key capabilities (integration, reporting, scheduling, etc.)
   - How it fits in the workflow (data collection, processing, action, etc.)
   - User experience aspects (easy-to-use, no-code, API-first, etc.)

5. **COVER ALL WORKFLOW STAGES**: Generate queries for different parts of the solution:
   - Data collection/input tools
   - Processing/analysis tools
   - Action/automation tools
   - Communication/collaboration tools
   - Monitoring/reporting tools
   - Integration/connector tools

6. **USE PROBLEM CONTEXT**: Reference the actual problem and requirements from the analysis
   - If they mentioned automation, emphasize automation features
   - If they mentioned integrations, describe integration capabilities
   - If they mentioned ease of use, emphasize user-friendly features
   - If they mentioned specific workflows, describe tools that support those workflows

7. **TECHNICAL PRECISION**: Use industry-standard terminology
   - Instead of "talks to other tools" → "REST API integration with webhook support"
   - Instead of "keeps track" → "real-time data synchronization and change tracking"
   - Instead of "sends messages" → "automated multi-channel communication orchestration"

EXAMPLE INPUT:
"I need to manage customer relationships and automate my sales follow-ups. I'm currently using spreadsheets and it's getting messy."

EXAMPLE OUTPUT QUERIES:

1. "A customer relationship management platform that centralizes contact information, tracks all customer interactions across multiple channels, and maintains a complete history of communications. The system should provide intuitive contact organization with custom fields, automated data entry, and intelligent contact segmentation. It needs to integrate seamlessly with email, calendar, and communication tools while offering mobile access for on-the-go updates."

2. "An intelligent sales automation solution that automatically schedules and sends personalized follow-up sequences based on customer behavior and interaction history. The tool should support multi-channel outreach including email, SMS, and social media, with smart timing optimization to send messages when prospects are most likely to engage. It must include response tracking, automatic task creation for manual follow-ups, and behavioral triggers that adapt sequences based on how contacts interact with your content."

3. "A workflow automation platform with visual builder that connects various business applications to eliminate manual data entry and repetitive tasks. The system should offer pre-built integrations with major business tools, support complex conditional logic for different scenarios, and enable data transformation between applications. It needs to provide monitoring and error handling to ensure reliable automation execution."

4. "An analytics and reporting dashboard that visualizes sales pipeline data, tracks key performance metrics, and provides actionable insights through automated reports. The solution should aggregate data from multiple sources, offer customizable visualizations, and support scheduled report delivery. It must enable forecasting based on historical data and provide alerts for significant changes in key metrics."

NOW, ANALYZE THE REFINED QUERY AND GENERATE {max_queries} SEARCH QUERIES:

Return ONLY valid JSON in this format:
{{
  "search_queries": [
    {{
      "query": "Detailed 2-4 sentence description of tool functionality without naming any tools",
      "category": "Category like 'CRM', 'Automation', 'Analytics', etc.",
      "workflow_stage": "input|processing|action|monitoring|integration",
      "priority": "high|medium|low",
      "reasoning": "Why this tool category is needed for the solution"
    }}
  ],
  "workflow_understanding": "Brief summary of the overall workflow these tools will create"
}}

IMPORTANT:
- Each query should be unique and describe different tool functionality
- Focus on capabilities and features, not brand names
- Make queries detailed enough that Pinecone can find the right tools by description similarity
- Cover all aspects of the solution described in the refined query
- Prioritize based on the problem's core requirements
"""

            response = await self.llm.generate_response(prompt)
            result = await self.llm.parse_json_response(response)

            if not result or "search_queries" not in result:
                logger.error("❌ Failed to parse search queries from LLM response")
                raise ValueError("Invalid response from LLM")

            search_queries = result.get("search_queries", [])
            logger.info(f"✅ Generated {len(search_queries)} intelligent search queries")

            # Log sample queries for debugging
            for i, query_obj in enumerate(search_queries[:2], 1):
                query_text = query_obj.get("query", "")
                logger.info(
                    f"  Query {i} ({query_obj.get('category', 'Unknown')}): {query_text[:100]}..."
                )

            return {
                "status": "success",
                "search_queries": search_queries,
                "workflow_understanding": result.get("workflow_understanding", ""),
                "total_queries": len(search_queries),
            }

        except Exception as e:
            logger.error(
                f"❌ Error generating intelligent search queries: {e}", exc_info=True
            )
            # Fallback: Create basic queries from original query
            # CRITICAL FIX: Do NOT use the entire refined_query as it is too large for search
            # If original_query is empty, use a truncated version of refined_query

            base_query = original_query
            if not base_query and refined_query:
                # Extract first 300 chars or first paragraph of refined query
                base_query = refined_query[:300].split("\n")[0]
                if len(base_query) < 50:  # If too short, take a bit more
                    base_query = refined_query[:300]

            logger.warning(
                f"⚠️ Using fallback query generation with base query: {base_query[:100]}..."
            )

            fallback_query = (
                f"Tools and software solutions that can help with: {base_query}. "
                f"Features should include automation, integration capabilities, "
                f"user-friendly interface, and data management."
            )

            return {
                "status": "success",
                "search_queries": [
                    {
                        "query": fallback_query,
                        "category": "General",
                        "workflow_stage": "general",
                        "priority": "high",
                        "reasoning": "Fallback query generated from original request",
                    }
                ],
                "workflow_understanding": "General workflow to address user's needs",
                "total_queries": 1,
            }

    async def enhance_query_for_pinecone(self, query_text: str) -> str:
        """
        Enhance a single query text for better Pinecone matching.

        Args:
            query_text: The query text to enhance

        Returns:
            Enhanced query optimized for embedding similarity
        """
        try:
            # Extract key terms and concepts
            prompt = f"""
Extract the key capabilities, features, and technical terms from this tool description.
Return them as a space-separated list of keywords and phrases for optimal semantic search.

Tool Description:
{query_text}

Return ONLY the extracted terms, no explanation.
Example format: "customer relationship management contact tracking automated follow-up email integration pipeline visualization"
"""

            enhanced = await self.llm.generate_response(prompt)
            enhanced = enhanced.strip().strip('"').strip("'")

            # Combine original with enhancement
            combined = f"{query_text} {enhanced}"

            logger.debug(f"Enhanced query length: {len(query_text)} → {len(combined)}")
            return combined

        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query_text  # Return original if enhancement fails
