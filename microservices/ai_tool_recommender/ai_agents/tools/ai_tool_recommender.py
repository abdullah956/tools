"""AI Tool Recommender service that combines Pinecone and Internet Search."""

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

from microservices.ai_tool_recommender.ai_agents.core.llm import get_shared_llm
from microservices.ai_tool_recommender.ai_agents.core.performance_monitor import (
    performance_monitor,
)
from microservices.ai_tool_recommender.ai_agents.core.redis_cache import (
    query_cache,
    redis_cache,
)
from microservices.ai_tool_recommender.ai_agents.core.validation import (
    ToolDataFormatter,
    ToolDataValidator,
)
from microservices.ai_tool_recommender.ai_agents.tools.internet_search import (
    InternetSearchService,
)
from microservices.ai_tool_recommender.ai_agents.tools.pinecone import PineconeService

logger = logging.getLogger(__name__)


class AIToolRecommender:
    """AI Tool Recommender service combining Pinecone and Internet Search."""

    def __init__(self):
        """Initialize the AI Tool Recommender service with Redis caching."""
        self.pinecone_service = PineconeService()
        self.internet_service = InternetSearchService()
        self._redis_connected = False
        self.workflow_prompt = self._create_workflow_prompt()
        logger.info("AI Tool Recommender service initialized with Redis caching")

    async def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            await redis_cache.connect()
            self._redis_connected = True
            logger.info("✅ Redis connected successfully")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
            self._redis_connected = False

    async def search_tools(
        self,
        query: str,
        max_results: int = 10,
        include_pinecone: bool = True,
        include_internet: bool = True,
    ) -> Dict[str, Any]:
        """Search for AI tools with Redis caching and background processing.

        Args:
            query: Search query
            max_results: Maximum number of results
            include_pinecone: Whether to include Pinecone results
            include_internet: Whether to include internet search results

        Returns:
            Dictionary with search results and performance metrics
        """
        try:
            import time

            start_time = time.time()
            logger.info(f"Starting optimized tool search for: {query[:50]}...")

            # Check cache first
            if self._redis_connected:
                cached_results = await query_cache.get_query_results(query, max_results)
                if cached_results:
                    logger.info("🚀 Cache hit - returning cached results")
                    return {
                        **cached_results,
                        "performance": {
                            "total_time": 0.1,
                            "status": "cached",
                            "optimization_level": "ultra_fast",
                        },
                    }
            else:
                # Try to initialize Redis if not connected
                await self._initialize_redis()

            # Search all sources in parallel for maximum speed
            async with performance_monitor.time_operation("total_search"):
                all_tools = await self._search_all_sources(
                    query,
                    max_results,
                    include_pinecone,
                    include_internet,
                )

            if not all_tools:
                logger.warning("No tools found from any source")
                return {
                    "status": "error",
                    "message": "No relevant AI tools found",
                    "tools": [],
                }

            logger.info(f"Found {len(all_tools)} tools from all sources")

            # Fast filtering
            async with performance_monitor.time_operation("filtering"):
                filtered_tools = await self._filter_tools(query, all_tools)

            if not filtered_tools:
                logger.warning("No tools selected after filtering")
                return {
                    "status": "error",
                    "message": "No tools selected after filtering",
                    "tools": [],
                }

            logger.info(f"Selected {len(filtered_tools)} tools after filtering")

            # Fast validation with parallel processing
            async with performance_monitor.time_operation("validation"):
                validator = ToolDataValidator()
                validated_tools = await validator.validate_tools_batch(filtered_tools)

                logger.info(
                    f"Validated {len(validated_tools)} tools with all required fields"
                )

            # Calculate performance metrics
            total_time = time.time() - start_time
            performance_monitor.record_request(total_time)

            # Prepare response
            response = {
                "status": "success",
                "tools": validated_tools,
                "message": f"Found {len(validated_tools)} relevant tools with complete data",
                "count": len(validated_tools),
                "validation_report": validator.get_validation_report(validated_tools),
                "performance": {
                    "total_time": round(total_time, 2),
                    "status": performance_monitor.get_performance_report()[
                        "performance_status"
                    ],
                    "optimization_level": "ultra_fast" if total_time < 2.0 else "fast",
                },
            }

            # Cache results for future requests
            if self._redis_connected:
                await query_cache.set_query_results(query, max_results, response)

            return response

        except Exception as e:
            logger.error(f"Error in search_tools: {e}")
            import traceback

            traceback.print_exc()
            return {"status": "error", "message": str(e), "tools": []}

    async def _search_all_sources(
        self,
        query: str,
        max_results: int,
        include_pinecone: bool,
        include_internet: bool,
    ) -> List[Dict[str, Any]]:
        """Search all available sources for AI tools in parallel for maximum speed."""
        import asyncio

        all_tools = []
        tasks = []

        # Create parallel tasks for both sources
        if include_pinecone:
            pinecone_task = asyncio.create_task(
                self.pinecone_service.search_tools(query, max_results // 2)
            )
            tasks.append(("pinecone", pinecone_task))

        if include_internet:
            internet_task = asyncio.create_task(
                self.internet_service.search_ai_tools(query, max_results // 2)
            )
            tasks.append(("internet", internet_task))

        # Wait for all tasks to complete in parallel
        if tasks:
            results = await asyncio.gather(
                *[task for _, task in tasks], return_exceptions=True
            )

            for _i, (source, result) in enumerate(
                zip([name for name, _ in tasks], results)
            ):
                if isinstance(result, Exception):
                    logger.error(f"{source} search failed: {result}")
                else:
                    all_tools.extend(result)
                    logger.info(f"{source} returned {len(result)} results")

        logger.info(f"Total results from all sources: {len(all_tools)}")
        return all_tools

    async def _filter_tools(
        self, query: str, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter and select the best tools using LLM."""
        try:
            # Prepare tools data for LLM
            tools_data = ToolDataFormatter.prepare_tools_data_for_prompt(tools)

            # Create prompt for LLM selection
            prompt = f"""Given the user query: "{query}"

            And the following AI tools:
            {tools_data}

            Select the most relevant tools that would work well together in a workflow.

            CRITICAL REQUIREMENTS:
            1. MUST include tools from BOTH Pinecone Vector Database AND Internet Search (if available)
            2. Select 5-8 tools total for a good workflow
            3. Prioritize direct relevance to the user's needs
            4. Ensure complementary functionality between tools
            5. Create a logical workflow sequence

            MANDATORY: You MUST select at least 2-3 tools from EACH source to ensure diversity.
            Look at the "Source" field for each tool to identify which source it comes from.

            Return ONLY numbers between 1 and {len(tools)}, separated by commas.
            Example response: 1,4,7,10,15,20
            Do not include any other text, quotes, or explanations - just the numbers.
            """  # nosec B608

            # Get LLM recommendation
            response_text = await get_shared_llm().generate_response(prompt)
            cleaned_response = response_text.replace('"', "").replace("'", "")

            # Parse selected indices
            selected_indices = []
            for num_str in cleaned_response.split(","):
                try:
                    idx = int(num_str.strip()) - 1
                    if 0 <= idx < len(tools):
                        selected_indices.append(idx)
                except ValueError:
                    continue

            # Ensure we have results
            if not selected_indices:
                selected_indices = list(range(min(5, len(tools))))

            # Force include tools from both sources if missing
            pinecone_indices = [
                i
                for i, tool in enumerate(tools)
                if tool.get("Source") == "Pinecone Vector Database"
            ]
            internet_indices = [
                i
                for i, tool in enumerate(tools)
                if "Internet Search" in tool.get("Source", "")
            ]

            # Add tools from missing sources
            if (
                not any(i in selected_indices for i in pinecone_indices)
                and pinecone_indices
            ):
                selected_indices.extend(pinecone_indices[:2])
                logger.info("Force-added Pinecone tools for diversity")

            if (
                not any(i in selected_indices for i in internet_indices)
                and internet_indices
            ):
                selected_indices.extend(internet_indices[:2])
                logger.info("Force-added Internet tools for diversity")

            # Remove duplicates and limit
            selected_indices = list(set(selected_indices))[:8]

            # Filter results
            filtered = [tools[i] for i in selected_indices]

            logger.info(f"Filtered to {len(filtered)} results")
            sources = [tool.get("Source", "Unknown") for tool in filtered]
            logger.info(f"Selected tools from sources: {sources}")

            return filtered

        except Exception as e:
            logger.error(f"Filter error: {e}")
            return tools[:5]

    async def _generate_workflow(
        self, query: str, tools: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate workflow using existing prompt service."""
        try:
            if not tools:
                return None

            logger.info(f"Generating workflow for {len(tools)} tools")

            # Prepare tools data for the prompt
            tools_data = ToolDataFormatter.prepare_tools_data_for_prompt(tools)

            # Create workflow prompt
            prompt_template = self._create_workflow_prompt()
            prompt = prompt_template.format(task=query, tools=tools_data)

            # Get workflow from LLM
            response_text = await get_shared_llm().generate_response(prompt)

            # Parse JSON response
            try:
                workflow_data = await get_shared_llm().parse_json_response(
                    response_text
                )
                logger.info("Successfully parsed LLM workflow response")
                return workflow_data

            except Exception as json_error:
                logger.error(f"JSON parsing error: {json_error}")
                return self._create_fallback_workflow(tools)

        except Exception as e:
            logger.error(f"Error generating workflow: {e}")
            return self._create_fallback_workflow(tools)

    def _create_fallback_workflow(self, tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a simple fallback workflow with proper structure."""
        return {
            "query": "Generated workflow",
            "nodes": self._create_workflow_nodes(tools),
            "edges": self._create_workflow_edges(len(tools)),
        }

    def _create_workflow_nodes(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create workflow nodes with proper structure."""
        nodes = []

        # Add trigger node
        trigger_node = {
            "id": "trigger_start",
            "type": "trigger",
            "data": {
                "label": "Workflow Start",
                "description": "Starting point of the automation workflow",
                "features": ["Initiates the automation process"],
                "tags": ["start", "automation"],
                "website": "",
                "twitter": "",
                "facebook": "",
                "linkedin": "",
                "instagram": "",
            },
        }
        nodes.append(trigger_node)

        # Add tool nodes
        for i, tool in enumerate(tools[:5], 1):  # Limit to 5 tools
            node_id = f"tool_{i:03d}"
            node = {
                "id": node_id,
                "type": "tool",
                "data": {
                    "label": tool.get("Title", "Unknown Tool"),
                    "description": tool.get("Description", ""),
                    "features": (
                        tool.get("Features", "").split(",")
                        if tool.get("Features")
                        else []
                    ),
                    "tags": (
                        tool.get("Category", "").split(",")
                        if tool.get("Category")
                        else []
                    ),
                    "website": tool.get("Website", ""),
                    "twitter": tool.get("Twitter", ""),
                    "facebook": tool.get("Facebook", ""),
                    "linkedin": tool.get("LinkedIn", ""),
                    "instagram": tool.get("Instagram", ""),
                },
            }
            nodes.append(node)

        return nodes

    def _create_workflow_edges(self, tool_count: int) -> List[Dict[str, Any]]:
        """Create workflow edges connecting nodes."""
        edges = []

        for i in range(1, tool_count + 1):
            edge_id = f"edge_{i:03d}"
            source_id = f"tool_{i - 1:03d}" if i > 1 else "trigger_start"
            target_id = f"tool_{i:03d}"

            edge = {
                "id": edge_id,
                "source": source_id,
                "target": target_id,
                "type": "default",
            }
            edges.append(edge)

        return edges

    def _create_workflow_prompt(self) -> ChatPromptTemplate:
        """Create the workflow generation prompt template."""
        return ChatPromptTemplate.from_template(
            """
You are an automation expert who creates simple tool connection sequences.

### Task:
Create a simple connection sequence for these tools to complete: {task}

**Available Tools:** {tools}

### Requirements:
1. **Select Best Tools**: Choose the most relevant tools from the list
2. **Source Diversity**: MUST include tools from BOTH Pinecone Vector Database AND Internet Search sources
3. **Create Sequence**: Show the order tools should be used
4. **Define Connections**: Specify how each tool connects to the next
5. **Add Conditional Logic**: Include if-else conditions and decision points in the workflow
6. **Keep It Simple**: Focus only on the workflow sequence

**IMPORTANT**: You must select at least 1 tool from Pinecone Vector Database and at least 1 tool from Internet Search to ensure source diversity.

### Response Format:
Return ONLY valid JSON with this EXACT structure (15 fields only):

```json
{{
    "query": "Original user query",
    "nodes": [
        {{
            "id": "node_001",
            "type": "tool",
            "data": {{
                "label": "Tool Name",
                "description": "Tool description",
                "features": ["feature1", "feature2"],
                "tags": ["tag1", "tag2"],
                "website": "https://example.com",
                "twitter": "https://twitter.com/username",
                "facebook": "https://facebook.com/username",
                "linkedin": "https://linkedin.com/company/username",
                "instagram": "https://instagram.com/username"
            }}
        }}
    ],
    "edges": [
        {{
            "id": "edge_001",
            "source": "node_001",
            "target": "node_002",
            "type": "default"
        }}
    ]
}}
```

**CRITICAL**: Return ONLY valid JSON. No explanations, no markdown, no code blocks. Start with {{ and end with }}.

"""
        )

    async def generate_workflow(self, task: str, tools: list) -> dict:
        """Generate a simple tool connection sequence.

        Args:
            task: The task description
            tools: List of available tools

        Returns:
            Generated workflow as dictionary
        """
        import asyncio

        try:
            # Format tools for the prompt with source information
            tools_text = "\n".join(
                [
                    f"- {tool.get('Title', 'Unknown')} ({tool.get('Source', 'Unknown Source')}): {tool.get('Description', 'No description')}"
                    for tool in tools
                ]
            )

            # Generate workflow using LLM with reduced timeout to prevent gateway timeouts
            try:
                response = await asyncio.wait_for(
                    get_shared_llm().generate_response(
                        self.workflow_prompt.format(task=task, tools=tools_text)
                    ),
                    timeout=120.0,  # 2 minutes - reduced from 4 to prevent gateway timeouts
                )
            except asyncio.TimeoutError:
                print("Workflow generation timed out, using fallback")
                return self._create_fallback_workflow(tools)

            # Parse JSON response
            try:
                workflow = self._parse_llm_response(response)
                if workflow:
                    print(
                        f"Successfully parsed LLM workflow with {len(workflow.get('nodes', []))} nodes and {len(workflow.get('edges', []))} edges"
                    )
                    return workflow
                else:
                    print("Failed to parse LLM response, using fallback workflow")
                    return self._create_fallback_workflow(tools)

            except Exception as e:
                print(f"JSON parsing failed: {e}")
                print(f"Raw response: {response[:500]}...")
                print("Using fallback workflow")
                return self._create_fallback_workflow(tools)

        except Exception as e:
            print(f"Error generating workflow: {e}")
            return self._create_fallback_workflow(tools)

    def _parse_llm_response(self, response: str) -> dict:
        """Parse LLM response and extract valid JSON workflow."""
        try:
            # Clean the response first
            cleaned_response = response.strip()

            # Remove markdown code blocks if present
            if "```json" in cleaned_response:
                cleaned_response = (
                    cleaned_response.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in cleaned_response:
                cleaned_response = (
                    cleaned_response.split("```")[1].split("```")[0].strip()
                )

            # Find JSON content between first { and last }
            start_idx = cleaned_response.find("{")
            end_idx = cleaned_response.rfind("}")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_content = cleaned_response[start_idx : end_idx + 1]
            else:
                json_content = cleaned_response

            # Try to parse the JSON
            workflow = json.loads(json_content)

            # Validate the workflow structure
            if not isinstance(workflow, dict):
                return None

            if (
                "query" not in workflow
                or "nodes" not in workflow
                or "edges" not in workflow
            ):
                return None

            if not isinstance(workflow["nodes"], list) or not isinstance(
                workflow["edges"], list
            ):
                return None

            # Validate node structure
            for node in workflow["nodes"]:
                if (
                    not isinstance(node, dict)
                    or "id" not in node
                    or "type" not in node
                    or "data" not in node
                ):
                    return None
                if not isinstance(node["data"], dict) or "label" not in node["data"]:
                    return None

            # Validate edge structure
            for edge in workflow["edges"]:
                if (
                    not isinstance(edge, dict)
                    or "id" not in edge
                    or "source" not in edge
                    or "target" not in edge
                    or "type" not in edge
                ):
                    return None

            return workflow

        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"Parse error: {e}")
            return None
