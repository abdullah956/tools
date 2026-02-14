"""Serializers for AI Tool Recommender app."""

from rest_framework import serializers

from management.tools.models import Tool

from .models import (
    AIToolSearchLog,
    BackgroundTask,
    ChatMessage,
    ConversationSession,
    DiscoveredTool,
    RefinedQuery,
    RefineQuerySession,
    ToolComparison,
    WorkflowGeneration,
    WorkflowImplementationGuide,
)


class ToolSerializer(serializers.ModelSerializer):
    """Serializer for Tool model."""

    class Meta:
        """Meta options."""

        model = Tool
        fields = [
            "id",
            "title",
            "description",
            "category",
            "features",
            "tags",
            "website",
            "twitter",
            "facebook",
            "linkedin",
            "tiktok",
            "youtube",
            "instagram",
            "price_from",
            "price_to",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class SearchQuerySerializer(serializers.Serializer):
    """Serializer for AI tool search queries."""

    query = serializers.CharField(
        required=True,
        help_text="Search query for AI tools",
    )
    max_results = serializers.IntegerField(
        default=10,
        min_value=1,
        max_value=50,
        help_text="Maximum number of results to return",
    )
    include_pinecone = serializers.BooleanField(
        default=True,
        help_text="Include results from Pinecone vector database",
    )
    include_internet = serializers.BooleanField(
        default=True,
        help_text="Include results from internet search",
    )
    workflow_id = serializers.UUIDField(
        required=False,
        help_text="Optional workflow ID to update existing workflow instead of creating new one",
    )


class SearchResultSerializer(serializers.Serializer):
    """Serializer for search results."""

    status = serializers.CharField()
    tools = serializers.ListField(child=serializers.DictField())
    query = serializers.CharField()
    refined_query = serializers.CharField()
    total_count = serializers.IntegerField()
    pinecone_count = serializers.IntegerField()
    internet_count = serializers.IntegerField()
    response_time_ms = serializers.FloatField(required=False)
    cache_hit = serializers.BooleanField(default=False)
    workflow = serializers.DictField(required=False)


class AIToolSearchLogSerializer(serializers.ModelSerializer):
    """Serializer for AI Tool Search Log."""

    user_email = serializers.EmailField(source="user.email", read_only=True)

    class Meta:
        """Meta options."""

        model = AIToolSearchLog
        fields = [
            "id",
            "user",
            "user_email",
            "query",
            "refined_query",
            "max_results",
            "include_pinecone",
            "include_internet",
            "pinecone_results_count",
            "internet_results_count",
            "total_results_count",
            "response_time_ms",
            "cache_hit",
            "status",
            "error_message",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "user_email",
            "refined_query",
            "pinecone_results_count",
            "internet_results_count",
            "total_results_count",
            "response_time_ms",
            "cache_hit",
            "status",
            "error_message",
            "created_at",
        ]


class DiscoveredToolSerializer(serializers.ModelSerializer):
    """Serializer for Discovered Tool."""

    class Meta:
        """Meta options."""

        model = DiscoveredTool
        fields = [
            "id",
            "title",
            "description",
            "category",
            "features",
            "tags",
            "website",
            "twitter",
            "facebook",
            "linkedin",
            "instagram",
            "youtube",
            "tiktok",
            "price_from",
            "price_to",
            "pricing_model",
            "source",
            "discovery_query",
            "relevance_score",
            "status",
            "discovered_at",
            "reviewed_at",
            "added_at",
            "tool",
        ]
        read_only_fields = [
            "id",
            "discovered_at",
            "reviewed_at",
            "added_at",
            "tool",
        ]


class BatchToolAddSerializer(serializers.Serializer):
    """Serializer for batch tool addition."""

    tools = serializers.ListField(
        child=serializers.DictField(),
        help_text="List of tool data dictionaries",
    )


class WorkflowGenerationSerializer(serializers.ModelSerializer):
    """Serializer for Workflow Generation."""

    user_email = serializers.EmailField(source="user.email", read_only=True)
    tools_data = ToolSerializer(source="tools", many=True, read_only=True)
    refined_query_from_db = serializers.SerializerMethodField()
    is_refined = serializers.SerializerMethodField()

    def get_refined_query_from_db(self, obj):
        """Get refined query from database if it exists."""
        from .models import RefinedQuery

        try:
            refined_query_obj = RefinedQuery.objects.filter(workflow_id=obj.id).first()
            if refined_query_obj:
                return refined_query_obj.refined_query
            return None
        except Exception:
            return None

    def get_is_refined(self, obj):
        """Get is_refined flag based on whether refined query exists."""
        from .models import RefinedQuery

        try:
            refined_query_obj = RefinedQuery.objects.filter(workflow_id=obj.id).first()
            if refined_query_obj:
                return True
            return False  # Return False instead of None
        except Exception:
            return False  # Return False instead of None

    class Meta:
        """Meta options."""

        model = WorkflowGeneration
        fields = [
            "id",
            "user",
            "user_email",
            "query",
            "workflow_data",
            "tools",
            "tools_data",
            "tools_count",
            "generation_method",
            "generation_time_ms",
            "search_log",
            "created_at",
            "refined_query_from_db",
            "is_refined",
        ]
        read_only_fields = [
            "id",
            "user",
            "user_email",
            "tools_count",
            "generation_time_ms",
            "refined_query_from_db",
            "is_refined",
            "created_at",
        ]


class BackgroundTaskSerializer(serializers.ModelSerializer):
    """Serializer for Background Task."""

    class Meta:
        """Meta options."""

        model = BackgroundTask
        fields = [
            "task_id",
            "task_type",
            "status",
            "params",
            "result",
            "error_message",
            "created_at",
            "started_at",
            "completed_at",
            "duration_seconds",
        ]
        read_only_fields = [
            "task_id",
            "created_at",
            "started_at",
            "completed_at",
            "duration_seconds",
        ]


class ExplainToolSerializer(serializers.Serializer):
    """Serializer for tool explanation requests."""

    json_object = serializers.DictField(
        help_text="Tool data as JSON object",
    )
    query = serializers.CharField(
        help_text="User's original query",
    )


class ToolComparisonRequestSerializer(serializers.Serializer):
    """Serializer for tool comparison requests."""

    node_id = serializers.CharField(
        help_text="ID of the node in the workflow to find alternatives for",
        max_length=100,
    )
    max_results = serializers.IntegerField(
        default=8,
        min_value=1,
        max_value=20,
        help_text="Maximum number of alternative tools to return",
    )
    include_explanations = serializers.BooleanField(
        default=True,
        help_text="Whether to include detailed comparison explanations",
    )


class ToolComparisonSerializer(serializers.ModelSerializer):
    """Serializer for Tool Comparison."""

    user_email = serializers.EmailField(source="user.email", read_only=True)
    workflow_query = serializers.CharField(source="workflow.query", read_only=True)

    class Meta:
        """Meta options."""

        model = ToolComparison
        fields = [
            "id",
            "user",
            "user_email",
            "workflow",
            "workflow_query",
            "original_tool_node_id",
            "original_tool_data",
            "comparison_query",
            "alternative_tools",
            "total_alternatives_found",
            "search_time_ms",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "user",
            "user_email",
            "workflow_query",
            "total_alternatives_found",
            "search_time_ms",
            "created_at",
        ]


class ToolComparisonResultSerializer(serializers.Serializer):
    """Serializer for tool comparison results."""

    status = serializers.CharField()
    original_tool = serializers.DictField()
    alternatives = serializers.ListField(child=serializers.DictField())
    comparison_query = serializers.CharField()
    total_found = serializers.IntegerField()
    node_id = serializers.CharField()
    performance = serializers.DictField(required=False)
    message = serializers.CharField(required=False)


class WorkflowToolReplaceSerializer(serializers.Serializer):
    """Serializer for workflow tool replacement requests."""

    node_id = serializers.CharField(
        help_text="ID of the node to replace in the workflow",
        max_length=100,
    )
    new_tool_data = serializers.DictField(
        help_text="New tool data to replace the existing tool with",
    )
    preserve_connections = serializers.BooleanField(
        default=True,
        help_text="Whether to preserve existing workflow connections",
    )


class WorkflowUpdateResultSerializer(serializers.Serializer):
    """Serializer for workflow update results."""

    status = serializers.CharField()
    workflow = serializers.DictField()
    message = serializers.CharField()
    performance = serializers.DictField(required=False)
    validation = serializers.DictField(required=False)


# Conversational AI Serializers
class ConversationChatSerializer(serializers.Serializer):
    """Serializer for conversational chat requests - matches SearchQuerySerializer."""

    query = serializers.CharField(
        required=True, help_text="User's query/message for workflow generation"
    )
    max_results = serializers.IntegerField(
        default=10,
        min_value=1,
        max_value=50,
        help_text="Maximum number of results to return",
    )
    include_pinecone = serializers.BooleanField(
        default=True,
        help_text="Include results from Pinecone vector database",
    )
    include_internet = serializers.BooleanField(
        default=True,
        help_text="Include results from internet search",
    )
    workflow_id = serializers.UUIDField(
        required=True,
        help_text="Workflow ID - Required to associate chat with a workflow",
    )
    agent = serializers.ChoiceField(
        choices=[
            "refine_query_generator",
            "workflow_builder",
            "general_assistant",
            "implementation_chat",
            "tool_assistant",
        ],
        required=False,
        allow_blank=True,
        allow_null=True,
        help_text=(
            "Optional: Specify which agent to use. "
            "Available agents: refine_query_generator (asks questions), workflow_builder (builds workflows), "
            "general_assistant (general help), implementation_chat (implementation guidance), "
            "tool_assistant (tool questions). If not provided, agent will be auto-selected based on context."
        ),
    )
    tool_id = serializers.CharField(
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Optional tool ID to focus conversation on a specific tool (used by tool_assistant agent)",
    )
    context = serializers.CharField(
        required=False,
        allow_null=True,
        allow_blank=True,
        help_text="Optional context for implementation chat agent",
    )


class AddWorkflowNodeSerializer(serializers.Serializer):
    """Serializer for adding a single workflow node."""

    session_id = serializers.UUIDField(
        required=True, help_text="Conversation session ID"
    )
    tool_query = serializers.CharField(
        required=True, help_text="Query to find the tool to add"
    )
    position = serializers.DictField(
        required=False, help_text="Position for the new node"
    )


class SelectToolSerializer(serializers.Serializer):
    """Serializer for selecting a tool from search results."""

    session_id = serializers.UUIDField(
        required=True, help_text="Conversation session ID"
    )
    tool_index = serializers.IntegerField(
        required=True, help_text="Index of the selected tool from available tools"
    )


class ConnectNodesSerializer(serializers.Serializer):
    """Serializer for connecting workflow nodes."""

    session_id = serializers.UUIDField(
        required=True, help_text="Conversation session ID"
    )
    source_node_id = serializers.CharField(
        required=True, help_text="ID of the source node"
    )
    target_node_id = serializers.CharField(
        required=True, help_text="ID of the target node"
    )


class GenerateFromChatSerializer(serializers.Serializer):
    """Serializer for generating workflow from chat history."""

    session_id = serializers.UUIDField(
        required=True, help_text="Conversation session ID"
    )
    final_instructions = serializers.CharField(
        required=False, help_text="Final instructions for workflow generation"
    )


class ConversationSessionSerializer(serializers.ModelSerializer):
    """Serializer for conversation sessions."""

    class Meta:
        model = ConversationSession
        fields = [
            "id",
            "session_id",
            "original_query",
            "chat_history",
            "current_context",
            "workflow_nodes",
            "workflow_edges",
            "suggested_tools",
            "is_active",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "session_id",
            "created_at",
            "updated_at",
        ]


class ChatMessageSerializer(serializers.ModelSerializer):
    """Serializer for chat messages."""

    class Meta:
        model = ChatMessage
        fields = [
            "id",
            "user_message",
            "ai_response",
            "message_type",
            "tools_mentioned",
            "workflow_changes",
            "intent_analysis",
            "created_at",
        ]
        read_only_fields = [
            "id",
            "created_at",
        ]


class RefinedQuerySerializer(serializers.ModelSerializer):
    """Serializer for refined queries."""

    is_refined = serializers.SerializerMethodField()

    class Meta:
        model = RefinedQuery
        fields = [
            "id",
            "workflow_id",
            "original_query",
            "refined_query",
            "workflow_info",
            "is_refined",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
        ]

    def get_is_refined(self, obj):
        """Always return True to indicate this is a refined query."""
        return True


class ConversationalResponseSerializer(serializers.Serializer):
    """Serializer for conversational responses - matches SearchResultSerializer."""

    status = serializers.CharField()
    query = serializers.CharField()
    refined_query = serializers.CharField()
    tools = serializers.ListField(child=serializers.DictField())
    total_count = serializers.IntegerField()
    pinecone_count = serializers.IntegerField()
    internet_count = serializers.IntegerField()
    workflow = serializers.DictField()
    workflow_id = serializers.CharField(required=False)
    response_time_ms = serializers.FloatField(required=False)
    cached = serializers.BooleanField(default=False)
    message = serializers.CharField()
    new_tools_discovered = serializers.IntegerField(required=False)
    auto_discovery = serializers.DictField(required=False)
    remaining_searches = serializers.DictField(required=False)
    examples = serializers.ListField(
        child=serializers.CharField(), required=False
    )  # Text examples list
    tool_examples = serializers.ListField(
        child=serializers.CharField(), required=False
    )  # Tool names list
    progress_percentage = serializers.IntegerField(
        required=False,
        help_text="Progress percentage (0-100) based on workflow_info fields filled",
    )
    is_refined = serializers.BooleanField(
        required=False, help_text="Flag indicating if this is a refined query (Phase 3)"
    )
    refined_query_from_db = serializers.CharField(
        required=False,
        allow_null=True,
        help_text="Refined query fetched from database (Markdown format, null until workflow is generated)",
    )


# Query Refinement Serializers
class RefineQueryInitSerializer(serializers.Serializer):
    """Serializer for initializing query refinement session."""

    original_prompt = serializers.CharField(
        required=True, help_text="User's initial high-level request"
    )


class RefineQueryResponseSerializer(serializers.Serializer):
    """Serializer for query refinement responses."""

    session_id = serializers.UUIDField()
    original_prompt = serializers.CharField()
    refined_query = serializers.CharField()
    status = serializers.CharField()
    refinement_suggestions = serializers.ListField(
        child=serializers.CharField(), required=False
    )
    clarification_questions = serializers.ListField(
        child=serializers.CharField(), required=False
    )
    next_steps = serializers.ListField(child=serializers.CharField(), required=False)


class RefineQuerySessionSerializer(serializers.ModelSerializer):
    """Serializer for refine query sessions."""

    class Meta:
        model = RefineQuerySession
        fields = [
            "id",
            "session_id",
            "original_prompt",
            "refined_query",
            "refinement_history",
            "status",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "session_id",
            "created_at",
            "updated_at",
        ]


class QueryRefinementSerializer(serializers.Serializer):
    """Serializer for query refinement requests."""

    session_id = serializers.UUIDField(required=True, help_text="Refinement session ID")
    user_feedback = serializers.CharField(
        required=False, help_text="User's feedback on refinement suggestions"
    )
    accept_refinement = serializers.BooleanField(
        required=False, help_text="Whether user accepts the refined query"
    )


class WorkflowImplementationGuideSerializer(serializers.ModelSerializer):
    """Serializer for WorkflowImplementationGuide model."""

    class Meta:
        """Meta options."""

        model = WorkflowImplementationGuide
        fields = [
            "id",
            "workflow_id",
            "implementation_guide",
            "tools_count",
            "generation_time_ms",
            "status",
            "error_message",
            "created_at",
            "updated_at",
        ]
        read_only_fields = [
            "id",
            "tools_count",
            "generation_time_ms",
            "status",
            "error_message",
            "created_at",
            "updated_at",
        ]


class WorkflowImplementationRequestSerializer(serializers.Serializer):
    """Serializer for workflow implementation guide request."""

    workflow_id = serializers.UUIDField(
        required=True,
        help_text="UUID of the workflow to generate implementation guide for",
    )


class WorkflowImplementationResponseSerializer(serializers.Serializer):
    """Serializer for workflow implementation guide response."""

    status = serializers.CharField(help_text="Status: created, updated, or error")
    implementation_guide = serializers.CharField(
        required=False, help_text="Generated implementation guide in Markdown format"
    )
    guide_id = serializers.UUIDField(
        required=False, help_text="ID of the implementation guide record"
    )
    tools_count = serializers.IntegerField(
        required=False, help_text="Number of tools in the workflow"
    )
    generation_time_ms = serializers.FloatField(
        required=False, help_text="Time taken to generate the guide in milliseconds"
    )
    error = serializers.CharField(
        required=False, help_text="Error message if generation failed"
    )
