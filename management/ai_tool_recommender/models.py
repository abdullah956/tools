"""Models for AI Tool Recommender app."""

import uuid

from django.conf import settings
from django.db import models


class AIToolSearchLog(models.Model):
    """Log of AI tool searches with results and metadata."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="ai_tool_searches",
        null=True,
        blank=True,
    )
    query = models.TextField(help_text="Original search query")
    refined_query = models.TextField(
        help_text="Refined query after processing", blank=True
    )
    max_results = models.IntegerField(default=10)
    include_pinecone = models.BooleanField(default=True)
    include_internet = models.BooleanField(default=True)

    # Results metadata
    pinecone_results_count = models.IntegerField(default=0)
    internet_results_count = models.IntegerField(default=0)
    total_results_count = models.IntegerField(default=0)

    # Performance metrics
    response_time_ms = models.FloatField(null=True, blank=True)
    cache_hit = models.BooleanField(default=False)

    # Status
    status = models.CharField(
        max_length=20,
        choices=[
            ("success", "Success"),
            ("error", "Error"),
            ("timeout", "Timeout"),
        ],
        default="success",
    )
    error_message = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        """Meta options."""

        db_table = "ai_tool_search_logs"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["status", "created_at"]),
        ]

    def __str__(self):
        """String representation."""
        return f"Search: {self.query[:50]} - {self.created_at}"


class DiscoveredTool(models.Model):
    """Newly discovered AI tools from internet searches."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=255)
    description = models.TextField()
    category = models.CharField(max_length=100, blank=True)
    features = models.TextField(blank=True)
    tags = models.CharField(max_length=500, blank=True)
    website = models.URLField(max_length=500)

    # Social media links
    twitter = models.URLField(max_length=500, blank=True)
    facebook = models.URLField(max_length=500, blank=True)
    linkedin = models.URLField(max_length=500, blank=True)
    instagram = models.URLField(max_length=500, blank=True)
    youtube = models.URLField(max_length=500, blank=True)
    tiktok = models.URLField(max_length=500, blank=True)

    # Pricing information
    price_from = models.CharField(max_length=50, blank=True)
    price_to = models.CharField(max_length=50, blank=True)
    pricing_model = models.CharField(max_length=100, blank=True)

    # Discovery metadata
    source = models.CharField(max_length=50, default="internet")
    discovery_query = models.TextField(help_text="Query that discovered this tool")
    relevance_score = models.FloatField(default=0.0)

    # Status
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending Review"),
            ("approved", "Approved"),
            ("rejected", "Rejected"),
            ("added", "Added to Main Database"),
        ],
        default="pending",
    )

    # Timestamps
    discovered_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    added_at = models.DateTimeField(null=True, blank=True)

    # Link to actual Tool if approved and added
    tool = models.ForeignKey(
        "tools.Tool",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="discovered_from",
    )

    class Meta:
        """Meta options."""

        db_table = "discovered_tools"
        ordering = ["-discovered_at"]
        indexes = [
            models.Index(fields=["status", "discovered_at"]),
            models.Index(fields=["website"]),
        ]

    def __str__(self):
        """String representation."""
        return f"{self.title} ({self.status})"


class WorkflowGeneration(models.Model):
    """Generated workflows for tool recommendations."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="workflow_generations",
        null=True,
        blank=True,
    )
    query = models.TextField(help_text="Original query for workflow")
    workflow_data = models.JSONField(
        help_text="Complete workflow JSON with nodes and edges"
    )

    # Tools included in workflow
    tools = models.ManyToManyField(
        "tools.Tool",
        related_name="workflows",
        blank=True,
    )
    tools_count = models.IntegerField(default=0)

    # Generation metadata
    generation_method = models.CharField(
        max_length=50,
        choices=[
            ("llm", "LLM Generated"),
            ("fallback", "Fallback Template"),
        ],
        default="llm",
    )
    generation_time_ms = models.FloatField(null=True, blank=True)

    # Related search log
    search_log = models.ForeignKey(
        AIToolSearchLog,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="workflows",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        """Meta options."""

        db_table = "workflow_generations"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
        ]

    def __str__(self):
        """String representation."""
        return f"Workflow: {self.query[:50]} - {self.tools_count} tools"


class ToolComparison(models.Model):
    """Track tool comparisons for battle cards functionality."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="tool_comparisons",
        null=True,
        blank=True,
    )
    workflow = models.ForeignKey(
        WorkflowGeneration,
        on_delete=models.CASCADE,
        related_name="tool_comparisons",
    )

    # Original tool being compared
    original_tool_node_id = models.CharField(
        max_length=100, help_text="Node ID in workflow"
    )
    original_tool_data = models.JSONField(
        help_text="Original tool data from workflow node"
    )

    # Comparison query and results
    comparison_query = models.TextField(
        help_text="Query used to find alternative tools"
    )
    alternative_tools = models.JSONField(
        help_text="List of alternative tools found", default=list
    )

    # Metadata
    total_alternatives_found = models.IntegerField(default=0)
    search_time_ms = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        """Meta options."""

        db_table = "tool_comparisons"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["workflow", "created_at"]),
        ]

    def __str__(self):
        """String representation."""
        return f"Comparison: {self.original_tool_node_id} - {self.total_alternatives_found} alternatives"


class BackgroundTask(models.Model):
    """Track background tasks for tool discovery and processing."""

    task_id = models.CharField(max_length=100, primary_key=True)
    task_type = models.CharField(
        max_length=50,
        choices=[
            ("discovery", "Tool Discovery"),
            ("scraping", "URL Scraping"),
            ("pinecone_update", "Pinecone Update"),
            ("cache_cleanup", "Cache Cleanup"),
            ("workflow_gen", "Workflow Generation"),
        ],
    )
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending"),
            ("running", "Running"),
            ("completed", "Completed"),
            ("failed", "Failed"),
            ("cancelled", "Cancelled"),
        ],
        default="pending",
    )

    # Task details
    params = models.JSONField(default=dict)
    result = models.JSONField(null=True, blank=True)
    error_message = models.TextField(blank=True)

    # Timing
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)

    class Meta:
        """Meta options."""

        db_table = "background_tasks"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "created_at"]),
            models.Index(fields=["task_type", "status"]),
        ]

    def __str__(self):
        """String representation."""
        return f"{self.task_type} - {self.status}"


class ConversationSession(models.Model):
    """Track conversation sessions and context for conversational AI tool recommender."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="conversation_sessions",
        null=True,
        blank=True,
    )
    session_id = models.UUIDField(default=uuid.uuid4, unique=True)

    # Chat context
    original_query = models.TextField(
        help_text="Initial user query that started the conversation"
    )
    refined_query = models.TextField(
        blank=True, help_text="AI-refined version of the original query"
    )
    chat_history = models.JSONField(
        default=list, help_text="Complete conversation history"
    )
    current_context = models.JSONField(
        default=dict, help_text="Current conversation context and state"
    )

    # Workflow building state
    workflow_nodes = models.JSONField(
        default=list, help_text="Nodes added during conversation"
    )
    workflow_edges = models.JSONField(
        default=list, help_text="Edges/connections between nodes"
    )
    suggested_tools = models.JSONField(
        default=list, help_text="Tools suggested during conversation"
    )

    # Questionnaire state
    workflow_state = models.CharField(
        max_length=30,
        choices=[
            ("initial", "Initial"),
            ("questioning", "Asking Questions"),
            ("generating_workflow", "Generating Workflow"),
            ("workflow_ready", "Workflow Ready"),
            ("error", "Error"),
        ],
        default="initial",
        help_text="Current state of the workflow generation process",
    )
    questionnaire_json = models.JSONField(
        default=dict,
        null=True,
        blank=True,
        help_text="Questionnaire with questions and answers",
    )
    current_question_index = models.IntegerField(
        default=0, help_text="Index of the current question being asked"
    )
    total_questions = models.IntegerField(
        default=0, help_text="Total number of questions in the questionnaire"
    )

    # Session metadata
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Meta options."""

        db_table = "conversation_sessions"
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["session_id"]),
            models.Index(fields=["is_active", "updated_at"]),
        ]

    def __str__(self):
        """String representation."""
        return f"Session: {self.original_query[:50]} - {self.created_at}"


class ChatMessage(models.Model):
    """Individual chat messages with context and intent analysis."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        ConversationSession, on_delete=models.CASCADE, related_name="chat_messages"
    )

    # Message content
    user_message = models.TextField(help_text="User's message")
    ai_response = models.TextField(help_text="AI's response")
    message_type = models.CharField(
        max_length=50,
        choices=[
            ("chat", "General Chat"),
            ("tool_suggestion", "Tool Suggestion"),
            ("workflow_building", "Workflow Building"),
            ("tool_exploration", "Tool Exploration"),
            ("workflow_generation", "Workflow Generation"),
        ],
        default="chat",
        help_text="Type of conversation message",
    )

    # Agent tracking
    agent_name = models.CharField(
        max_length=50,
        choices=[
            ("refine_query_generator", "Refine Query Generator"),
            ("workflow_builder", "Workflow Builder"),
            ("general_assistant", "General Assistant"),
            ("implementation_chat", "Implementation Chat"),
            ("tool_assistant", "Tool Assistant"),
            ("unknown", "Unknown"),
        ],
        default="unknown",
        help_text="Name of the agent that generated this response",
    )
    user_responded_to_agent = models.CharField(
        max_length=50,
        blank=True,
        help_text="Name of the agent the user was responding to",
    )

    # Context and metadata
    tools_mentioned = models.JSONField(
        default=list, help_text="Tools mentioned in this message"
    )
    workflow_changes = models.JSONField(
        default=dict, help_text="Changes made to workflow"
    )
    intent_analysis = models.JSONField(
        default=dict, help_text="AI analysis of user intent"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        """Meta options."""

        db_table = "chat_messages"
        ordering = ["created_at"]
        indexes = [
            models.Index(fields=["session", "created_at"]),
            models.Index(fields=["message_type", "created_at"]),
        ]

    def __str__(self):
        """String representation."""
        return f"Message: {self.user_message[:50]} - {self.message_type}"


class RefinedQuery(models.Model):
    """Store refined queries with workflow information."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    workflow_id = models.UUIDField(db_index=True, help_text="Associated workflow ID")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="refined_queries",
        null=True,
        blank=True,
    )
    session = models.ForeignKey(
        ConversationSession,
        on_delete=models.CASCADE,
        related_name="refined_queries",
        null=True,
        blank=True,
    )

    # Query details
    original_query = models.TextField(help_text="Original user query")
    refined_query = models.TextField(help_text="AI-refined query from Phase 3")

    # Workflow information extracted from questionnaire
    workflow_info = models.JSONField(
        default=dict,
        help_text="Complete workflow information (intent, outcome, tools, etc.)",
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Meta options."""

        db_table = "refined_queries"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["workflow_id"]),
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["session"]),
        ]

    def __str__(self):
        """String representation."""
        return (
            f"Refined Query for Workflow {self.workflow_id}: {self.refined_query[:50]}"
        )


class RefineQuerySession(models.Model):
    """Track query refinement sessions for high-level user requests."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="refine_query_sessions",
        null=True,
        blank=True,
    )
    session_id = models.UUIDField(default=uuid.uuid4, unique=True)

    # Query refinement data
    original_prompt = models.TextField(help_text="User's initial high-level request")
    refined_query = models.TextField(
        blank=True, help_text="Refined query after AI processing"
    )
    refinement_history = models.JSONField(
        default=list, help_text="History of query refinements and AI suggestions"
    )

    # Session status
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending Refinement"),
            ("refining", "Being Refined"),
            ("refined", "Query Refined"),
            ("completed", "Completed"),
            ("cancelled", "Cancelled"),
        ],
        default="pending",
        help_text="Current status of the refinement session",
    )

    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Meta options."""

        db_table = "refine_query_sessions"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["session_id"]),
            models.Index(fields=["status", "created_at"]),
        ]

    def __str__(self):
        """String representation."""
        return f"Refine Session: {self.original_prompt[:50]} - {self.status}"


class WorkflowImplementationGuide(models.Model):
    """Store implementation guides for workflows."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="workflow_implementations",
        null=True,
        blank=True,
    )

    # Workflow reference - can be either WorkflowGeneration or regular Workflow
    workflow_id = models.UUIDField(
        db_index=True,
        help_text="ID of the associated workflow (from either Workflow or WorkflowGeneration model)",
    )

    # Implementation guide content
    implementation_guide = models.TextField(
        help_text="Complete step-by-step implementation guide in Markdown format"
    )

    # Workflow snapshot for reference
    workflow_snapshot = models.JSONField(
        help_text="Snapshot of workflow data when implementation was generated",
        default=dict,
    )

    # Generation metadata
    tools_count = models.IntegerField(
        default=0, help_text="Number of tools in the workflow"
    )
    generation_time_ms = models.FloatField(null=True, blank=True)

    # Status tracking
    status = models.CharField(
        max_length=20,
        choices=[
            ("generating", "Generating"),
            ("completed", "Completed"),
            ("error", "Error"),
        ],
        default="generating",
    )
    error_message = models.TextField(blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        """Meta options."""

        db_table = "workflow_implementation_guides"
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["workflow_id"]),
            models.Index(fields=["user", "created_at"]),
            models.Index(fields=["status", "created_at"]),
        ]
        # Ensure one implementation guide per workflow per user
        unique_together = [["user", "workflow_id"]]

    def __str__(self):
        """String representation."""
        return f"Implementation Guide for Workflow {self.workflow_id} - {self.status}"
