"""CHAT VIEWS."""
import logging

from drf_spectacular.utils import OpenApiParameter, extend_schema, extend_schema_view
from envs.env_loader import env_loader  # Import the env_loader instance
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .models import CHAT_TYPE_CHOICES, ChatMessage, ChatSession
from .serializers import (
    ChatInputSerializer,
    ChatMessageSerializer,
    ChatSessionListSerializer,
    ChatSessionSerializer,
)

logger = logging.getLogger(__name__)

# Use environment variables from env_loader
OPENAI_API_KEY = env_loader.openai_api_key
OPENAI_MODEL = "gpt-4o-mini"  # This will get "text-embedding-3-small" from env file


@extend_schema_view(
    create=extend_schema(
        description="Start a new chat session or send a message in existing session",
        request=ChatInputSerializer,
        responses={200: ChatSessionSerializer},
    ),
    list=extend_schema(
        description="List all chat sessions for the current user",
        request=ChatSessionListSerializer,
        responses={200: ChatSessionSerializer(many=True)},
    ),
)
class ChatViewSet(viewsets.ModelViewSet):
    """ViewSet for handling chat interactions."""

    permission_classes = [IsAuthenticated]
    serializer_class = ChatSessionSerializer

    def get_queryset(self):
        """Return chat sessions for the current user only."""
        workflow_id = self.request.query_params.get("workflow_id")
        chat_type = self.request.query_params.get("chat_type")
        if workflow_id and chat_type:
            return ChatSession.objects.filter(
                user=self.request.user, workflow_id=workflow_id, chat_type=chat_type
            )
        return ChatSession.objects.filter(user=self.request.user)

    def create(self, request):
        """Handle new messages using LangChain and ChatGPT."""
        input_serializer = ChatInputSerializer(data=request.data)
        if not input_serializer.is_valid():
            return Response(input_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        try:
            session_id = input_serializer.validated_data.get("session_id")
            workflow_data = input_serializer.validated_data["workflow"]
            message = input_serializer.validated_data["message"]
            workflow_id = input_serializer.validated_data.get("workflow_id")
            chat_type = input_serializer.validated_data.get("chat_type", "general")

            # Process workflow data to create
            # context
            workflow_context = self._create_workflow_context(workflow_data)

            # Get or create chat session
            if session_id:
                session = ChatSession.objects.get(
                    id=session_id,
                    user=request.user,
                    workflow_id=workflow_id,
                    chat_type=chat_type,
                )
            else:
                # check if session already exists with workflow_id and chat_type
                session = ChatSession.objects.filter(
                    user=request.user,
                    workflow_id=workflow_id,
                    chat_type=chat_type,
                ).first()
                if not session:
                    session = ChatSession.objects.create(
                        user=request.user,
                        workflow_data=workflow_data,
                        workflow_id=workflow_id,
                        chat_type=chat_type,
                    )

            # Store user message
            ChatMessage.objects.create(session=session, role="user", content=message)

            # Get chat history for context
            history = ChatMessage.objects.filter(session=session).order_by("timestamp")

            # Convert messages to LangChain format
            messages = []
            # Add workflow context and user's original query as system message
            system_message = (
                f"{workflow_context}\n\n"
                f"User's Current Query: {message}\n\n"
                f"Please analyze this workflow and provide a detailed explanation that "
                f"addresses the user's query while considering the workflow structure."
            )
            messages.append(SystemMessage(content=system_message))

            # Add chat history
            for msg in history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))

            # Initialize ChatGPT model with environment variables
            chat = ChatOpenAI(
                temperature=0.7, model_name=OPENAI_MODEL, openai_api_key=OPENAI_API_KEY
            )

            # Get response from ChatGPT
            try:
                ai_response = chat(messages)
                ai_message = ai_response.content
            except Exception as e:
                logger.error("ChatGPT API error: %s", str(e))
                return Response(
                    {"error": "Failed to get response from ChatGPT"},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

            # Store AI response
            ai_message_obj = ChatMessage.objects.create(
                session=session, role="assistant", content=ai_message
            )

            # Update session timestamp
            session.save()

            # Return only the AI message instead of the whole session
            return Response(
                {
                    "role": "assistant",
                    "content": ai_message,
                    "timestamp": ai_message_obj.timestamp,
                    "session_id": session.id,
                }
            )

        except Exception as e:
            logger.error("Chat error: %s", str(e))
            return Response(
                {"error": "An error occurred while processing your message"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _create_workflow_context(self, workflow_data):
        """Create context string from workflow data."""
        nodes = workflow_data.get("nodes", [])
        edges = workflow_data.get("edges", [])

        # Extract relevant information from nodes
        nodes_info = []
        for node in nodes:
            node_data = node.get("data", {})
            node_info = (
                f"Node {node['id']}:\n"
                f"Type: {node['type']}\n"
                f"Label: {node_data.get('label', 'N/A')}\n"
                f"Description: {node_data.get('description', 'N/A')}\n"
            )
            if node_data.get("features"):
                node_info += f"Features: {', '.join(node_data['features'])}\n"
            if node_data.get("tags"):
                node_info += f"Tags: {', '.join(node_data['tags'])}\n"
            nodes_info.append(node_info)

        # Create workflow context with enhanced AI instructions
        workflow_context = (
            f"You are an AI assistant specialized in explaining workflows and "
            f"automation processes. The following workflow was generated based on "
            f"a user's request.\n\n"
            f"Workflow Structure:\n"
            f"Total Nodes: {len(nodes)}\n"
            f"Total Connections: {len(edges)}\n"
            f"\nDetailed Node Information:\n"
            f"{''.join(nodes_info)}\n"
            f"\nYour Role:\n"
            f"1. Understand that this workflow was created in response to the "
            f"user's message/query\n"
            f"2. Analyze the workflow structure and explain how it addresses "
            f"the user's needs\n"
            f"3. Explain the flow of automation, how nodes connect, and what "
            f"each component does\n"
            f"4. Provide practical insights about how this workflow would "
            f"function in real-world scenarios\n"
            f"5. If asked about specific parts of the workflow, focus on those "
            f"components while maintaining context\n"
            f"\nWhen responding:\n"
            f"- Be clear and concise in your explanations\n"
            f"- Connect your explanations to the user's original intent\n"
            f"- Highlight the practical benefits and functionality of the workflow\n"
            f"- If you notice any potential improvements or considerations, "
            f"mention them\n"
        )

        return workflow_context

    @action(detail=True, methods=["delete"])
    def clear_history(self, request, pk=None):
        """Clear chat history for a specific session."""
        try:
            session = self.get_object()
            session.messages.all().delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except Exception as e:
            logger.error("Error clearing chat history: %s", str(e))
            return Response(
                {"error": "Failed to clear chat history"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @extend_schema(
        summary="Get Chat Messages by Workflow ID and Chat Type",
        parameters=[
            OpenApiParameter(
                name="workflow_id",
                description="ID of the workflow",
                required=True,
                type=str,
            ),
            OpenApiParameter(
                name="chat_type",
                description="Type of chat (workflow or general)",
                required=True,
                type=str,
                enum=[choice[0] for choice in CHAT_TYPE_CHOICES],
            ),
        ],
        responses={200: ChatMessageSerializer(many=True)},
    )
    @action(detail=False, methods=["get"], url_path="by-workflow")
    def get_by_workflow(self, request):
        """Get chat messages by workflow_id and chat_type."""
        workflow_id = request.query_params.get("workflow_id")
        chat_type = request.query_params.get("chat_type")

        if not workflow_id or not chat_type:
            return Response(
                {"error": "Both workflow_id and chat_type are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate chat_type against allowed values
        valid_chat_types = [choice[0] for choice in CHAT_TYPE_CHOICES]
        if chat_type not in valid_chat_types:
            return Response(
                {
                    "error": "Invalid chat_type. Must be one of: "
                    + ", ".join(valid_chat_types)
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Get the chat session first
            chat_session = ChatSession.objects.get(
                user=request.user,
                workflow_id=workflow_id,
                chat_type=chat_type,
            )
            # Get all messages for this session
            messages = ChatMessage.objects.filter(session=chat_session).order_by(
                "timestamp"
            )
            serializer = ChatMessageSerializer(messages, many=True)
            return Response(serializer.data)
        except ChatSession.DoesNotExist:
            return Response(
                {"error": "Chat session not found"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            logger.error("Error retrieving chat messages: %s", str(e))
            return Response(
                {"error": "An error occurred while retrieving the chat messages"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
