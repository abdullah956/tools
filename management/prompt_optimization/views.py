"""Views for managing prompt optimization."""

from drf_spectacular.utils import OpenApiParameter, extend_schema, extend_schema_view
from envs.env_loader import env_loader  # Import the env_loader instance
from langchain_openai import ChatOpenAI
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from configurations.pagination import CustomPagination

from .models import UserQuery
from .serializers import UserQuerySerializer

print("env_loader.openai_api_key", env_loader.openai_api_key)
# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=env_loader.openai_api_key)


@extend_schema_view(
    list=extend_schema(
        summary="Retrieve a list of all user queries",
        responses={200: UserQuerySerializer(many=True)},
    ),
    create=extend_schema(
        summary="Refine a user query and save it",
        request=UserQuerySerializer,
        responses={201: UserQuerySerializer},
    ),
    retrieve=extend_schema(
        summary="Retrieve a specific user query",
        responses={200: UserQuerySerializer},
    ),
    destroy=extend_schema(
        summary="Delete a user query",
        responses={204: None},
    ),
)
class UserQueryViewSet(viewsets.ModelViewSet):
    """A viewset for viewing and refining user queries."""

    serializer_class = UserQuerySerializer
    permission_classes = [IsAuthenticated]
    pagination_class = CustomPagination

    def get_queryset(self):
        """Return queries for the current user only."""
        return UserQuery.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        """Refine the user query and save it."""
        original_query = serializer.validated_data["original_query"]
        workflow = serializer.validated_data.get("workflow")

        # Check if workflow exists and belongs to the current user
        if workflow and workflow.owner != self.request.user:
            raise PermissionDenied(
                "The specified workflow does not exist or does not belong to you."
            )

        # Get the user's unique ID
        user_unique_id = self.request.user.unique_id

        # Define the prompt template for AI tool discovery
        prompt_template = """
        You are an intelligent AI tool discovery assistant.
        Your task is to refine the user's query into a comprehensive, structured format that clearly explains how the solution works, its key features, and step-by-step implementation.

        ### User Information:
        User ID: {user_unique_id}

        ### Format Requirements:
        Transform the user query into a detailed project description with the following structure:

        1. **Project Title**: A clear, descriptive title
        2. **How it works**: Brief explanation of the solution
        3. **Key Features**: Bullet points of main capabilities
        4. **Step-by-step**: Detailed implementation process
        5. **API Keys Required**: List of necessary integrations
        6. **Setup Instructions**: Configuration steps

        ### Example:
        **User Query:**
        "I need to automate my marketing. What are the best apps for me?"

        **Refined Query:**
        "Personal Life Manager with Telegram, Google Services & Voice-Enabled AI

        How it works:
        This project teaches you to create a personal AI assistant named Caylee that operates through Telegram. Caylee can summarize unread emails, check calendar events, manage Google Tasks, and handle both voice and text interactions. The assistant provides a comprehensive digital life management solution accessible via Telegram messaging.

        Key Features:
        • Supports hands-free voice interaction
        • Maintains conversation memory
        • Integrates with major Google services
        • Provides personalized assistance for email management, scheduling, and task organization

        Step-by-step:
        Telegram Trigger:
        The workflow starts with a Telegram trigger that listens for incoming message events. The system determines if the incoming message is voice or text input.

        Voice Processing:
        If a voice message is received, the workflow retrieves the voice file from Telegram and uses OpenAI's transcription API to convert speech to text.

        AI Assistant: The processed text (whether original text or transcribed voice) is passed to Caylee, the AI assistant powered by OpenRouter's language model.

        Tools Integration:
        Caylee is equipped with several productivity tools:
        • Get Email: Uses Gmail API to fetch unread emails from the inbox with sender, date, subject, and summary information
        • Google Calendar: Retrieves calendar events for specified dates, filtering out irrelevant future events
        • Google Tasks: Both creates new tasks and retrieves existing tasks from Google Tasks lists

        API Keys Required:
        • Telegram Bot API: Create a bot via @BotFather on Telegram to get your bot token
        • OpenAI API: Required for voice-to-text transcription
        • OpenRouter API: Powers the AI language model responses
        • Google OAuth2: Needed for Gmail, Google Calendar, and Google Tasks integration

        Response Generation:
        The AI formulates intelligent responses based on the gathered information, current date context, and conversation history, then sends the response back to the user via Telegram in Markdown format."

        Now, refine the following user query into a well-structured, comprehensive project description:
        **User Query:** "{user_query}"
        """

        # Format the prompt with user unique ID
        formatted_prompt = prompt_template.format(
            user_query=original_query, user_unique_id=user_unique_id
        )

        # Use the invoke method and ensure the input is a list of messages
        response = llm.invoke([{"role": "user", "content": formatted_prompt}])
        refined_query = response.content

        # Save the original and refined query along with the current user
        serializer.save(refined_query=refined_query, user=self.request.user)

    @extend_schema(
        summary="Get Chat History for a Specific Workflow",
        parameters=[
            OpenApiParameter(
                name="workflow_id",
                type=str,
                location=OpenApiParameter.QUERY,
                required=True,
                description="The ID of the workflow to retrieve chat history for.",
            )
        ],
        responses={200: UserQuerySerializer(many=True)},
    )
    @action(detail=False, methods=["get"], url_path="chat-history")
    def chat_history(self, request):
        """Retrieve chat history for a specific workflow."""
        user = request.user
        workflow_id = request.query_params.get("workflow_id")

        if not workflow_id:
            return Response(
                {"error": "workflow_id is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Filter chat history based on user and workflow_id
        chat_history = UserQuery.objects.filter(
            user=user, workflow_id=workflow_id
        ).order_by("created_at")

        # Serialize and return the data
        serializer = UserQuerySerializer(chat_history, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    @extend_schema(
        summary="Refine Query and Search AI Tools",
        request=UserQuerySerializer,
        responses={200: UserQuerySerializer},
    )
    @action(detail=False, methods=["post"], url_path="refine-and-search")
    def refine_and_search(self, request):
        """Refine a user query and search for AI tools using the AI Tool Recommender."""
        try:
            original_query = request.data.get("original_query")
            if not original_query:
                return Response(
                    {"error": "original_query is required"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Get the user's unique ID
            user_unique_id = request.user.unique_id

            # Define the prompt template for AI tool discovery
            prompt_template = """
            You are an intelligent AI tool discovery assistant.
            Your task is to refine the user's query into a comprehensive, structured format that clearly explains how the solution works, its key features, and step-by-step implementation.

            ### User Information:
            User ID: {user_unique_id}

            ### Format Requirements:
            Transform the user query into a detailed project description with the following structure:

            1. **Project Title**: A clear, descriptive title
            2. **How it works**: Brief explanation of the solution
            3. **Key Features**: Bullet points of main capabilities
            4. **Step-by-step**: Detailed implementation process
            5. **API Keys Required**: List of necessary integrations
            6. **Setup Instructions**: Configuration steps

            ### Example:
            **User Query:**
            "I need to automate my marketing. What are the best apps for me?"

            **Refined Query:**
            "AI Research Agents to Automate PDF Analysis with Mistral's Best-in-Class OCR

            Overview

            Mistral OCR is a cutting-edge document understanding API that improves how businesses extract and process information from complex documents. With top scores in benchmarks for accuracy and comprehension capabilities, Mistral OCR handles multi-column text, charts, diagrams, and multiple languages.

            This workflow uses Mistral's Document understanding OCR API to automatically turns dense PDFs (such as financial reports) into either deep research reports or concise newsletters

            Key Features

            • Superior Document Understanding: Processes complex documents with high-fidelity rendering
            • Multi-Format Support: Handles PDFs containing text, images, charts, and diagrams
            • Multilingual Capabilities: Accurately processes documents in various languages
            • Seamless API Integration: Easy implementation through cloud-based API
            • Customizable Research Depth: Generate comprehensive 8-page reports or concise 1,750-word newsletters

            How It Works

            • Document Upload: Submit your PDF through an n8n form interface.
            • Output Format Selection: Choose between comprehensive deep research (3,500 words) or Concise newsletter (1,750 words)
            • Custom Instructions: Tailor the analysis by adding specific focus areas (e.g., quantitative data, growth catalysts).
            • AI Processing: The document undergoes multi-stage AI analysis: OCR and text extraction using Mistral AI and Content structuring and summarization using GPT models

            Agents:

            • Research Leader: Plans and conducts initial research, creating a table of contents.
            • Project Planner: Breaks down the table of contents into manageable sections.
            • Research Assistants: Multiple agents that conduct in-depth research on assigned sections.
            • Editor: Compiles and refines the final article, ensuring coherence and proper citations.

            Setup

            • API Key Acquisition:
              - Obtain an API key from OpenRouter.ai
              - Get an API key from Mistral.ai
            • n8n Configuration:
              - In your n8n instance, navigate to the credentials section.
              - Create new credentials for OpenRouter and Mistral, inputting the respective API keys.
            • Form Configuration:
              - Customize the input form fields if needed (e.g., adding company-specific options).
            • Output Customization: Adjust the word count parameters in the Project Planner node to change output length."

            Now, refine the following user query into a well-structured, comprehensive project description:
            **User Query:** "{user_query}"
            """

            # Format the prompt with user unique ID
            formatted_prompt = prompt_template.format(
                user_query=original_query, user_unique_id=user_unique_id
            )

            # Use the invoke method and ensure the input is a list of messages
            response = llm.invoke([{"role": "user", "content": formatted_prompt}])
            refined_query = response.content

            # Save the original and refined query
            user_query = UserQuery.objects.create(
                original_query=original_query,
                refined_query=refined_query,
                user=request.user,
            )

            # Now search for AI tools using the refined query
            import os

            import requests

            # Get the AI Tool Recommender service URL
            ai_tool_service_url = os.getenv(
                "AI_TOOL_RECOMMENDER_URL", "http://localhost:12000"
            )

            search_endpoint = f"{ai_tool_service_url}/search_tools/"

            # Make request to AI Tool Recommender
            search_payload = {
                "query": refined_query,
                "max_results": 10,
                "include_pinecone": True,
                "include_internet": True,
            }

            try:
                search_response = requests.post(
                    search_endpoint, json=search_payload, timeout=30
                )
                search_response.raise_for_status()
                ai_tools_data = search_response.json()
            except requests.RequestException as e:
                ai_tools_data = {
                    "status": "error",
                    "message": f"Failed to search AI tools: {str(e)}",
                    "tools": [],
                }

            # Serialize the user query
            serializer = UserQuerySerializer(user_query)

            # Return both the refined query and AI tools search results
            return Response(
                {
                    "refined_query": serializer.data,
                    "ai_tools_search": ai_tools_data,
                    "message": "Query refined and AI tools searched successfully",
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {"error": f"An error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
