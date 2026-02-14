"""Views for the contractor app."""

import json

import numpy as np
from django.shortcuts import get_object_or_404
from drf_spectacular.utils import OpenApiResponse, extend_schema
from envs.env_loader import env_loader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from authentication.models import CustomUser
from management.configurations.utils import send_html_email
from management.email_templates.meeting_booking_template import meeting_booking_template

from .serializers import ConsultantSearchSerializer, MeetingBookingSerializer
from .utils.database import pinecone_manager
from .utils.embeddings import get_embedding


class ConsultantSearchViewSet(viewsets.ViewSet):
    """ViewSet for searching for consultants."""

    permission_classes = [IsAuthenticated]
    serializer_class = ConsultantSearchSerializer

    CONSULTANT_JSON_TEMPLATE = """{{
        'name': match.metadata.get('name'),
        'expertise': match.metadata.get('expertise'),
        'experience': match.metadata.get('experience'),
        'website': match.metadata.get('website'),
        'phone': match.metadata.get('phone'),
        'gmail': match.metadata.get('gmail'),
        'apps_included': match.metadata.get('apps_included'),
        'language': match.metadata.get('language'),
        'country': match.metadata.get('country'),
        'company_name': match.metadata.get('company_name'),
        'type_of_services': match.metadata.get('type_of_services'),
        'countries_with_office_locations': match.metadata.get(
            'countries_with_office_locations'
        ),
        'about': match.metadata.get('about'),
        'date': match.metadata.get('date'),
        'time': match.metadata.get('time')
    }}"""

    @extend_schema(
        parameters=[ConsultantSearchSerializer],
        description="Search for consultants based on query and user work description",
        responses={
            200: OpenApiResponse(
                description="Successfully retrieved consultant recommendations",
            ),
            400: OpenApiResponse(
                description="Invalid parameters",
                response={
                    "type": "object",
                    "properties": {"error": {"type": "string"}},
                },
            ),
            404: OpenApiResponse(
                description="No consultants found",
                response={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                },
            ),
        },
        tags=["consultant-search"],
    )
    @action(detail=False, methods=["get"])
    def search(self, request):
        """Search for consultants based on a query and user work description."""
        serializer = self.serializer_class(data=request.query_params)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_data = serializer.validated_data
        query = validated_data["query"]
        user_work_description = validated_data["user_work_description"]

        # Generate embedding for the search query
        query_embedding = get_embedding(query)
        query_embedding = np.array(query_embedding, dtype=np.float32).tolist()

        # Get Pinecone index and search
        db = pinecone_manager.get_index()
        response = db.query(
            vector=query_embedding,
            top_k=10,
            namespace="consultants",
            include_values=True,
            include_metadata=True,
        )
        if not response.matches:
            return Response(
                {"message": "No consultants found."}, status=status.HTTP_404_NOT_FOUND
            )

        # Prepare prompt for consultant filtering
        prompt_text = (
            "You are the best consultant recommender. "
            "If Need translate in english Experience in english. "
            "The best choice depends on the specific needs of "
            "the business seeking consulting services. "
            "Given the following consultants with their "
            "work descriptions, select only the best "
            "ones based on expertise and experience. "
            "Return only the selected consultants as "
            "a JSON array, with each consultant as a JSON object "
            f"in the format:\n\n{self.CONSULTANT_JSON_TEMPLATE}\n\n"
            "Do not include any additional text or description. "
            "Consultants: {consultants}. "
            "User's work description: {user_work_description}. "
            "User's tools: {query}"
        )

        prompt_template = ChatPromptTemplate.from_template(prompt_text)
        consultants = [match.metadata for match in response.matches]

        llm = ChatOpenAI(
            api_key=env_loader.openai_api_key,
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=4000,
        )
        chain = prompt_template | llm

        best_consultants = chain.invoke(
            {
                "consultants": consultants,
                "user_work_description": user_work_description,
                "query": query,
            }
        )

        try:
            response_content = (
                best_consultants.content.replace("```json", "")
                .replace("```", "")
                .strip()
            )
            best_consultants_list = json.loads(response_content)
        except json.JSONDecodeError as e:
            print(e)
            return Response(
                {"error": "Failed to parse consultant recommendations."},
                status=status.HTTP_404_NOT_FOUND,
            )

        return Response(
            {
                "user": request.user.username,
                "best_consultants": best_consultants_list,
            }
        )

    def get_contractor_data(self, contractor_id):
        """Get contractor data from either Pinecone or database."""
        # First try Pinecone
        db = pinecone_manager.get_index()
        response = db.fetch(
            ids=[contractor_id],
            namespace="consultants",
        )

        if response.vectors:
            return response.vectors[contractor_id].metadata, "pinecone"

        # If not found in Pinecone, try database
        try:
            contractor = get_object_or_404(
                CustomUser, unique_id=contractor_id, role="contractor"
            )
            # Convert database model to dictionary format matching Pinecone structure
            contractor_data = {
                "name": f"{contractor.first_name} {contractor.last_name}".strip()
                or contractor.username,
                "gmail": contractor.email,
                "company_name": contractor.company_name,
                "expertise": contractor.expertise,
                "experience": contractor.experience,
                "website": contractor.website,
                "phone": contractor.phone,
                "apps_included": contractor.apps_included,
                "language": contractor.language,
                "country": contractor.country,
                "type_of_services": contractor.type_of_services,
                "countries_with_office_locations": contractor.countries_with_office_locations,
                "about": contractor.about,
                "date": contractor.availability_date.strftime("%Y-%m-%d")
                if contractor.availability_date
                else None,
                "time": contractor.availability_time.strftime("%H:%M")
                if contractor.availability_time
                else None,
            }
            return contractor_data, "db"
        except CustomUser.DoesNotExist:
            return None, None

    @extend_schema(
        request=MeetingBookingSerializer,
        description="Book a meeting with a contractor",
        responses={
            200: OpenApiResponse(
                description="Meeting request sent successfully",
            ),
            400: OpenApiResponse(
                description="Invalid parameters",
            ),
            404: OpenApiResponse(
                description="Contractor not found",
            ),
        },
        tags=["consultant-meetings"],
    )
    @action(detail=False, methods=["post"])
    def book_meeting(self, request):
        """Send a meeting booking request to a contractor."""
        serializer = MeetingBookingSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        validated_data = serializer.validated_data
        contractor_id = validated_data["contractor_id"]

        # Get contractor details from either Pinecone or database
        contractor_data, source = self.get_contractor_data(contractor_id)

        if not contractor_data:
            return Response(
                {"message": "Contractor not found in either Pinecone or database."},
                status=status.HTTP_404_NOT_FOUND,
            )

        contractor_email = contractor_data.get("gmail")  # For Pinecone format
        if not contractor_email and source == "db":
            contractor_email = contractor_data.get("email")  # For database format

        if not contractor_email:
            return Response(
                {"message": "Contractor email not found."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Prepare email content
        email_context = {
            "contractor_name": contractor_data.get("name", "Contractor"),
            "client_name": validated_data["client_name"],
            "client_email": validated_data["client_email"],
            "company_name": validated_data.get("company_name", ""),
            "project_description": validated_data["project_description"],
            "preferred_date": validated_data["preferred_date"].strftime("%B %d, %Y"),
            "preferred_time": validated_data["preferred_time"].strftime("%I:%M %p"),
            "accept_meeting_link": "https://app.example.com",
        }

        # Format email template with context
        html_content = meeting_booking_template.format(**email_context)

        # Send email
        success, error = send_html_email(
            recipient_email=contractor_email,
            subject=f"New Meeting Request from {validated_data['client_name']}",
            html_body=html_content,
            text_body=f"You have received a new meeting request from {validated_data['client_name']}. Please check your email in an HTML-capable email client.",
        )

        if not success:
            return Response(
                {"message": f"Failed to send meeting request: {error}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response(
            {
                "message": "Meeting request sent successfully",
                "contractor_id": contractor_id,
                "source": source,  # Optionally include where the contractor data was found
            }
        )
