"""Views for tool scraping API endpoints."""

import logging

import pandas as pd
from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response

from .models import CombinedText, ScrapingJob, SitePage, ToolSite
from .serializers import (
    BulkInternetToolsSerializer,
    CombinedTextSerializer,
    ScrapingJobSerializer,
    SitePageSerializer,
    ToolSiteSerializer,
)
from .tasks import process_csv_rows, process_internet_discovered_tools

logger = logging.getLogger(__name__)


class ScrapingJobViewSet(viewsets.ModelViewSet):
    """ViewSet for managing scraping jobs."""

    queryset = ScrapingJob.objects.all()
    serializer_class = ScrapingJobSerializer
    parser_classes = (MultiPartParser, FormParser)

    @action(detail=False, methods=["post"], url_path="upload-csv")
    def upload_csv(self, request):
        """
        POST /api/v1/tool-scraping/jobs/upload-csv/.

        Upload a CSV file to start a scraping job.
        Validates CSV headers synchronously before accepting.
        Returns 202 Accepted with job ID.
        """
        if "file" not in request.data:
            logger.error("No file provided in upload request")
            return Response(
                {"error": "No file provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        file_obj = request.data["file"]

        # Validate file extension
        if not file_obj.name.endswith(".csv"):
            logger.error(f"Invalid file type: {file_obj.name}")
            return Response(
                {"error": "File must be a CSV"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Synchronous CSV header validation
        try:
            # Read only headers to validate
            df = pd.read_csv(file_obj, nrows=0)

            required_columns = [
                "Embedding",
                "Nr",
                "Title",
                "Description",
                "Website",
                "User Case Suggestion",
                "Category",
                "Features",
                "Master Category",
                "Twitter",
                "Linkedin",
                "TikTok",
                "Youtube",
                "Instagram",
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required CSV columns: {missing_columns}")
                return Response(
                    {
                        "error": f'Missing required columns: {", ".join(missing_columns)}'
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # Reset file pointer for saving
            file_obj.seek(0)

        except Exception as e:
            logger.exception(f"Failed to read CSV: {e}")
            return Response(
                {"error": f"Failed to read CSV: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Create job - manually set defaults to avoid serializer issues
        job_data = {
            "file": file_obj,
            "job_type": ScrapingJob.JobType.CSV_UPLOAD,
            "payload": {},
            "status": ScrapingJob.Status.PENDING,
        }

        job = ScrapingJob.objects.create(**job_data)
        logger.info(f"Created scraping job {job.id}")

        # Trigger background processing
        process_csv_rows.delay(str(job.id))
        logger.info(f"Triggered process_csv_rows task for job {job.id}")

        # Return serialized response
        serializer = self.get_serializer(job)
        return Response(serializer.data, status=status.HTTP_202_ACCEPTED)

    @action(detail=False, methods=["post"], url_path="submit-internet-tools")
    def submit_internet_tools(self, request):
        """
        POST /api/v1/tool-scraping/jobs/submit-internet-tools/.

        Submit tools discovered via internet search for background scraping and indexing.
        Accepts a list of tools with Title, Description, and Website.
        Returns 202 Accepted with job ID.
        """
        serializer = BulkInternetToolsSerializer(data=request.data)
        if not serializer.is_valid():
            logger.error(f"Invalid internet tools data: {serializer.errors}")
            return Response(
                {"error": "Invalid data", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST,
            )

        tools_data = serializer.validated_data["tools"]
        source_query = serializer.validated_data.get("source_query", "")

        if not tools_data:
            return Response(
                {"error": "No tools provided"}, status=status.HTTP_400_BAD_REQUEST
            )

        # Create job for internet discovery
        job_data = {
            "job_type": ScrapingJob.JobType.INTERNET_DISCOVERY,
            "payload": {
                "source_query": source_query,
                "tools_count": len(tools_data),
                "tools": tools_data,  # Store original tool data
            },
            "status": ScrapingJob.Status.PENDING,
        }

        job = ScrapingJob.objects.create(**job_data)
        logger.info(
            f"Created internet discovery job {job.id} with {len(tools_data)} tools"
        )

        # Trigger background processing
        process_internet_discovered_tools.delay(str(job.id))
        logger.info(
            f"Triggered process_internet_discovered_tools task for job {job.id}"
        )

        # Return serialized response
        return Response(
            {
                "id": str(job.id),
                "job_type": job.job_type,
                "status": job.status,
                "tools_count": len(tools_data),
                "message": f"Accepted {len(tools_data)} tools for background scraping and indexing",
            },
            status=status.HTTP_202_ACCEPTED,
        )

    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        """
        GET /api/v1/tool-scraping/jobs/{job_id}/status/.

        Get the status and logs of a specific job.
        """
        job = self.get_object()

        # Count related objects
        sites_count = job.sites.count()
        pages_count = SitePage.objects.filter(site__job=job).count()
        combined_texts_count = CombinedText.objects.filter(site__job=job).count()

        return Response(
            {
                "id": job.id,
                "job_type": job.job_type,
                "status": job.status,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "created_at": job.created_at,
                "updated_at": job.updated_at,
                "logs": job.logs,
                "stats": {
                    "sites": sites_count,
                    "pages": pages_count,
                    "combined_texts": combined_texts_count,
                },
            }
        )


class ToolSiteViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only viewset for ToolSite."""

    queryset = ToolSite.objects.all()
    serializer_class = ToolSiteSerializer


class SitePageViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only viewset for SitePage."""

    queryset = SitePage.objects.all()
    serializer_class = SitePageSerializer


class CombinedTextViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only viewset for CombinedText."""

    queryset = CombinedText.objects.all()
    serializer_class = CombinedTextSerializer
