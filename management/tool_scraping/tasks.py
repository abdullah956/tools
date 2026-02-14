"""Celery tasks for tool scraping pipeline."""

import logging
import os

import pandas as pd
from celery import chain, shared_task
from django.db import transaction
from django.utils import timezone

from .models import CombinedText, ScrapingJob, SitePage, ToolSite
from .services.llm import LLMService
from .services.pinecone_service import PineconeService
from .services.scrapers import ApifyService, HTMLCleaner

logger = logging.getLogger(__name__)


def _check_job_completion(job):
    """Check if all sites in a job are complete and mark job as done if so."""
    total_sites = job.sites.count()
    indexed_sites = job.sites.filter(status=ToolSite.Status.INDEXED).count()
    failed_sites = job.sites.filter(status=ToolSite.Status.FAILED).count()
    completed_sites = indexed_sites + failed_sites

    # Job is complete when all sites are either INDEXED or FAILED
    if completed_sites == total_sites:
        logger.info(
            f"All sites processed for job {job.id} "
            f"({indexed_sites} indexed, {failed_sites} failed), marking job complete"
        )
        job.status = ScrapingJob.Status.COMPLETED
        job.finished_at = timezone.now()
        job.logs.append(
            f"Job complete: {indexed_sites} sites indexed, "
            f"{failed_sites} sites failed out of {total_sites} total."
        )
        job.save()

        # Delete the uploaded CSV file after processing is complete
        if job.file:
            try:
                file_path = job.file.path
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Deleted CSV file: {file_path}")
                    job.logs.append(f"CSV file deleted: {job.file.name}")
                # Clear the file field in database
                job.file.delete(save=False)
                job.save()
                logger.info(f"Cleared file reference for job {job.id}")
            except Exception as e:
                logger.exception(f"Failed to delete CSV file for job {job.id}: {e}")
                job.logs.append(f"Warning: Could not delete CSV file: {str(e)}")
                job.save()


def _continue_to_next_site_if_sequential(site):
    """
    Helper function to continue to next site in sequential processing if this site failed.

    This ensures that when a site fails at any stage (sitemap, scraping, aggregation, etc.),
    the worker continues to process the next site instead of stopping.

    SAFETY: This function is designed to NEVER crash. All exceptions are caught and logged.

    Args:
        site: ToolSite instance that failed
    """
    try:
        # Safely get job - if this fails, we can't continue anyway
        try:
            job = site.job
        except Exception as e:
            logger.warning(
                f"Could not get job for site {site.id} in _continue_to_next_site_if_sequential: {e}"
            )
            return  # Can't continue without job info

        # Safely get worker tracking info
        try:
            worker_site_ids = (
                site.fields.get("_worker_site_ids") if site.fields else None
            )
            worker_site_index = (
                site.fields.get("_worker_site_index", -1) if site.fields else -1
            )
        except Exception as e:
            logger.warning(
                f"Could not get worker tracking info for site {site.id}: {e}"
            )
            return  # Can't continue without worker info

        if worker_site_ids and worker_site_index >= 0:
            # This is a sequential job - trigger next site for this worker
            next_index = worker_site_index + 1
            if next_index < len(worker_site_ids):
                logger.info(
                    f"⚠️ Site {site.id} failed, worker continuing to next site "
                    f"({next_index + 1}/{len(worker_site_ids)})"
                )
                # Process next site for this worker
                # Wrap in try-except to prevent Celery task scheduling errors from crashing
                try:
                    process_site_sequentially.delay(job.id, worker_site_ids, next_index)
                except Exception as celery_error:
                    logger.error(
                        f"Failed to schedule next site task for site {site.id}: {celery_error}. "
                        f"This is non-fatal - the job will continue when manually triggered."
                    )
            else:
                logger.info(
                    f"⚠️ Site {site.id} failed, but worker finished all {len(worker_site_ids)} assigned sites"
                )
                # This worker is done - check if all workers are done
                try:
                    _check_job_completion(job)
                except Exception as completion_error:
                    logger.error(
                        f"Error checking job completion for job {job.id}: {completion_error}"
                    )
    except Exception as e:
        # Final safety net - catch ANY exception and just log it
        # This ensures the function NEVER crashes the calling code
        logger.exception(
            f"Unexpected error in _continue_to_next_site_if_sequential for site {getattr(site, 'id', 'unknown')}: {e}"
        )


@shared_task(bind=True, max_retries=3)
def process_internet_discovered_tools(self, job_id):
    """
    Process tools discovered via internet search.

    Similar to process_csv_rows but for internet-discovered tools.
    Creates ToolSite records and triggers the same scraping pipeline.

    Args:
        self: Celery task instance (bound task)
        job_id: UUID of the ScrapingJob
    """
    try:
        job = ScrapingJob.objects.get(id=job_id)
        job.status = ScrapingJob.Status.PROCESSING
        job.started_at = timezone.now()
        job.save()

        logger.info(f"🌐 Starting internet discovery processing for job {job_id}")

        # Get tools from payload
        tools_data = job.payload.get("tools", [])
        source_query = job.payload.get("source_query", "")

        if not tools_data:
            logger.error(f"No tools data found in job {job_id} payload")
            job.logs.append("Error: No tools data in payload")
            job.status = ScrapingJob.Status.FAILED
            job.finished_at = timezone.now()
            job.save()
            return

        logger.info(f"Processing {len(tools_data)} internet-discovered tools")

        # Process each tool and create ToolSite records
        sites_created = 0
        invalid_urls = 0
        duplicate_urls = 0
        site_ids = []  # Store site IDs for sequential processing

        for index, tool_data in enumerate(tools_data):
            # Handle both lowercase and capitalized field names
            url = tool_data.get("website") or tool_data.get("Website", "")
            title = tool_data.get("title") or tool_data.get("Title", "")
            description = tool_data.get("description") or tool_data.get(
                "Description", ""
            )

            # Skip if URL is empty
            if not url or not url.strip():
                logger.warning(f"Tool {index}: Empty URL")
                invalid_urls += 1
                continue

            url = url.strip()

            # Normalize URL
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "https://" + url
                logger.debug(f"Tool {index}: Added https:// to {url}")

            # Remove query parameters and fragments
            if "?" in url:
                url = url.split("?")[0]
            if "#" in url:
                url = url.split("#")[0]

            # Validate domain
            try:
                from urllib.parse import urlparse

                parsed = urlparse(url)
                if not parsed.netloc or "." not in parsed.netloc:
                    raise ValueError("Invalid domain")
            except Exception as e:
                logger.warning(f"Tool {index}: Invalid URL {url} - {e}")
                job.logs.append(f"Tool {index} ({title}): Invalid URL {url}")
                invalid_urls += 1
                continue

            # Check for duplicates in this job
            if ToolSite.objects.filter(job=job, website=url).exists():
                logger.info(f"Tool {index}: Duplicate URL {url}, skipping")
                duplicate_urls += 1
                continue

            # Create ToolSite
            try:
                site = ToolSite.objects.create(
                    job=job,
                    csv_row_nr=index,  # Use index as row number
                    website=url,
                    title=title,
                    description=description,
                    category=tool_data.get("category") or tool_data.get("Category", ""),
                    master_category=tool_data.get("master_category")
                    or tool_data.get("Master Category", ""),
                    status=ToolSite.Status.PENDING,
                    # Store additional fields in JSONB
                    fields={
                        "source": tool_data.get("source")
                        or tool_data.get("Source", "Internet Search"),
                        "relevance_score": tool_data.get("relevance_score")
                        or tool_data.get("Relevance Score", 0),
                        "features": tool_data.get("features")
                        or tool_data.get("Features", ""),
                        "twitter": tool_data.get("twitter")
                        or tool_data.get("Twitter", ""),
                        "facebook": tool_data.get("facebook")
                        or tool_data.get("Facebook", ""),
                        "linkedin": tool_data.get("linkedin")
                        or tool_data.get("LinkedIn", ""),
                        "instagram": tool_data.get("instagram")
                        or tool_data.get("Instagram", ""),
                        "source_query": source_query,
                    },
                )
                sites_created += 1
                site_ids.append(site.id)
                logger.info(f"✅ Created ToolSite {site.id} for {title} ({url})")

            except Exception as e:
                logger.exception(f"Tool {index}: Failed to create site for {url} - {e}")
                job.logs.append(
                    f"Tool {index} ({title}): Failed to create site - {str(e)}"
                )

        # Process sites sequentially: scrape → store → index one at a time
        # Distribute sites across available workers for parallel processing
        if site_ids:
            logger.info(
                f"🔄 Processing {len(site_ids)} sites sequentially: "
                f"scrape → store → index (one site at a time per worker)"
            )
            logger.info(
                f"📊 Distributing {len(site_ids)} sites across workers for parallel processing"
            )
            # Store site IDs in job payload for sequential processing
            job.payload["_sequential_site_ids"] = site_ids
            job.payload["_current_site_index"] = 0
            job.save()

            # Distribute sites across workers (round-robin)
            # Each worker will process its assigned sites sequentially
            # Example with 4 workers: Worker 1 gets sites 0, 4, 8, 12...; Worker 2 gets 1, 5, 9, 13...; etc.
            # This allows parallel processing while maintaining sequential per-worker
            # FIX: Configurable via CELERY_WORKER_CONCURRENCY environment variable
            # Recommended: CELERY_WORKER_CONCURRENCY=8 (balanced performance and Apify memory limits)
            # Default fallback: 4 workers (only used if env var not set)
            # Number of workers should match Celery worker concurrency setting
            num_workers = int(os.getenv("CELERY_WORKER_CONCURRENCY", "4"))
            for worker_id in range(min(num_workers, len(site_ids))):
                worker_sites = [
                    site_ids[i] for i in range(worker_id, len(site_ids), num_workers)
                ]
                if worker_sites:
                    logger.info(
                        f"🚀 Worker {worker_id + 1} assigned {len(worker_sites)} sites: "
                        f"{worker_sites[:3]}{'...' if len(worker_sites) > 3 else ''}"
                    )
                    # Each worker processes its sites sequentially
                    process_site_sequentially.delay(job_id, worker_sites, 0)

        # Update job
        job.logs.append(
            f"Processed {len(tools_data)} tools. "
            f"Created {sites_created} sites. "
            f"Invalid URLs: {invalid_urls}. "
            f"Duplicates: {duplicate_urls}."
        )
        job.save()
        logger.info(
            f"✅ Job {job_id} internet discovery complete: {sites_created} sites created"
        )

    except Exception as e:
        logger.exception(
            f"❌ Fatal error in process_internet_discovered_tools for job {job_id}: {e}"
        )
        try:
            job = ScrapingJob.objects.get(id=job_id)
            job.status = ScrapingJob.Status.FAILED
            job.logs.append(f"Task failed: {str(e)}")
            job.finished_at = timezone.now()
            job.save()
        except Exception:
            pass
        raise


@shared_task(bind=True, max_retries=3)
def process_csv_rows(self, job_id):
    """
    Task A: Validate URLs, create site records, enqueue sitemap task.

    - Reads CSV file
    - Normalizes URLs (add scheme, remove query params)
    - Validates domains
    - Creates ToolSite records
    - Triggers Task B for each site
    """
    try:
        job = ScrapingJob.objects.get(id=job_id)
        job.status = ScrapingJob.Status.PROCESSING
        job.started_at = timezone.now()
        job.save()

        logger.info(f"Starting CSV processing for job {job_id}")

        # Read CSV
        try:
            df = pd.read_csv(job.file.path)
            logger.info(f"Successfully read CSV with {len(df)} rows")
        except Exception as e:
            logger.exception(f"Failed to read CSV file: {e}")
            job.logs.append(f"Error reading CSV: {str(e)}")
            job.status = ScrapingJob.Status.FAILED
            job.finished_at = timezone.now()
            job.save()
            return

        # Process each row and create ToolSite records
        sites_created = 0
        invalid_urls = 0
        site_ids = []  # Store site IDs for sequential processing

        for index, row in df.iterrows():
            url = row.get("Website", "")

            # Skip if URL is empty or invalid type
            if not isinstance(url, str) or not url.strip():
                logger.warning(f"Row {index}: Empty or invalid URL")
                invalid_urls += 1
                continue

            url = url.strip()

            # Normalize URL
            original_url = url
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "https://" + url
                logger.debug(f"Row {index}: Added https:// to {original_url}")

            # Remove query parameters and fragments
            if "?" in url:
                url = url.split("?")[0]
                logger.debug(f"Row {index}: Removed query params")
            if "#" in url:
                url = url.split("#")[0]
                logger.debug(f"Row {index}: Removed fragment")

            # Validate domain (basic check)
            try:
                from urllib.parse import urlparse

                parsed = urlparse(url)
                if not parsed.netloc or "." not in parsed.netloc:
                    raise ValueError("Invalid domain")
            except Exception as e:
                logger.warning(f"Row {index}: Invalid URL {url} - {e}")
                job.logs.append(f"Row {index}: Invalid URL {url}")
                invalid_urls += 1
                continue

            # Create ToolSite
            try:
                site = ToolSite.objects.create(
                    job=job,
                    csv_row_nr=int(row.get("Nr", index)),
                    website=url,
                    title=str(row.get("Title", "")),
                    description=str(row.get("Description", "")),
                    category=str(row.get("Category", "")),
                    master_category=str(row.get("Master Category", "")),
                    status=ToolSite.Status.PENDING,
                    # Store all other CSV fields in the fields JSONB
                    fields={
                        k: str(v) if pd.notna(v) else ""
                        for k, v in row.items()
                        if k
                        not in [
                            "Website",
                            "Nr",
                            "Title",
                            "Description",
                            "Category",
                            "Master Category",
                        ]
                    },
                )
                sites_created += 1
                site_ids.append(site.id)
                logger.info(f"Created ToolSite {site.id} for {url}")

            except Exception as e:
                logger.exception(f"Row {index}: Failed to create site for {url} - {e}")
                job.logs.append(f"Row {index}: Failed to create site - {str(e)}")

        # Process sites sequentially: scrape → store → index one at a time
        # Distribute sites across available workers for parallel processing
        if site_ids:
            logger.info(
                f"🔄 Processing {len(site_ids)} sites sequentially: "
                f"scrape → store → index (one site at a time per worker)"
            )
            logger.info(
                f"📊 Distributing {len(site_ids)} sites across workers for parallel processing"
            )
            # Store site IDs in job payload for sequential processing
            job.payload["_sequential_site_ids"] = site_ids
            job.payload["_current_site_index"] = 0
            job.save()

            # Distribute sites across workers (round-robin)
            # Each worker will process its assigned sites sequentially
            # Example with 4 workers: Worker 1 gets sites 0, 4, 8, 12...; Worker 2 gets 1, 5, 9, 13...; etc.
            # This allows parallel processing while maintaining sequential per-worker
            # FIX: Configurable via CELERY_WORKER_CONCURRENCY environment variable
            # Recommended: CELERY_WORKER_CONCURRENCY=8 (balanced performance and Apify memory limits)
            # Default fallback: 4 workers (only used if env var not set)
            # Number of workers should match Celery worker concurrency setting
            num_workers = int(os.getenv("CELERY_WORKER_CONCURRENCY", "4"))
            for worker_id in range(min(num_workers, len(site_ids))):
                worker_sites = [
                    site_ids[i] for i in range(worker_id, len(site_ids), num_workers)
                ]
                if worker_sites:
                    logger.info(
                        f"🚀 Worker {worker_id + 1} assigned {len(worker_sites)} sites: "
                        f"{worker_sites[:3]}{'...' if len(worker_sites) > 3 else ''}"
                    )
                    # Each worker processes its sites sequentially
                    process_site_sequentially.delay(job_id, worker_sites, 0)

        # Update job
        job.logs.append(
            f"Processed {len(df)} rows. Created {sites_created} sites. "
            f"Invalid URLs: {invalid_urls}."
        )
        job.save()
        logger.info(
            f"Job {job_id} CSV processing complete: {sites_created} sites created"
        )

    except Exception as e:
        logger.exception(f"Fatal error in process_csv_rows for job {job_id}: {e}")
        try:
            job = ScrapingJob.objects.get(id=job_id)
            job.status = ScrapingJob.Status.FAILED
            job.logs.append(f"Task failed: {str(e)}")
            job.finished_at = timezone.now()
            job.save()
        except Exception:
            pass
        raise


@shared_task(bind=True, max_retries=3)
def process_site_sequentially(self, job_id, site_ids, current_index):
    """
    Process sites sequentially: scrape → store → index one at a time.

    This ensures each site completes fully (scrape → aggregate → LLM → index)
    before moving to the next site. Multiple workers can process different sites
    in parallel, but each worker processes its sites sequentially.

    Args:
        self: Celery task instance (bound task)
        job_id: UUID of the ScrapingJob
        site_ids: List of site IDs to process
        current_index: Current index in the site_ids list
    """
    try:
        if current_index >= len(site_ids):
            logger.info(
                f"✅ Worker finished processing all assigned sites for job {job_id}"
            )
            return

        site_id = site_ids[current_index]
        logger.info(
            f"🔄 Worker processing site {current_index + 1}/{len(site_ids)} "
            f"(site_id: {site_id}) for job {job_id}"
        )

        # Store worker's site list in the site's fields for later retrieval
        # This allows the indexing task to know which sites belong to this worker
        site = ToolSite.objects.get(id=site_id)
        site.fields["_worker_site_ids"] = site_ids
        site.fields["_worker_site_index"] = current_index
        site.save()

        # Start processing this site
        # The flow: build_sitemap → scrape pages → aggregate → LLM → index
        # When indexing completes, it will trigger the next site automatically
        build_sitemap_and_enqueue_scrape.delay(site_id)

    except Exception as e:
        logger.exception(
            f"Error in process_site_sequentially for job {job_id}, site index {current_index}: {e}"
        )
        # Even if there's an error, try to continue to next site
        try:
            if current_index < len(site_ids):
                site_id = site_ids[current_index]
                site = ToolSite.objects.get(id=site_id)
                site.status = ToolSite.Status.FAILED
                site.save()
                # Continue to next site if this is part of a sequential job
                _continue_to_next_site_if_sequential(site)
        except Exception as inner_error:
            logger.exception(
                f"Error in error handler for process_site_sequentially: {inner_error}"
            )


@shared_task(bind=True, max_retries=3)
def build_sitemap_and_enqueue_scrape(self, site_id):
    """
    Task B: Generate sitemap, enqueue page scrape tasks.

    - PRIMARY: Tries to extract URLs from sitemap.xml (standard sitemap format)
    - FALLBACK: If sitemap.xml not found, parses HTML navigation elements
      (<nav>, <header>, <footer>, <menu>)
    - Extracts same-domain URLs
    - Creates SitePage records
    - Triggers Task C for each page
    """
    try:
        site = ToolSite.objects.get(id=site_id)
        site.status = ToolSite.Status.QUEUED
        site.save()

        logger.info(f"Building sitemap for site {site_id}: {site.website}")

        scraper = ApifyService()

        # Step 1: Try to extract URLs from sitemap.xml (primary method)
        logger.info(f"Attempting to extract URLs from sitemap.xml for {site.website}")
        urls = scraper.extract_sitemap_xml_urls(site.website)

        # Step 2: Fallback to HTML navigation parsing if sitemap.xml not found
        if not urls:
            logger.info(
                f"Sitemap.xml not found for {site.website}, falling back to HTML navigation parsing"
            )
            # Fetch homepage HTML (needed for link extraction)
            raw_html = scraper.fetch_page_content(site.website, return_markdown=False)

            if not raw_html:
                logger.error(f"Failed to fetch homepage for {site.website}")
                try:
                    site.status = ToolSite.Status.FAILED
                    site.save()
                    # Continue to next site if this is part of a sequential job
                    _continue_to_next_site_if_sequential(site)
                except Exception as e:
                    logger.exception(
                        f"Error handling homepage fetch failure for site {site.id}: {e}"
                    )
                return

            # Extract sitemap URLs from HTML navigation
            urls = scraper.extract_sitemap_urls(raw_html, site.website)
            logger.info(f"Extracted {len(urls)} URLs from HTML navigation")

            # If HTML extraction also found nothing, we'll still add homepage below
            # This ensures we at least try to scrape the homepage
        else:
            logger.info(f"Successfully extracted {len(urls)} URLs from sitemap.xml")

        # Ensure homepage is included (even if no other URLs found)
        if site.website not in urls:
            urls.insert(0, site.website)  # Add homepage at the beginning

        # Final check: if we still have no URLs at all (shouldn't happen, but safety check)
        if not urls:
            logger.error(
                f"No URLs found for site {site_id} ({site.website}) after all attempts"
            )
            try:
                site.status = ToolSite.Status.FAILED
                site.save()
                _continue_to_next_site_if_sequential(site)
            except Exception as e:
                logger.exception(
                    f"Error handling no URLs found for site {site_id}: {e}"
                )
            return

        # LIMIT: Only process first 10 URLs per site (or all if fewer than 10)
        MAX_PAGES_PER_SITE = 10
        original_url_count = len(urls)
        urls = urls[:MAX_PAGES_PER_SITE]  # Slice safely handles cases with < 10 URLs

        if original_url_count > MAX_PAGES_PER_SITE:
            logger.info(
                f"Limiting to first {MAX_PAGES_PER_SITE} URLs (found {original_url_count} total URLs)"
            )
        elif original_url_count < MAX_PAGES_PER_SITE:
            logger.info(
                f"Found {original_url_count} URLs (less than {MAX_PAGES_PER_SITE} limit, processing all)"
            )

        # SAFETY CHECK: Ensure we never exceed 10 pages per site
        existing_pages_count = site.pages.count()
        remaining_slots = MAX_PAGES_PER_SITE - existing_pages_count

        if remaining_slots <= 0:
            logger.warning(
                f"Site {site_id} already has {existing_pages_count} pages (max {MAX_PAGES_PER_SITE}), "
                f"skipping page creation"
            )
            # If site already has pages, it should continue processing
            # Don't mark as failed, just return (aggregation should be triggered elsewhere)
            return

        # Limit URLs to remaining slots
        if len(urls) > remaining_slots:
            urls = urls[:remaining_slots]
            logger.info(
                f"Limiting to {remaining_slots} URLs (site already has {existing_pages_count} pages)"
            )

        # Create SitePage records (max 10 per site total, or all if fewer)
        pages_created = 0
        for url in urls:
            # Avoid duplicates (unique_together constraint on site + url)
            page, created = SitePage.objects.get_or_create(
                site=site, url=url, defaults={"status": SitePage.Status.PENDING}
            )
            if created:
                pages_created += 1
                # Trigger Task C: Page scraping
                scrape_page_via_apify.delay(page.id)

                # Double-check: Stop if we've reached the limit
                if site.pages.count() >= MAX_PAGES_PER_SITE:
                    logger.info(
                        f"Reached {MAX_PAGES_PER_SITE} page limit for site {site_id}, stopping"
                    )
                    break

        logger.info(
            f"Created {pages_created} pages for site {site_id} "
            f"(found {original_url_count} URLs total, site now has {site.pages.count()}/{MAX_PAGES_PER_SITE} pages)"
        )

        # If no pages were created and no pages exist, mark as failed and continue
        if pages_created == 0 and site.pages.count() == 0:
            logger.warning(
                f"No pages created for site {site_id} ({site.website}), marking as failed"
            )
            site.status = ToolSite.Status.FAILED
            site.save()
            # Continue to next site if this is part of a sequential job
            _continue_to_next_site_if_sequential(site)

    except Exception as e:
        logger.exception(
            f"Error in build_sitemap_and_enqueue_scrape for site {site_id}: {e}"
        )
        try:
            site = ToolSite.objects.get(id=site_id)
            site.status = ToolSite.Status.FAILED
            site.save()
            # Continue to next site if this is part of a sequential job
            _continue_to_next_site_if_sequential(site)
        except Exception:
            pass


@shared_task(bind=True, max_retries=3)
def scrape_page_via_apify(self, page_id):
    """
    Task C: Fetch page, save HTML and text.

    - Fetches page via Apify
    - Saves raw_html
    - Cleans HTML (removes script/style)
    - Extracts page_text
    - Checks if all pages scraped for the site → triggers Task D
    """
    try:
        page = SitePage.objects.get(id=page_id)
        logger.info(f"Scraping page {page_id}: {page.url}")

        # Fetch markdown content (default is markdown)
        # FIX: Added better error handling for API timeouts and failures
        scraper = ApifyService()
        try:
            markdown_content = scraper.fetch_page_content(page.url)
        except Exception as scrape_error:
            logger.warning(
                f"Error fetching markdown for {page.url}: {scrape_error}, will retry with HTML"
            )
            markdown_content = None

        # If markdown not available, fetch HTML and clean it
        if not markdown_content or not markdown_content.strip():
            logger.warning(
                f"Markdown not available for {page.url}, falling back to HTML"
            )
            try:
                raw_html = scraper.fetch_page_content(page.url, return_markdown=False)
            except Exception as html_error:
                logger.error(f"Failed to fetch HTML for {page.url}: {html_error}")
                raw_html = None

            if not raw_html:
                logger.error(
                    f"Failed to fetch page {page.url} (both markdown and HTML failed)"
                )
                page.status = SitePage.Status.FAILED
                page.save()
                # FIX: Don't return immediately - check if this was the last page
                # This ensures aggregation can still be triggered even if some pages fail
                # Continue to the aggregation check below
                return
            cleaner = HTMLCleaner()
            clean_html, page_text = cleaner.clean(raw_html)
            # FIX: Remove NUL characters (0x00) which PostgreSQL doesn't allow in text fields
            page.raw_html = raw_html.replace("\x00", "") if raw_html else ""
            page.clean_html = clean_html.replace("\x00", "") if clean_html else ""
        else:
            # Use markdown content
            page_text = markdown_content
            # Still fetch HTML for reference
            raw_html = scraper.fetch_page_content(page.url, return_markdown=False)
            # FIX: Remove NUL characters (0x00) which PostgreSQL doesn't allow in text fields
            page.raw_html = raw_html.replace("\x00", "") if raw_html else ""
            page.clean_html = ""  # No need for clean HTML when we have markdown

        # FIX: Remove NUL characters from page_text before saving
        page.page_text = page_text.replace("\x00", "") if page_text else ""
        page.status = SitePage.Status.SCRAPED
        page.scraped_at = timezone.now()
        page.save()

        logger.info(
            f"Successfully scraped page {page_id}, text length: {len(page_text)}"
        )

        # Check if all pages for this site are finished (scraped or failed)
        # FIX: Use atomic transaction with select_for_update to prevent race condition
        # This ensures only one page can trigger aggregation, even if multiple finish simultaneously
        site_id = page.site_id
        with transaction.atomic():
            # Lock the site row to prevent concurrent modifications
            site = ToolSite.objects.select_for_update().get(id=site_id)

            # Check if aggregation was already triggered (idempotency check)
            if site.status in [ToolSite.Status.PROCESSED, ToolSite.Status.INDEXED]:
                logger.info(
                    f"Site {site_id} already processed/indexed, skipping aggregation trigger"
                )
                return

            # Check if CombinedText already exists (another idempotency check)
            if hasattr(site, "combined_text_record"):
                logger.info(
                    f"Site {site_id} already has CombinedText, skipping aggregation trigger"
                )
                return

            # Count pages atomically within the transaction
            total_pages = site.pages.count()
            scraped_pages = site.pages.filter(status=SitePage.Status.SCRAPED).count()
            failed_pages = site.pages.filter(status=SitePage.Status.FAILED).count()
            finished_pages = scraped_pages + failed_pages

            if finished_pages == total_pages and total_pages > 0:
                # Set a flag in the site's fields to prevent duplicate triggers
                # This acts as a second layer of protection
                if site.fields.get("_aggregation_triggered"):
                    logger.info(
                        f"Aggregation already triggered for site {site_id}, skipping"
                    )
                    return

                # Mark aggregation as triggered
                site.fields["_aggregation_triggered"] = True
                site.save(update_fields=["fields"])

                logger.info(
                    f"All {total_pages} pages finished for site {site.id} ({scraped_pages} scraped, {failed_pages} failed), triggering aggregation"
                )
                # Trigger aggregation outside the transaction to avoid long-running operations
                aggregate_site_text.delay(site.id)

    except Exception as e:
        logger.exception(f"Error in scrape_page_via_apify for page {page_id}: {e}")
        try:
            page = SitePage.objects.get(id=page_id)
            page.status = SitePage.Status.FAILED
            page.save()
            # FIX: Even if page scraping fails, check if all pages are done
            # This ensures aggregation can be triggered even with some failures
            site = page.site
            total_pages = site.pages.count()
            finished_pages = site.pages.filter(
                status__in=[SitePage.Status.SCRAPED, SitePage.Status.FAILED]
            ).count()

            if finished_pages == total_pages and total_pages > 0:
                # Use atomic transaction to prevent race condition
                with transaction.atomic():
                    site = ToolSite.objects.select_for_update().get(id=site.id)
                    if (
                        site.status
                        not in [ToolSite.Status.PROCESSED, ToolSite.Status.INDEXED]
                        and not hasattr(site, "combined_text_record")
                        and not site.fields.get("_aggregation_triggered")
                    ):
                        site.fields["_aggregation_triggered"] = True
                        site.save(update_fields=["fields"])
                        logger.info(
                            f"All pages finished for site {site.id} (including failures), triggering aggregation"
                        )
                        aggregate_site_text.delay(site.id)
        except Exception as inner_error:
            logger.exception(
                f"Error in error handler for page {page_id}: {inner_error}"
            )


@shared_task(bind=True, max_retries=3)
def aggregate_site_text(self, site_id):
    """
    Task D: Combine text, create CombinedText record, trigger LLM and embedding.

    - Concatenates all page_text (navigation order first, then alphabetical)
    - Deduplicates similar content to avoid repetitive chunks
    - Stores as CombinedText record with UUID
    - Triggers Task E (LLM extraction) and embedding generation

    FIX: Added idempotency checks to prevent duplicate aggregation.
    """
    try:
        site = ToolSite.objects.get(id=site_id)

        # FIX: Idempotency check - skip if already processed/indexed
        if site.status in [ToolSite.Status.PROCESSED, ToolSite.Status.INDEXED]:
            logger.info(
                f"Site {site_id} already in status {site.status}, skipping aggregation"
            )
            return

        # FIX: Idempotency check - skip if CombinedText already exists
        if hasattr(site, "combined_text_record"):
            logger.info(
                f"Site {site_id} already has CombinedText record {site.combined_text_record.id}, skipping aggregation"
            )
            # If site is not PROCESSED but has CombinedText, update status
            if site.status != ToolSite.Status.PROCESSED:
                site.status = ToolSite.Status.PROCESSED
                site.save(update_fields=["status"])
            return

        logger.info(f"Aggregating text for site {site_id}")

        # Get all scraped pages in logical order
        # LIMIT: Only process first 10 pages per site
        MAX_PAGES_PER_SITE = 10
        all_pages = site.pages.filter(status=SitePage.Status.SCRAPED).order_by("url")
        pages = all_pages[:MAX_PAGES_PER_SITE]

        total_scraped = all_pages.count()
        if total_scraped > MAX_PAGES_PER_SITE:
            logger.info(
                f"Limiting aggregation to first {MAX_PAGES_PER_SITE} pages "
                f"(found {total_scraped} scraped pages total)"
            )

        # If no pages were scraped (all failed), mark site as failed and continue
        if total_scraped == 0:
            total_pages = site.pages.count()
            try:
                if total_pages > 0:
                    logger.warning(
                        f"Site {site_id} has {total_pages} pages but none were successfully scraped, marking as failed"
                    )
                    site.status = ToolSite.Status.FAILED
                    site.save()
                    # Continue to next site if this is part of a sequential job
                    _continue_to_next_site_if_sequential(site)
                else:
                    logger.warning(
                        f"Site {site_id} has no pages at all, marking as failed"
                    )
                    site.status = ToolSite.Status.FAILED
                    site.save()
                    # Continue to next site if this is part of a sequential job
                    _continue_to_next_site_if_sequential(site)
            except Exception as e:
                logger.exception(
                    f"Error handling no pages scraped for site {site_id}: {e}"
                )
            return

        # Deduplicate and concatenate page text
        seen_texts = set()
        unique_texts = []

        for page in pages:
            if page.page_text and page.page_text.strip():
                # Use first 200 characters as fingerprint to detect duplicates
                fingerprint = page.page_text[:200].strip()
                if fingerprint not in seen_texts:
                    seen_texts.add(fingerprint)
                    unique_texts.append(page.page_text.strip())
                    logger.debug(f"Added unique text from {page.url}")
                else:
                    logger.debug(f"Skipped duplicate text from {page.url}")

        combined_text_content = "\n\n".join(unique_texts)

        logger.info(f"Combined text length: {len(combined_text_content)} characters")

        # If combined text is empty (all pages had no text), mark as failed and continue
        if not combined_text_content or len(combined_text_content.strip()) == 0:
            logger.warning(
                f"Site {site_id} has no text content after aggregation, marking as failed"
            )
            try:
                site.status = ToolSite.Status.FAILED
                site.save()
                # Continue to next site if this is part of a sequential job
                _continue_to_next_site_if_sequential(site)
            except Exception as e:
                logger.exception(
                    f"Error handling empty combined text for site {site_id}: {e}"
                )
            return

        # FIX: Use get_or_create to prevent duplicate CombinedText records
        # This provides additional idempotency protection
        combined_text_record, created = CombinedText.objects.get_or_create(
            site=site,
            defaults={
                "combined_text": combined_text_content,
                "char_count": len(combined_text_content),
            },
        )

        if created:
            logger.info(
                f"Created CombinedText record {combined_text_record.id} for site {site.id}"
            )
        else:
            logger.info(
                f"CombinedText record {combined_text_record.id} already exists for site {site.id}, using existing"
            )
            # Update if content changed (shouldn't happen, but safety check)
            if combined_text_record.combined_text != combined_text_content:
                combined_text_record.combined_text = combined_text_content
                combined_text_record.char_count = len(combined_text_content)
                combined_text_record.save()

        site.status = ToolSite.Status.PROCESSED
        site.save()

        # Chain tasks: LLM extraction FIRST, then indexing
        # This ensures metadata is extracted before embeddings are generated
        chain(
            generate_and_store_fields.s(site_id),
            generate_embeddings_and_index.s(site_id),
        ).delay()

        logger.info(f"Site {site_id} text aggregation complete")

    except Exception as e:
        logger.exception(f"Error in aggregate_site_text for site {site_id}: {e}")
        try:
            site = ToolSite.objects.get(id=site_id)
            site.status = ToolSite.Status.FAILED
            site.save()
            # Continue to next site if this is part of a sequential job
            _continue_to_next_site_if_sequential(site)
        except Exception:
            pass


@shared_task(bind=True, max_retries=3)
def generate_and_store_fields(self, site_id):
    """
    Task E: Call LLM to fill ~200 metadata fields.

    - Passes combined_text to LLM
    - Extracts metadata fields
    - Stores in ToolSite.fields (JSONB)
    - Updates dedicated columns (category, master_category, etc.)
    """
    try:
        site = ToolSite.objects.get(id=site_id)
        logger.info(f"Generating metadata fields for site {site_id}")

        # Get combined text from CombinedText record
        combined_text_record = getattr(site, "combined_text_record", None)
        if not combined_text_record:
            logger.error(f"No combined text record found for site {site_id}")
            return

        # Extract metadata via LLM
        llm_service = LLMService()
        metadata = llm_service.extract_metadata(
            combined_text_record.combined_text,
            existing_data={
                "title": site.title,
                "description": site.description,
                "category": site.category,
                "master_category": site.master_category,
                **site.fields,
            },
        )

        # Update site with extracted metadata
        # Preserve worker tracking keys (_worker_site_ids, _worker_site_index) if they exist
        worker_tracking_keys = {}
        if "_worker_site_ids" in site.fields:
            worker_tracking_keys["_worker_site_ids"] = site.fields["_worker_site_ids"]
        if "_worker_site_index" in site.fields:
            worker_tracking_keys["_worker_site_index"] = site.fields[
                "_worker_site_index"
            ]

        # Merge metadata with preserved worker tracking keys
        site.fields = {**metadata, **worker_tracking_keys}

        # Update dedicated columns if LLM provided better values
        if metadata.get("title"):
            site.title = metadata["title"]
        if metadata.get("category"):
            site.category = metadata["category"]
        if metadata.get("master_category"):
            site.master_category = metadata["master_category"]

        site.save()

        logger.info(
            f"Successfully extracted {len(metadata)} metadata fields for site {site_id}"
        )

    except Exception as e:
        logger.exception(f"Error in generate_and_store_fields for site {site_id}: {e}")
        try:
            site = ToolSite.objects.get(id=site_id)
            site.status = ToolSite.Status.FAILED
            site.save()
            # Continue to next site if this is part of a sequential job
            _continue_to_next_site_if_sequential(site)
        except Exception:
            pass


@shared_task(bind=True, max_retries=3)
def generate_embeddings_and_index(self, previous_result, site_id):
    """
    Task F: Generate embeddings and push to Pinecone.

    - Gets combined text from CombinedText record
    - Chunks text in memory
    - Generates embeddings for each chunk
    - Indexes directly to Pinecone (no database storage of chunks)

    Args:
        self: Celery task instance (bound task)
        previous_result: Result from previous task in chain (automatically passed, ignored)
        site_id: UUID of the ToolSite (from .s(site_id) signature)
    """
    try:
        site = ToolSite.objects.get(id=site_id)
        logger.info(f"Generating embeddings and indexing for site {site_id}")

        # Get combined text record
        combined_text_record = getattr(site, "combined_text_record", None)
        if not combined_text_record:
            logger.error(f"No combined text record found for site {site_id}")
            return

        # Use PineconeService to handle chunking, embedding, and indexing
        pinecone_service = PineconeService()
        success = pinecone_service.index_site(site)

        if success:
            site.status = ToolSite.Status.INDEXED
            site.save()
            logger.info(f"Successfully indexed site {site_id}")
        else:
            site.status = ToolSite.Status.FAILED
            site.save()
            logger.error(f"Failed to index site {site_id}")
            # Still continue to check for next site even if this one failed

        # Check if this site was part of a sequential job (worker-specific)
        job = site.job
        worker_site_ids = site.fields.get("_worker_site_ids")
        worker_site_index = site.fields.get("_worker_site_index", -1)

        if worker_site_ids and worker_site_index >= 0:
            # This is a sequential job - trigger next site for this worker
            next_index = worker_site_index + 1
            if next_index < len(worker_site_ids):
                logger.info(
                    f"✅ Site {site_id} indexed, worker moving to next site "
                    f"({next_index + 1}/{len(worker_site_ids)})"
                )
                # Process next site for this worker
                process_site_sequentially.delay(job.id, worker_site_ids, next_index)
            else:
                logger.info(
                    f"✅ Worker finished all {len(worker_site_ids)} assigned sites"
                )
                # This worker is done - check if all workers are done
                _check_job_completion(job)
        else:
            # Legacy parallel processing - check if all sites are done
            _check_job_completion(job)

        logger.info(f"Completed indexing for site {site_id}")

    except Exception as e:
        logger.exception(
            f"Error in generate_embeddings_and_index for site {site_id}: {e}"
        )
        try:
            site = ToolSite.objects.get(id=site_id)
            site.status = ToolSite.Status.FAILED
            site.save()
            # Continue to next site if this is part of a sequential job
            _continue_to_next_site_if_sequential(site)
        except Exception:
            pass
