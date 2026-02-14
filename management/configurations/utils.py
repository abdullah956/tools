"""This module contains utility functions for sending emails."""

import requests
from envs.env_loader import env_loader


def send_email(recipient_email, subject, body, timeout=30):
    """Send email using Mailgun API.

    Args:
        recipient_email (str): Email address of the recipient
        subject (str): Subject of the email
        body (str): Body content of the email
        timeout (int): Timeout in seconds for the API request (default: 30)

    Returns:
        tuple: (bool, str) - (success status, error message if any)
    """
    try:
        # Mailgun configuration
        MAILGUN_API_KEY = env_loader.mailgun_api_key
        MAILGUN_DOMAIN = env_loader.mailgun_domain
        MAILGUN_BASE_URL = f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}"

        print("Preparing to send email via Mailgun...")

        # Prepare the email data
        data = {
            "from": f"App <noreply@{MAILGUN_DOMAIN}>",
            "to": recipient_email,
            "subject": subject,
            "text": body,
        }

        # Send the request to Mailgun API with timeout
        print("Sending request to Mailgun API...")
        response = requests.post(
            f"{MAILGUN_BASE_URL}/messages",
            auth=("api", MAILGUN_API_KEY),
            data=data,
            timeout=timeout,
        )

        # Check if the request was successful
        if response.status_code == 200:
            print("Message sent successfully!")
            return True, None
        else:
            error_msg = f"Failed to send email. Response: {response.text}"
            print(error_msg)
            return False, error_msg

    except requests.Timeout:
        error_msg = "Request timed out while sending email"
        print(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return False, error_msg


def send_html_email(recipient_email, subject, html_body, text_body=None, timeout=30):
    """Send HTML email using Mailgun API.

    Args:
        recipient_email (str): Email address of the recipient
        subject (str): Subject of the email
        html_body (str): HTML body content of the email
        text_body (str, optional): Plain text alternative. If None, a simple text
                                  version will be extracted from HTML.
        timeout (int): Timeout in seconds for the API request (default: 30)

    Returns:
        tuple: (bool, str) - (success status, error message if any)
    """
    try:
        # Mailgun configuration
        MAILGUN_API_KEY = env_loader.mailgun_api_key
        MAILGUN_DOMAIN = env_loader.mailgun_domain
        MAILGUN_BASE_URL = f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}"

        print("Preparing to send HTML email via Mailgun...")

        # If no text body is provided, create a simple one
        if text_body is None:
            # Very simple text extraction - for better results consider using a
            # proper HTML to text converter
            text_body = "Please view this email in an HTML-capable email client."

        # Prepare the email data
        data = {
            "from": f"App <noreply@{MAILGUN_DOMAIN}>",
            "to": recipient_email,
            "subject": subject,
            "html": html_body,
            "text": text_body,
        }

        # Send the request to Mailgun API with timeout
        print("Sending HTML request to Mailgun API...")
        response = requests.post(
            f"{MAILGUN_BASE_URL}/messages",
            auth=("api", MAILGUN_API_KEY),
            data=data,
            timeout=timeout,
        )

        # Check if the request was successful
        if response.status_code == 200:
            print("HTML message sent successfully!")
            return True, None
        else:
            error_msg = f"Failed to send HTML email. Response: {response.text}"
            print(error_msg)
            return False, error_msg

    except Exception as e:
        error_msg = f"Exception occurred while sending HTML email: {str(e)}"
        print(error_msg)
        return False, error_msg
