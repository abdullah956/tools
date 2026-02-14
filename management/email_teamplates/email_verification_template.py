"""Email verification template."""

email_verification_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333333;
            margin: 0;
            padding: 20px;
        }
        .button {
            display: inline-block;
            padding: 10px 20px;
            background-color: #27ae60;
            color: #ffffff;
            text-decoration: none;
            border-radius: 4px;
            font-weight: 500;
        }
    </style>
</head>
<body>
    <h2>Welcome to Early Access!</h2>
    <p>Thank you for registering for early access to our platform. You are one of our
    early stars!</p>
    <p>Please verify your email address by clicking the link below:</p>
    <p><a href="verification_url" class="button">Verify Email Address</a></p>
    <p>If you have any questions or need assistance, please don't hesitate to contact
    our support team.</p>
    <p>We're excited to soon be able and open up our doors and until then expect an
    email from us when we go live. The payment solution for your lifetime subscription
    is live and if you feel that you want to help us out before we go live, much
    appreciated!</p>
    <p>Best regards,<br>
    The Team</p>
</body>
</html>
"""
