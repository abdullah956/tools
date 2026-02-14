"""Meeting booking email template."""

meeting_booking_template = """
<!DOCTYPE html><html lang="en"><head>
  <title>Meeting Request - example.com Contractor Booking</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      background-color: #f7f7f7;
      color: #333;
      padding: 20px;
    }}

    .container {{
      max-width: 600px;
      margin: auto;
      background-color: #ffffff;
      padding: 30px;
      border-radius: 8px;
    }}

    .meeting-details {{
      background-color: #f8f9fa;
      padding: 20px;
      border-radius: 5px;
      margin: 20px 0;
    }}

    .button {{
      background-color: #4a90e2;
      color: white !important;
      text-decoration: none;
      padding: 12px 24px;
      border-radius: 5px;
      font-size: 16px;
      display: inline-block;
    }}

    @media (prefers-color-scheme: dark) {{
      body {{
        background-color: #121212;
        color: #e0e0e0;
      }}

      .container {{
        background-color: #1e1e1e;
      }}

      .meeting-details {{
        background-color: #2d2d2d;
      }}

      h2, h3 {{
        color: #ffffff;
      }}

      .button {{
        background-color: #3b82f6;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <h2>Meeting Request</h2>

    <p>Dear {contractor_name},</p>

    <p>You have received a new meeting request through example.com platform.</p>

    <div class="meeting-details">
      <h3>Meeting Details:</h3>
      <p><strong>Requested by:</strong> {client_name}</p>
      <p><strong>Company:</strong> {company_name}</p>
      <p><strong>Project Description:</strong> {project_description}</p>
      <p><strong>Preferred Date:</strong> {preferred_date}</p>
      <p><strong>Preferred Time:</strong> {preferred_time}</p>
    </div>

    <p>Please review this meeting request and respond at your earliest convenience.</p>

    <p style="text-align: center; margin: 30px 0;">
      <a href="{accept_meeting_link}" class="button">Accept Meeting</a>
    </p>

    <p>If you need to suggest an alternative time or have any questions, please respond directly to this email or contact the client at {client_email}.</p>

    <br>

    <p>Best regards,<br>
    <strong>The example.com Team</strong></p>
  </div>
</body></html>
"""
