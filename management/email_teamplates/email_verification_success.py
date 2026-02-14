"""Email verification success template."""

email_verification_success_template = """
<!DOCTYPE html><html lang="en"><head>

  <title>Welcome to example.com Early Access</title>
  <style>
    :root {

    }

    body {
      font-family: Arial, sans-serif;
      background-color: #f7f7f7;
      color: #333;
      padding: 20px;
    }

    .container {
      max-width: 600px;
      margin: auto;
      background-color: #ffffff;
      padding: 30px;
      border-radius: 8px;
    }

    a.button {
      background-color: #4a90e2;
      color: white !important;
      text-decoration: none;
      padding: 12px 24px;
      border-radius: 5px;
      font-size: 16px;
      display: inline-block;
    }

    @media (prefers-color-scheme: dark) {
      body {
        background-color: #121212;
        color: #e0e0e0;
      }

      .container {
        background-color: #1e1e1e;
      }

      h2, h3 {
        color: #ffffff;
      }

      a.button {
        background-color: #3b82f6;
      }
    }
  </style>
</head>
<body>
  <div class="container">


    <p>Hey, we hope you're ready!</p>
    <p>Your email is verified successfully ✅</p>

    <p>We're excited to officially welcome you to the early access of <strong>example.com</strong> — and thank you for being one of the pioneers on this journey.</p>

    <p>Because you joined the waiting list early, you now have <strong>lifetime access</strong> to the platform. That means you're not just using example.com — you're helping shape it.</p>

<p style="text-align: center; margin: 30px 0;">
  <a href="https://app.example.com/signup" style="background-color: #4a90e2; color: white; text-decoration: none; padding: 12px 24px; border-radius: 5px; font-size: 16px;">Go to app.example.com</a>
</p>

    <p>You received a link to the Lifetime benefits payment with Stripe earlier from noreply@example.com.</p>

    <p>Our vision is bold: <em>take a simple AI or automation query and turn it into full implementation.</em> Whether it's a single-use case or a complex workflow, we're building tools that get you from idea to execution, fast.</p>

    <p>In the coming weeks, you'll see new features, improvements, and community-driven updates. Want a sneak peek into what's coming? Join our community and see the <strong>feature roadmap</strong> firsthand.</p>

    <br>

    <h3>👥 Join the Community on Discord</h3>
    <p>We've set up a private Discord server just for early access users like you. It's where:</p>
    <ul>
      <li>You get real-time updates</li>
      <li>You can share feedback directly with our team</li>
      <li>You help shape features before they launch</li>
      <li>You see ideas from the community turn into working features</li>
    </ul>

    <p><strong>New to Discord?</strong> It's a free chat and community app. Just click below, create a quick account if needed, and join us.</p>

    <p style="text-align: center; margin: 30px 0;">
      <a href="https://discord.gg/YOUR_INVITE_LINK" class="button">Join the Community on Discord</a>
    </p>

    <br>

    <h3>📢 Help Us Grow the Future of AI and Automation</h3>
    <p>We've saved a few exclusive spots for new users. If you know someone curious about building with AI and automation — whether they're technical or not — invite them to the waiting list.</p>

    <p>The more voices we have, the stronger we build.</p>

    <br>

    <p>Thanks again for being part of this,<br>
    <strong>The example.com Team</strong></p>
  </div>


</body></html>
"""
