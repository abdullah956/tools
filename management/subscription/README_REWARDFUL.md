# Rewardful Affiliate Program Integration

This document explains how the Rewardful affiliate program is integrated with our subscription system.

## Overview

The integration allows affiliates to earn a 30% commission on successful subscription purchases made through their referral links. The system works by:

1. Capturing the Rewardful referral ID (`ref`) from the frontend
2. Storing this ID in Stripe Checkout metadata
3. Processing the commission when a payment is successful

## Implementation Details

### Environment Setup

Add the Rewardful API key to your environment variables:

```
REWARDFUL_API_KEY=your_rewardful_api_key
```

### Data Flow

1. **Frontend**: The frontend passes the `ref` parameter from Rewardful to the backend when creating a checkout session.

2. **Checkout Creation**: The backend stores this referral ID in:
   - The Stripe Checkout session metadata
   - The UserSubscription record in our database

3. **Payment Success**: When payment is successful (either via webhook or redirect), the system:
   - Confirms the subscription is active
   - Retrieves payment details from Stripe
   - Sends the commission information to Rewardful's API

### API Endpoints

#### Create Checkout Session

```
POST /subscription/api/subscription-products/checkout/
```

Parameters:
- `product_id`: ID of the subscription product
- `ref`: (Optional) Rewardful referral ID

#### Webhook Handler

```
POST /subscription/webhook/stripe/
```

This endpoint listens for Stripe's `checkout.session.completed` event and processes the Rewardful commission.

### Commission Structure

- Commission Type: Percentage
- Commission Value: 30%

## Testing

The integration includes test cases that verify:
- The referral ID is correctly stored in the checkout session
- The Rewardful API is called with the correct parameters when a payment is successful

## Troubleshooting

Common issues:
- Missing Rewardful API key in environment variables
- Network errors when contacting Rewardful API
- Invalid referral IDs

Check the logs for detailed error messages related to the Rewardful integration.

## Frontend Integration

To integrate with the frontend, ensure that:
1. The Rewardful JavaScript snippet is added to your website
2. The `ref` parameter from the URL is passed to the checkout endpoint
3. The checkout flow preserves the referral information

Example frontend code:

```javascript
// Get referral ID from URL
const urlParams = new URLSearchParams(window.location.search);
const referralId = urlParams.get('ref');

// Pass to checkout API
async function createCheckout(productId) {
  const response = await fetch('/subscription/api/subscription-products/checkout/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      product_id: productId,
      ref: referralId  // Include referral ID if present
    }),
  });

  const data = await response.json();
  window.location.href = data.checkout_url;
}
```
