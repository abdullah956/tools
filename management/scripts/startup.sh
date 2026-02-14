#!/bin/bash
set -e

# Environment file check (optional in ECS)
if [ ! -f /app/.env_vars/.env.staging ]; then
  echo "Warning: .env.staging file not found. Using environment variables from ECS."
fi

echo "Starting application..."

# Apply database migrations
echo "Applying database migrations..."
python manage.py migrate --noinput

# Collect static files
echo "Collecting static files..."
python manage.py collectstatic --noinput

# Start Gunicorn server
echo "Starting Gunicorn server..."
gunicorn management.wsgi:application --bind 0.0.0.0:8000 --workers 4 --timeout 300 --access-logfile - --error-logfile -
