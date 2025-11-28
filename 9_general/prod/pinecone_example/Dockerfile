FROM python:3.9-slim

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

# Run Flask app
CMD ["python", "app/main.py"]
