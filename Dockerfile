# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data needed for preprocessing
RUN python -m nltk.downloader -d /usr/local/nltk_data stopwords punkt punkt_tab

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV SENTIFY_API_KEY=""

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
