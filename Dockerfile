FROM python:3.8-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port for the Flask API
EXPOSE 5000

# Command to run when starting the container
CMD ["python", "api.py"] 