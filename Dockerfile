FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (needed for opencv & tensorflow)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first (important)
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements first (for caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Now copy rest of the app
COPY . .

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]