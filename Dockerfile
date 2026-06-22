# ============================================================
# Dockerfile for Customer Churn Prediction API
# ============================================================

# Step 1 — Base image
FROM python:3.11-slim

# Step 2 - Set working directory inside container
WORKDIR /app

# Step 3 - Copy requirements 
COPY requirements.txt .

# Step 4 — Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 5 — Copy the rest of your project into the container
COPY . .

# Step 6 — Expose the port your API listens on
EXPOSE 8000

# Step 7 — The command that starts your API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
