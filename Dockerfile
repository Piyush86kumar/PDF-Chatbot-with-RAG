FROM python:3.12.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app files
COPY . .

# Huggingface Writes to /tmp/ .streamlit
ENV STREAMLIT_HOME=/tmp/.streamlit
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose streamlit default port 
EXPOSE 8501

# Heathcheck so HF knows the app is running 
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

#RUN the app 
ENTRYPOINT [ "streamlit", "run","src/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0"]