# Use the official Python image from the Docker Hub
FROM python:3.11

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y curl

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install -y git-lfs

# Set the working directory inside the container
WORKDIR /app2/app/models

RUN git clone https://huggingface.co/facebook/fasttext-it-vectors
RUN git clone https://huggingface.co/Babelscape/wikineural-multilingual-ner
RUN git clone https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
#RUN git clone https://huggingface.co/Jiva/xlm-roberta-large-it-mnli

# Set the working directory inside the container
WORKDIR /app2/

# Install Git LFS hooks (required after git-lfs installation)
RUN git lfs install

# Copy the requirements.txt file into the container at /app2
COPY requirements.txt /app2/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app2
COPY . /app2/

RUN python3 /app2/app/utils/reduce_fasttext_size.py

# Expose 7766 port
EXPOSE 7766

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7766"]
