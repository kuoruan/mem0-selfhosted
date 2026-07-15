FROM python:3.12-slim

RUN apt-get update \
	&& apt-get install -y --no-install-recommends libpq5 curl \
	&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install server dependencies (without mem0ai)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install mem0 from local source (via additional_contexts: root=..)
WORKDIR /tmp
COPY --from=root mem0 ./mem0
COPY --from=root pyproject.toml .
COPY --from=root README.md .
RUN pip install --no-cache-dir ".[nlp]" \
	&& rm -rf /tmp/mem0 /tmp/pyproject.toml /tmp/README.md

WORKDIR /app
# Copy server code
COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]