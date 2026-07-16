FROM python:3.12

WORKDIR /app

# Install server dependencies (without mem0ai)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install mem0 from local source in editable mode
WORKDIR /app/packages
COPY --from=root mem0 ./mem0
COPY --from=root pyproject.toml .
COPY --from=root README.md .
RUN pip install -e ".[nlp]"

# Return to app directory and copy server code
WORKDIR /app
COPY . .

ENTRYPOINT ["sh", "/app/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
