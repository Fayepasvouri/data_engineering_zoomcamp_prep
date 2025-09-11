FROM python:3.9

RUN apt-get update && apt-get install -y wget
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pandas pyarrow sqlalchemy psycopg2

WORKDIR /app
COPY ingest_data.py ingest_data.py

ENTRYPOINT ["python", "ingest_data.py"]