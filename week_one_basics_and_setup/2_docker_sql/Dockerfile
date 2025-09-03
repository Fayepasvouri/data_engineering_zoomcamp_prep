FROM python:3.9

RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pandas

WORKDIR /app
COPY pipeline.py pipeline.py

ENTRYPOINT ["bash"]