FROM python:3.11-slim

WORKDIR /api

COPY ./api/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

COPY ./api /api 

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]