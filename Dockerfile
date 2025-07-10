FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install accelerate

EXPOSE 8000

CMD ["uvicorn", "backendkiwi:app", "--host", "0.0.0.0", "--port", "8000"]