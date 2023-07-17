FROM python:3.9

WORKDIR /app

COPY app.py .


RUN pip install numpy pandas scikit-learn datasets PIL functools torch datasets transformers evaluate huggingface_hub

CMD ["python", "app.py"]