FROM pytorchlightning/pytorch_lightning:latest

WORKDIR /app

COPY src src

CMD ["python", "./src/index.py"]
