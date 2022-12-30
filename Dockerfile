FROM pytorchlightning/pytorch_lightning

WORKDIR /app

COPY src src

CMD ["python", "./src/index.py"]
