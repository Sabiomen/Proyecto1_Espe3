# Imagen base
FROM python:3.11-slim

# Crear directorio de la app
WORKDIR /app

# Instalar dependencias del sistema (ej. para faiss y pypdf)
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar dependencias
COPY requirements.txt .

# Instalar Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Variables de entorno
ENV OPENAI_API_KEY=changeme
ENV DEEPSEEK_API_KEY=changeme

# Puerto
EXPOSE 80

# Comando de inicio
CMD ["gunicorn", "-b", "0.0.0.0:80", "web:app"]
