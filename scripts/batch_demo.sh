#!/bin/bash
set -e

# Ruta del proyecto
PROJECT_DIR="$(pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="python3"

echo "Iniciando ejecución automática del pipeline y evaluación de gold set..."

# Crear entorno virtual si no existe
if [ ! -d "$VENV_DIR" ]; then
    echo "Creando entorno virtual en $VENV_DIR..."
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

# Activar entorno virtual
echo "Activando entorno virtual..."
source "$VENV_DIR/bin/activate"

# Instalar dependencias
if [ -f "requirements.txt" ]; then
    echo "Instalando dependencias desde requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "No se encontró requirements.txt, instalando dependencias básicas..."
    pip install --upgrade pip
    pip install pypdf pdfminer.six sentence-transformers faiss-cpu openai python-dotenv pandas click flask
fi

# Ejecutar ingesta
echo "Ejecutando ingesta de documentos..."
python -m rag.ingest

# Generar embeddings e índice
echo "Generando embeddings y creando índice FAISS..."
python -m rag.embed

# Ejecutar evaluación batch con gold_set usando el proveedor chatgpt (puedes cambiarlo)
echo "Ejecutando evaluación batch sobre gold_set con ChatGPT..."
python app.py evaluate --provider chatgpt --k 4

echo "Pipeline y evaluación completados exitosamente."