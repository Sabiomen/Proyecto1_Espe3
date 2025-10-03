# Proyecto1_Espe3
# UFRO RAG Chatbot

Este proyecto desarrolla un asistente conversacional que combina recuperación de información y generación de texto para brindar respuestas precisas sobre la normativa y reglamentos de la Universidad de La Frontera (UFRO), apoyándose en documentos oficiales almacenados localmente como PDFs y archivos de texto.

---

## Flujo del Proyecto

1. **Ingesta** de documentos normativos (PDF/TXT), extrae texto de PDFs y TXT, lo divide en fragmentos con metadatos y lo guarda procesado.
2. **Embeddings** convierte fragmentos en vectores con all-MiniLM-L6-v2 y crea índice **FAISS** para búsqueda.
3. **Consulta** busca fragmentos relevantes y genera respuesta con LLM (ChatGPT o DeepSeek) vía providers.
4. **Evaluación** prueba automática con preguntas gold set y cálculo de métricas clave.

---

## Requisitos

* **Python 3.10+**
* **Git**
* **Claves API válidas**:

  * `OPENAI_API_KEY` (para ChatGPT vía OpenRouter)
  * `DEEPSEEK_API_KEY` (para DeepSeek)

---

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/Sabiomen/Proyecto1_Espe3
cd Proyecto1_Espe3
```

Dar permisos al script de ejecución (opcional, Linux):

```bash
chmod 700 scripts/batch_demo.sh
```

Configurar las variables de entorno en `.env`:

```bash
cp .env.example .env
```

Luego edita `.env` para agregar tus API keys (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`).

---

## ▶️ Ejecución

Ejecutar el pipeline completo (creación venv, instalación, ingesta, embeddings, evaluación, demo):

```bash
./scripts/batch_demo.sh
```

o

```bash
bash scripts/batch_demo.sh
```

---

## 💬 Consultas manuales (CLI)

Antes de realizar consultas manuales, activa el entorno virtual:

**Linux/macOS**

```bash
source appenv/bin/activate
```

**Windows**

```powershell
 .\.venv\Scripts\activate
```

Ejecuta el chatbot:

```bash
python app.py chat --provider <PROVEEDOR> --k <N> 
```

### Parámetros disponibles

* `--provider` → modelo LLM a usar. Valores:

  * `chatgpt`
  * `deepseek`

* `--k` → número de fragmentos recuperados desde FAISS.

  * Valores: enteros (ej: `3`, `5`). Default: `3`.

  * Ejemplo:

    ```bash
    python app.py chat --provider chatgpt --k 4

    "Dame las normas de convivencia"
    ```

---

## 📊 Evaluación

El sistema incluye un archivo `eval/gold_set.jsonl` con 10 Q&A de referencia.

Para correr la evaluación:

```bash
python eval/evaluate.py
```

Esto genera archivos CSV con métricas como:

* Exact Match (EM)
* Similitud coseno
* Cobertura de citas (%)
* Latencia end-to-end y por etapa (retriever/LLM)
* Costo estimado por consulta (tokens × tarifa)

---

## ⚖️ Ética y Política de Abstención

* El asistente **no reemplaza consultas oficiales** ni emite asesoría legal vinculante.
* Se abstiene de responder a casos específicos de carácter disciplinario o académico.
* Sirve como **guía inicial** para normativa general y siempre entrega **referencias a documentos oficiales**.
* Advierte sobre la **vigencia de documentos**: última actualización del índice en **octubre 2025**.
* Los usuarios son responsables de verificar la normativa actualizada en fuentes oficiales.
* **Privacidad**: no se almacenan consultas de manera permanente; las API externas (OpenAI, DeepSeek) procesan las solicitudes bajo sus propias políticas.

---

## 📑 Tabla de Trazabilidad

| doc_id | Documento                               | URL/Origen           | Páginas | Vigencia  |
| ------ | --------------------------------------- | -------------------- | ------- | --------- |
| 001    | Reglamento de Convivencia Universitaria | [PDF oficial UFRO]   | 1–45    | 2023–2025 |
| 002    | Reglamento de Régimen de Estudios 2023  | [PDF oficial UFRO]   | 1–60    | 2023–2025 |
| 003    | Calendario Académico UFRO 2025          | [Sitio oficial UFRO] | 1–10    | Año 2025  |

*(Los enlaces y fechas deben actualizarse según las fuentes efectivas usadas en `data/sources.csv`).*

---

## 🌐 Interfaz Web

```bash
python web.py
```

Accede desde el navegador en `http://localhost:8081` o desde la IP pública de un servidor en AWS EC2.

---

