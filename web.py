from flask import Flask, request, render_template_string, jsonify
from rag.pipeline import RAGPipeline
from rag.retrieve import Retriever
from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider

app = Flask(__name__)

# Inicializar retriever y providers
retriever = Retriever()
providers = {
    "chatgpt": ChatGPTProvider(),
    "deepseek": DeepSeekProvider()
}

pipelines = {name: RAGPipeline(retriever, p) for name, p in providers.items()}

# HTML simple de chat
HTML_TEMPLATE = """
<!doctype html>
<html>
  <head><title>UFRO Assistant</title></head>
  <body>
    <h1>UFRO Assistant</h1>
    <form method="POST" action="/chat">
      <label>Pregunta:</label><br>
      <input type="text" name="question" size="80"><br><br>
      <label>Proveedor:</label>
      <select name="provider">
        <option value="chatgpt">ChatGPT</option>
        <option value="deepseek">DeepSeek</option>
      </select><br><br>
      <input type="submit" value="Enviar">
    </form>
    {% if answer %}
    <h2>Respuesta ({{ provider }})</h2>
    <p>{{ answer }}</p>
    <h3>Referencias</h3>
    <pre>{{ references }}</pre>
    {% endif %}
  </body>
</html>
"""

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        question = request.form.get("question")
        provider = request.form.get("provider", "chatgpt")
        pipeline = pipelines.get(provider)
        res = pipeline.synthesize(question, top_k=4)
        return render_template_string(
            HTML_TEMPLATE,
            answer=res.get("answer", ""),
            references="\n".join(res.get("citations", [])),
            provider=provider
        )
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    question = data.get("question")
    provider = data.get("provider", "chatgpt")
    pipeline = pipelines.get(provider)
    res = pipeline.synthesize(question, top_k=4)
    return jsonify(res)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
