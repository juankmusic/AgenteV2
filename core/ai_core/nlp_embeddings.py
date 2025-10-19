# core/ai_core/nlp_embeddings.py
import os
from openai import OpenAI as OpenAIClient
from dotenv import load_dotenv

# Cargar las variables de entorno 
load_dotenv()

# Claves de API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Clientes de API
openai_client = OpenAIClient(api_key=OPENAI_API_KEY)
deepseek_client = OpenAIClient(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com/v1")

# --------------------------------------------------------------
# FUNCIÓN PARA GENERAR EMBEDDINGS CON OPENAI
# --------------------------------------------------------------
def generate_embedding(text: str) -> list:
    try:
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []
# --------------------------------------------------------------
# FUNCIÓN PARA EXTRAER ENTIDADES CLAVE USANDO DEEPSEEK
# --------------------------------------------------------------
def extract_entities(text: str) -> dict:
    """
    Utiliza el modelo de DeepSeek para detectar entidades o conceptos
    relevantes en el texto del usuario: nombres, fechas, áreas, virtudes, etc.
    """
    try:
        prompt = f"""
        Extrae las entidades relevantes del siguiente texto.
        Clasifícalas en categorías: persona, área, virtud, periodo, acción.
        Devuelve la respuesta en formato JSON válido, sin usar bloques de código ni texto adicional.

        Texto:
        {text}
        """

        res = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un extractor semántico experto en análisis de texto. Devuelve siempre JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        raw_output = res.choices[0].message.content.strip()

        # 🔹 Limpieza del formato (elimina ```json, ``` y \n extraños)
        clean_output = (
            raw_output.replace("```json", "")
                      .replace("```", "")
                      .replace("\\n", "")
                      .replace("\n", "")
                      .strip()
        )

        # 🔹 Intentar convertir a JSON
        import json
        try:
            entities = json.loads(clean_output)
        except json.JSONDecodeError:
            entities = {"texto": text, "entidades_raw": clean_output}

        return entities

    except Exception as e:
        print(f"❌ Error extrayendo entidades: {e}")
        return {"error": str(e)}

# --------------------------------------------------------------
# FUNCIÓN PRINCIPAL DE ANÁLISIS NLP COMPLETO
# --------------------------------------------------------------
def analyze_text(text: str) -> dict:
    """
    Procesa un texto: genera embedding + extrae entidades.
    Devuelve un diccionario estructurado con la información.
    """
    print(f"🧠 Analizando texto: {text[:60]}...")

    embedding = generate_embedding(text)
    entities = extract_entities(text)

    return {
        "texto": text,
        "embedding": embedding,
        "entidades": entities
    }
