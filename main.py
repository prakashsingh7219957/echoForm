from flask import Flask, request, jsonify
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import ollama
import json
from ollama import Client
import os

# ✅ Correct base URL
ollama_client = Client(host="https://sanguinarily-unprolongable-katy.ngrok-free.dev:11434")

app = Flask(__name__)

data = [
    {"name": "Aarti Mehta", "company": "SheJobs", "city": "Mumbai", "category": "Recruitment", "email": "aarti@shejobs.in","price":3000,"discount":"10%","delivery":"instore"},
    {"name": "Prakash Singh", "company": "Sitha", "city": "Bangalore", "category": "Tech Services", "email": "prakash@sitha.com","price":200,"discount":"15%","delivery":"at-dore-step"},
    {"name": "Rohan Verma", "company": "Medicare+", "city": "Delhi", "category": "Healthcare", "email": "rohan@medicare.com","price":1500,"discount":"18%","delivery":" "}
]
df = pd.DataFrame(data)

texts = df.apply(lambda x: f"{x['name']} from {x['company']} in {x['city']} offering {x['category']}", axis=1).tolist()
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = FAISS.from_texts(texts, embedding=embeddings)

def generate_form_data(user_text):
    results = vector_store.similarity_search(user_text, k=3)
    context = "\n".join([r.page_content for r in results])

    prompt = f"""
    You are a form-filling assistant.
    User input: "{user_text}"
    Here are similar past provider records:
    {context}

    Based on these, extract or infer:
    name, company, city, category, email, price, discount, delivery.
    Return valid JSON.
    """

    response = ollama_client.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response['message']['content']

    try:
        data = json.loads(content)
    except:
        data = {"raw_output": content}
    return data

@app.route("/fill_form", methods=["POST"])
def fill_form():
    text = request.json.get("text", "")
    result = generate_form_data(text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
#    app.run(host="0.0.0.0", port=5000, debug=True)
