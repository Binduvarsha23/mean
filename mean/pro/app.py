from flask import Flask, render_template, request, jsonify
import pdfplumber
from transformers import pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
qa_model = pipeline("text2text-generation", model="t5-large")

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "pdf" not in request.files or "question" not in request.form:
        return jsonify({"error": "Missing PDF file or question"}), 400

    pdf_file = request.files["pdf"]
    question = request.form["question"]

    # Extract text from PDF
    document = extract_text_from_pdf(pdf_file)

    # Prompt for T5 model
    prompt = (
        f"Based on the document, answer the following question and provide an in-depth explanation "
        f"with context without including the question and prompt: '{question}' Document: {document}"
    )

    # Generate response
    result = qa_model(prompt, max_length=200, do_sample=True)
    answer = result[0]['generated_text']

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
