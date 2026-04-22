from flask import Flask, request, jsonify, render_template, session
import json
import logging
from model import CareerRetriever, is_followup
from llm_generation import LLMGenerator
import os

app = Flask(__name__)
# Secret key is required to use Flask sessions securely
app.secret_key = "career-bot-super-secret-key"

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize classes
try:
    retriever = CareerRetriever(data_path="data/careers.csv")
    llm_gen = LLMGenerator()
    logging.info("Initialized Retriever and LLM layers successfully.")
except Exception as e:
    logging.error(f"Error initializing modules: {e}")

@app.route("/")
def home():
    """
    Renders the Frontend UI and resets session
    """
    session.clear()
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    Endpoint that receives User Input, orchestrates ML retrieval,
    builds the prompt, gets LLM response, and returns the result.
    """
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        
        if not user_input:
            return jsonify({"error": "No input provided"}), 400
            
        method = data.get("llm_method", "gemini") 
        if os.getenv("GEMINI_API_KEY"):
            method = "gemini"
        else:
            method = "mock"

        # 1. Initialize Server Side Memory
        if "history" not in session:
            session["history"] = []
        
        history = session["history"]

        # 2. Check follow-up logic
        follow_up = is_followup(user_input, history)
        
        retrieved_careers = []
        if follow_up:
            logging.info("Detected as FOLLOW-UP. Skipping ML Retrieval.")
        else:
            logging.info("Detected as NEW QUERY. Running ML Retrieval...")
            # If they explicitly restart, clear the history.
            session["history"] = []
            history = session["history"]
            retrieved_careers = retriever.retrieve_careers(user_input, top_n=3)

        # 3. Generation using LLM
        json_response_str = llm_gen.generate_guidance(user_input, retrieved_careers, method=method, history=history)
        
        # Parse the JSON returned by LLM
        try:
            response_data = json.loads(json_response_str)
        except json.JSONDecodeError:
            logging.error(f"LLM output invalid JSON: {json_response_str}")
            return jsonify({
                "error": "The AI provided an invalid format response.",
                "raw": json_response_str
            }), 500

        # 4. Append interaction to backend session properly
        history.append({"role": "user", "content": user_input})
        history.append({"role": "bot", "content": json.dumps(response_data)})
        
        # Limit history to last 10 messages (5 pairs) to prevent payload maxing
        session["history"] = history[-10:]

        return jsonify({
            "status": "success",
            "data": response_data,
            "method_used": method,
            "is_followup": follow_up
        })

    except Exception as e:
        logging.error(f"Error processing chat: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
