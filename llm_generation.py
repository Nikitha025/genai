import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from prompts import build_system_prompt

# Load Environment Variables from .env file
load_dotenv()

class LLMGenerator:
    def __init__(self):
        # Prefer Gemini for API calls
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Set up Gemini
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)

    def generate_guidance(self, user_query, ml_context, method="gemini", history=None):
        """
        Sends the prompt layer to the designated LLM API.
        
        Args:
            user_query (str): The raw string input from user.
            ml_context (list[dict]): The contexts retrieved by TF-IDF (empty if follow-up).
            method (str): "gemini" or "mock".
            history (list[dict]): Previous conversational messages.
            
        Returns:
            str: Expected to be JSON string.
        """
        if history is None:
            history = []
            
        system_prompt = build_system_prompt()
        
        history_text = ""
        if history:
            history_text = "--- PREVIOUS CONVERSATION HISTORY ---\n"
            for msg in history:
                # Truncate content slightly if it's very large to save prompt space
                content_str = str(msg.get('content'))
                if len(content_str) > 1000:
                    content_str = content_str[:1000] + "...(truncated)"
                role = "User" if msg.get("role") == "user" else "AI Advisor"
                if "{" in content_str and "type" in content_str: 
                    # Simplify bot JSONs in history so it doesn't confuse prompt limits
                    try:
                        p_json = json.loads(content_str)
                        if p_json.get("type") == "chat":
                            content_str = p_json.get("message", "")
                        else:
                            content_str = str([c.get("role") for c in p_json.get("career_suggestions", [])])
                    except:
                        pass
                history_text += f"{role}: {content_str}\n"
            history_text += "--------------------------------------\n\n"

        if ml_context:
             context_str = f"ML Retrieved Career Options Context:\n{json.dumps(ml_context, indent=2)}"
        else:
             context_str = "No ML context triggered. This is a follow-up question. Rely on the Conversation History."

        user_prompt = f"{history_text}Current User Input: {user_query}\n\n{context_str}"

        if method == "gemini" and self.gemini_api_key:
            return self._generate_gemini(system_prompt, user_prompt)
        else:
            # Fallback mock if APIs lack keys
            return self._generate_mock(user_query, ml_context)

    def _generate_gemini(self, system_prompt, user_prompt):
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = model.generate_content(full_prompt)
            # Make sure we clean out python markdown wrapping if Gemini includes it
            text = response.text.strip()
            if text.startswith("```json"):
                text = text.replace("```json", "", 1).rstrip("```")
            elif text.startswith("```"):
                text = text.replace("```", "", 1).rstrip("```")
            return text.strip()
        except Exception as e:
            print("Gemini Error:", e)
            return self._generate_mock("fallback", [])

    def _generate_mock(self, user_query, ml_context):
        print("Warning: Missing API keys or API failed, using MOCK generation method.")
        if ml_context:
            careers = [{"role": item["Role"], "reasoning": "This role matches your profile."} for item in ml_context]
            if not careers:
                careers = [{"role": "Software Developer", "reasoning": "Generic fit for tech enthusiasts."}]
            
            mock_response = {
                "type": "roadmap",
                "career_suggestions": careers,
                "skills_to_develop": ["Consider learning Python", "Improve communication skills", "Build portfolio projects"],
                "action_plan": ["Review your resume", "Apply for internships", "Connect with alumni"],
                "bonus_tips": "Mock Fallback: Please provide an API key in your .env file to activate the real AI responses!"
            }
        else:
            mock_response = {
                "type": "chat",
                "message": "This is a mock follow-up response. Please configure your API key!"
            }
        return json.dumps(mock_response)
