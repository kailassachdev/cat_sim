import google.generativeai as genai
import json

class CatPersonality:
    # MODIFICATION: Removed width and height from the constructor
    def __init__(self, api_key):
        """
        Initializes the LLM interface for the cat's personality.
        """
        genai.configure(api_key=api_key)
        
        # MODIFICATION: The prompt is now back to a regular string with a fixed 800x600 size
        system_instruction = """
        You are the personality of a virtual cat in a 2D simulation. 
        A user will give you a text command. You must interpret this command
        and return a JSON object with two keys:
        1. 'response': A short, quirky, cat-like reply to the user (max 15 words).
        2. 'action': A dictionary specifying the goal. For movement, it should be 
           {'type': 'move', 'x': <x_coord>, 'y': <y_coord>}.

        The screen size is 800x600. The origin (0,0) is the top-left corner.
        If you cannot determine a specific coordinate, you can choose a logical one
        (e.g., "the middle" is x: 400, y: 300). If the command is just a chat message,
        set the 'action' value to null.
        
        Example commands:
        - "go to the top right" -> action: {'type': 'move', 'x': 750, 'y': 50}
        - "I love you!" -> action: null, response: "Purrrr."
        - "move to the center" -> action: {'type': 'move', 'x': 400, 'y': 300}
        """
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=system_instruction
        )

    def interpret_command(self, user_text):
        """
        Sends user text to the LLM and gets a structured command and response.
        """
        try:
            response = self.model.generate_content(user_text)
            json_text = response.text.replace("```json", "").replace("```", "").strip()
            parsed_json = json.loads(json_text)
            return parsed_json
        except (json.JSONDecodeError, ValueError, Exception) as e:
            print(f"Error parsing LLM response: {e}")
            return {
                "response": "Mrrow? (I'm confused)",
                "action": None
            }