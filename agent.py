import os
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from a .env file if present

class GeminiAgent:
    """
    A simple AI Agent that uses the Gemini API's chat service to maintain
    conversational context (memory) and a defined persona.
    """
    def __init__(self, model_name: str = 'gemini-2.5-flash'):
        """
        Initializes the Gemini client and starts a chat session.

        Args:
            model_name: The name of the Gemini model to use.
        """
        # The client will automatically look for the GEMINI_API_KEY
        # environment variable.
        try:
            self.client = genai.Client()
        except Exception as e:
            print("Error initializing Gemini client. Make sure the GEMINI_API_KEY environment variable is set.")
            print(f"Details: {e}")
            self.client = None
            return

        # 1. Define the agent's persona using system instructions
        system_instruction = (
            "You are a friendly and encouraging AI tutor named 'Sage'. "
            "Your goal is to help users learn about climate change. "
            "Keep your answers concise, educational, and positive. "
            "Always try to provide educational and verified scientific information. "
        )

        # 2. Configure the chat session with the persona
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.7
        )

        # 3. Create the chat instance, which handles memory automatically
        self.chat = self.client.chats.create(
            model=model_name,
            config=config
        )
        self.model_name = model_name
        print(f"Agent 'Sage' initialized using model: {self.model_name}")

    def send_message(self, prompt: str) -> str:
        """
        Sends a user message to the agent and returns the response.
        The agent remembers all prior messages in the current session.
        """
        if not self.client:
            return "Agent failed to initialize. Please check your API key setup."

        try:
            # The chat object automatically sends the history with the new prompt
            response = self.chat.send_message(prompt)
            return response.text
        except Exception as e:
            return f"An error occurred during message processing: {e}"

    def get_history(self):
        """Prints the entire conversation history."""
        print("\n--- Conversation History ---")
        for message in self.chat.get_history():
            # Get the text part, handling potential non-text parts gracefully
            text = message.parts[0].text if message.parts and hasattr(message.parts[0], 'text') else "[Non-Text Content]"
            role = "USER" if message.role == "user" else "SAGE"
            print(f"[{role}]: {text}")
        print("----------------------------\n")

def run_agent_cli():
    """Runs the interactive command-line interface for the agent."""
    # Ensure the API key is set before proceeding
    if 'GEMINI_API_KEY' not in os.environ:
        print("CRITICAL: Please set the 'GEMINI_API_KEY' environment variable.")
        print("You can get a key from Google AI Studio.")
        return

    agent = GeminiAgent()

    print("\n------------------------------------------------------")
    print("Welcome! You are chatting with Sage, the Climate Tutor.")
    print("Type 'history' to see the conversation history.")
    print("Type 'quit' or 'exit' to end the session.")
    print("------------------------------------------------------\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                print("\nSage: Goodbye! Stay curious about our planet!")
                break
            
            if user_input.lower() == 'history':
                agent.get_history()
                continue

            if not user_input.strip():
                continue

            response = agent.send_message(user_input)
            print(f"Sage: {response}")

        except KeyboardInterrupt:
            print("\nSage: Goodbye! Stay curious about our planet!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    run_agent_cli()
