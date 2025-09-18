import openai
import os
from typing import List, Dict, Any
import streamlit as st

class AgenticLLMSystem:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the agent with API credentials and model choice
        """
        if api_key:
            openai.api_key = api_key
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
        self.max_memory = 3  # Store last 3 user-assistant pairs
        self.api_key_valid = bool(api_key)
        
    def detect_intent(self, user_input: str) -> str:
        """
        Detect whether the user input is a factual query or creative prompt
        """
        if not self.api_key_valid:
            # Simple rule-based intent detection as fallback
            creative_keywords = ["write", "create", "generate", "story", "poem", "caption", "joke", "imagine"]
            if any(keyword in user_input.lower() for keyword in creative_keywords):
                return "creative"
            return "factual"
        
        try:
            # Prepare context with conversation history
            messages = self.conversation_history.copy()
            messages.append({"role": "user", "content": f"""
            Classify this input as 'factual' or 'creative': "{user_input}"
            Respond with ONLY one word.
            """})
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=10,
                temperature=0.1
            )
            
            intent = response.choices[0].message.content.strip().lower()
            return intent if intent in ["factual", "creative"] else "factual"
            
        except Exception as e:
            # Fallback to rule-based detection
            creative_keywords = ["write", "create", "generate", "story", "poem", "caption", "joke", "imagine"]
            if any(keyword in user_input.lower() for keyword in creative_keywords):
                return "creative"
            return "factual"
    
    def handle_factual_query(self, query: str) -> str:
        """
        Handle factual queries
        """
        if not self.api_key_valid:
            # Fallback responses without API
            responses = {
                "api full form": "API stands for Application Programming Interface.",
                "what is api": "An API (Application Programming Interface) is a set of rules that allows different software applications to communicate with each other.",
                "who is the ceo of google": "Sundar Pichai is the CEO of Google.",
                "who is the ceo of openai": "Sam Altman is the CEO of OpenAI.",
                "what is the capital of france": "Paris is the capital of France.",
                "who won 2024 t20 world cup": "India won the 2024 T20 World Cup.",
                "who is modi": "Narendra Modi is the Prime Minister of India.",
            }
            
            query_lower = query.lower().strip()
            for pattern, response in responses.items():
                if pattern in query_lower:
                    return response
            
            return "I don't have that information in my local knowledge base. Please check online sources for the most current information."
        
        try:
            # Use conversation history for context
            messages = self.conversation_history.copy()
            messages.append({"role": "user", "content": f"""
            Provide a comprehensive answer to: "{query}"
            Be accurate and informative.
            """})
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=200,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"I encountered an error: {e}"
    
    def handle_creative_prompt(self, prompt: str) -> str:
        """
        Handle creative prompts
        """
        if not self.api_key_valid:
            # Simple fallback creative responses
            if "caption" in prompt.lower():
                return "A beautiful moment captured forever."
            elif "poem" in prompt.lower():
                return "Roses are red, Violets are blue, This is a poem, Just for you."
            elif "joke" in prompt.lower():
                return "Why don't scientists trust atoms? Because they make up everything!"
            else:
                return "That's an interesting creative request! I'd need API access to generate a proper response."
        
        try:
            # Use conversation history for context
            messages = self.conversation_history.copy()
            messages.append({"role": "user", "content": f"""
            Create a creative response to: "{prompt}"
            """})
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"I encountered an error: {e}"
    
    def update_memory(self, user_input: str, agent_response: str):
        """
        Maintain conversation history - store last 3 user-assistant pairs
        """
        # Add new conversation pair
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": agent_response})
        
        # Keep only the last max_memory * 2 messages (each interaction has user + assistant)
        if len(self.conversation_history) > self.max_memory * 2:
            self.conversation_history = self.conversation_history[-(self.max_memory * 2):]
    
    def get_conversation_history(self) -> str:
        """
        Return the conversation history in a readable format
        """
        if not self.conversation_history:
            return "No conversation history yet."
        
        history_text = "Our conversation history:\n"
        for i in range(0, len(self.conversation_history), 2):
            if i + 1 < len(self.conversation_history):
                user_msg = self.conversation_history[i]['content']
                assistant_msg = self.conversation_history[i + 1]['content']
                history_text += f"\n{i//2 + 1}. You: {user_msg}\n   Me: {assistant_msg}\n"
        
        return history_text
    
    def process_input(self, user_input: str) -> str:
        """
        Main method to process user input
        """
        # Check for special commands first
        user_input_lower = user_input.lower()
        
        if any(cmd in user_input_lower for cmd in ["clear memory", "forget", "reset"]):
            self.conversation_history = []
            return "Conversation history cleared! Let's start fresh."
        
        if any(cmd in user_input_lower for cmd in ["history", "previous", "what did i ask", "conversation history"]):
            return self.get_conversation_history()
        
        # Detect intent
        intent = self.detect_intent(user_input)
        
        # Handle based on intent
        if intent == "factual":
            response = self.handle_factual_query(user_input)
        else:
            response = self.handle_creative_prompt(user_input)
        
        # Update memory
        self.update_memory(user_input, response)
        
        return response