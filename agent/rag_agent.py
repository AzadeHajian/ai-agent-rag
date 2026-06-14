# agent/rag_agent.py
# -----------------------------------------------------------
# RAGAgent — the "brain" of the Docs Assistant.
# Same interface as SQLAgent, but uses RAG_TOOLS (retrieve_docs)
# and the docs-assistant system prompt.
#
# `provider` controls only the CHAT model (Claude or GPT) — the
# embeddings used inside retrieve_docs are always OpenAI
# (see rag/embeddings.py).
# -----------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from llm import get_llm
from tools import RAG_TOOLS
from agent.rag_prompt import get_full_prompt

load_dotenv()


class RAGAgent:
    """
    The Docs Assistant agent.
    Uses LangGraph's create_react_agent under the hood.

    Usage:
        agent = RAGAgent(provider="openai")
        response = agent.run("what does AsyncIteratorCallbackHandler do?")
    """

    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.llm = get_llm(provider)
        self.model_name = self.llm.get_model_name()
        self.tools = RAG_TOOLS
        self.system_prompt = get_full_prompt()

        # Build the LangGraph agent
        self.agent = create_react_agent(
            model=self.llm.client,
            tools=self.tools,
            prompt=self.system_prompt,
        )

    def run(self, user_message: str, chat_history: List[Dict] = []) -> str:
        """
        Send a user message to the agent and get a response.

        Args:
            user_message:  The natural language question from the user.
            chat_history:  Previous messages for multi-turn conversation.

        Returns:
            The agent's final answer as a string.
        """
        try:
            # Build messages list
            messages = []

            # Add chat history
            for m in chat_history:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

            # Add current message
            messages.append(HumanMessage(content=user_message))

            # Run the agent
            response = self.agent.invoke({
                "messages": messages
            })

            # Extract the last message content
            return response["messages"][-1].content

        except Exception as e:
            return f"Agent error: {e}"

    def get_model_name(self) -> str:
        return self.model_name

    def get_provider(self) -> str:
        return self.provider
