# agent/agent.py
# -----------------------------------------------------------
# The brain of the project.
# Uses LangGraph's create_react_agent (modern approach).
# LangChain v1.0 removed AgentExecutor in favor of LangGraph.
# -----------------------------------------------------------

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent

from llm import get_llm
from tools import SUPABASE_TOOLS
from agent.prompt import get_full_prompt

load_dotenv()


class SQLAgent:
    """
    The main agent class.
    Uses LangGraph's create_react_agent under the hood.

    Usage:
        agent = SQLAgent(provider="anthropic")
        response = agent.run("show me all users")
    """

    def __init__(self, provider: str = "anthropic"):
        self.provider = provider
        self.llm = get_llm(provider)
        self.model_name = self.llm.get_model_name()
        self.tools = SUPABASE_TOOLS
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