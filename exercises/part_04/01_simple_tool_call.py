"""Simple tool-calling exercise for the Cheese & Yodel workshop."""

from datetime import datetime

import uvicorn
from loguru import logger
from pydantic_ai import Agent

from graph_rag_workshop.utils.pydantic_utils import get_ollama_model

agent = Agent(
    model=get_ollama_model(),
    instructions=(
        "You are a helpful assistant that can perform tasks using tools. "
        "Use the provided tools when they are useful for the user's question."
    ),
)


# EXERCISE - Tool definition:
# Define a function that returns the current date and time as a string.
# Decorate it with @agent.tool_plain so Pydantic AI can call it as a tool.
# Tip: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
...


if __name__ == "__main__":
    app = agent.to_web()
    logger.info("Starting Simple Tool Call Agent on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
