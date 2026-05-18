"""Simple connectivity check for a Pydantic AI agent using Ollama.

This script creates a small Pydantic AI agent connected to a local Ollama model.
It is useful as a smoke test before running more complex RAG examples.

The script demonstrates the following steps:
1. Define a Pydantic AI agent with basic instructions and a model.
2. Run a simple test prompt to check that the agent can generate a response.
3. Optionally, expose the agent as a local web app.
"""

import os
from time import perf_counter

from loguru import logger
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

# Define Model Configuration.
DEFAULT_OLLAMA_MODEL = "qwen3.5:2b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


#################################################################
# STEP 1 - Define the Pydantic AI agent
#################################################################


# Define a helper function to create an Ollama-backed model for the agent.
def get_ollama_model(
    model_name: str | None = None,
    base_url: str | None = None,
) -> OpenAIChatModel:
    """Create a Pydantic AI model backed by the local Ollama OpenAI API.

    Args:
        model_name: Optional name of the Ollama model to use. If not provided, it will be read from the OLLAMA_MODEL environment variable or default to DEFAULT_OLLAMA_MODEL.
        base_url: Optional base URL for the Ollama API. If not provided, it will be read from the OLLAMA_BASE_URL environment variable or default to DEFAULT_OLLAMA_BASE_URL.

    Returns:
        An instance of OpenAIChatModel configured to use the specified Ollama model and API base URL.
    """
    return OpenAIChatModel(
        model_name=model_name or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        provider=OllamaProvider(
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        ),
    )


# Define the Pydantic AI agent with the Ollama model and simple instructions.
agent = Agent(
    model=get_ollama_model(),
    instructions="You are a helpful assistant.",
    model_settings=ModelSettings(thinking="minimal"),
)

#################################################################
# STEP 2 - Run a simple smoke test to check that the agent can generate a response
#################################################################


# Define a simple smoke test function to check that the agent can generate a response.
def smoke_test() -> None:
    """Run a simple test prompt to check that the agent works.

    This function sends a basic question to the model and prints the answer.
    It also measures the response time and logs the approximate number of
    output tokens generated per second.

    Returns:
        None.
    """
    # Start measuring execution time.
    start = perf_counter()

    # Send a simple prompt to the agent.
    result = agent.run_sync("What is the capital of Italy?")

    # Compute the total generation time.
    duration = perf_counter() - start

    # Log an approximate speed value.
    logger.info(
        "Token per second: {:.2f} tokens/s",
        result.usage().output_tokens / duration,
    )

    # Print the model response.
    print(result.output)


# Run the smoke test when the script is executed.
smoke_test()

#################################################################
# STEP 3 - Expose the agent as a local web app
#################################################################

# Uncomment the lines below to expose the agent as a small local web app.

# if __name__ == "__main__":
#     import uvicorn
#
#     app = agent.to_web()
#     uvicorn.run(app, host="127.0.0.1", port=8000)
#     logger.info("Starting Simple Pydantic AI Agent on http://127.0.0.1:8000")
