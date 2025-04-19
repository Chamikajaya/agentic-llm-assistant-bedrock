import boto3
from langchain.agents import Tool
from langchain_aws import BedrockLLM
from langchain_aws import ChatBedrock
from langchain_community.tools import DuckDuckGoSearchRun
from .calculator import CustomCalculatorTool
from .config import AgenticAssistantConfig

config = AgenticAssistantConfig()
bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.bedrock_region)


# ! TODO: Move the 2 clients to the config file since they are used here and handler.py as well
claude_llm = BedrockLLM(
    provider=config.llm_model_provider, 
    model_id=config.llm_model_id,
    client=bedrock_runtime,
    model_kwargs={
        "max_tokens_to_sample": 1000,
        "temperature": 0.0,
        "top_p": 0.99
    },
)

claude_chat_llm = ChatBedrock(
    provider=config.llm_model_provider, 
    model_id=config.llm_model_id,
    client=bedrock_runtime,
    model_kwargs={
        "max_tokens": 1000,
        "temperature": 0.0,
        "top_p": 0.99
    },
)

search = DuckDuckGoSearchRun()
custom_calculator = CustomCalculatorTool()

LLM_AGENT_TOOLS = [
    Tool(
        name="WebSearch",
        func=search.invoke,
        description=(
            "Use this tool to search for information on current events, news, or general knowledge topics. "
            "For example, you can use this tool to find information about recent news events, famous people, or common facts."
        ),
    ),
    Tool(
        name="Calculator",
        func=custom_calculator,
        description=(
            "Use this tool when you need to perform mathematical calculations. "
            "The input to this tool should be a valid mathematical expression, such as '55/3' or '(10 + 20) * 5'."
        ),
    ),
]