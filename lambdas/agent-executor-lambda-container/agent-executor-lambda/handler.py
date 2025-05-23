import logging
import traceback
import boto3
from langchain.agents import AgentExecutor, create_react_agent
from assistant.prompts import CLAUDE_AGENT_PROMPT
from assistant.tools import LLM_AGENT_TOOLS
from langchain.chains import ConversationChain
from langchain_aws import ChatBedrock
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
from assistant.config import AgenticAssistantConfig
from assistant.prompts import CLAUDE_PROMPT
from assistant.utils import parse_markdown_content

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ssm = boto3.client("ssm")
config = AgenticAssistantConfig()
bedrock_runtime = boto3.client("bedrock-runtime", region_name=config.bedrock_region)



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

def get_basic_chatbot_conversation_chain(
    session_id, clean_history, verbose=True
):
    message_history = DynamoDBChatMessageHistory(
        table_name=config.chat_message_history_table_name, session_id=session_id
    )

    if clean_history:
        message_history.clear()

    memory = ConversationBufferMemory(
        memory_key="history",
        chat_memory=message_history,
        human_prefix="Hu",
        return_messages=False
    )

    
    conversation_chain = ConversationChain(
        prompt=CLAUDE_PROMPT, llm=claude_chat_llm, verbose=verbose, memory=memory
    )

    return conversation_chain


def get_agentic_chatbot_conversation_chain(
    session_id, clean_history, verbose=True
): 
    message_history = DynamoDBChatMessageHistory(
        table_name=config.chat_message_history_table_name, session_id=session_id
    )

    if clean_history:
        message_history.clear()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=message_history,
        human_prefix="Hu",
        return_messages=False
    )

    agent = create_react_agent(
        llm=claude_chat_llm,
        tools=LLM_AGENT_TOOLS,
        prompt=CLAUDE_AGENT_PROMPT,

    )

    agent_chain = AgentExecutor(
        agent=agent,
        tools=LLM_AGENT_TOOLS,
        return_intermediate_steps=False,
        verbose=verbose,
        memory=memory,
        handle_parsing_errors="Check your output and make sure it conforms.",
    )
    
    return agent_chain

# Main entry point for the Lambda function
def lambda_handler(event, context):
    
    logger.info(event)
    user_input = event["user_input"]
    session_id = event["session_id"]
    chatbot_type = event.get("chatbot_type", "basic")
    clean_history = event.get("clean_history", False)

    if chatbot_type == "basic":
        conversation_chain = get_basic_chatbot_conversation_chain(
            session_id, clean_history
        ).invoke  
    elif chatbot_type == "agentic":
        conversation_chain = get_agentic_chatbot_conversation_chain(
            session_id, clean_history
        ).invoke
    else:
        return {
            "statusCode": 400,
            "response": (
                f"The chatbot_type {chatbot_type} is not supported."
                f" Please use chatbot_type: 'basic'"
            ),
        }

    try:
        response = conversation_chain({"input": user_input})

        if chatbot_type == "basic":
            response = response["response"]
            response = parse_markdown_content(response)
        elif chatbot_type == "agentic":
            response = response["output"]
            
    except Exception:
        response = (
            "Unable to respond due to an internal issue. Please try again later"
        )
        print(traceback.format_exc())

    return {"statusCode": 200, "response": response}