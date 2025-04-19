import os
from dataclasses import dataclass
import boto3

ssm = boto3.client("ssm")

@dataclass
class AgenticAssistantConfig:
    bedrock_region: str = ssm.get_parameter(
        Name=os.environ["BEDROCK_REGION_PARAMETER"]
    )["Parameter"]["Value"]

    llm_model_id: str = ssm.get_parameter(Name=os.environ["LLM_MODEL_ID_PARAMETER"])[
        "Parameter"
    ]["Value"]

    chat_message_history_table_name: str = os.environ["CHAT_MESSAGE_HISTORY_TABLE"]