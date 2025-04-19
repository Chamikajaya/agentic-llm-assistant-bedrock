import math
from typing import Optional, Type

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    # This string will contain the mathematical expression the agent wants evaluated
    question: str = Field()


def _evaluate_expression(expression: str) -> str:
    import numexpr  # rather than using eval,  to avoid security issues using numexpr

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
    except Exception as e:
        raise ValueError(
            f'LLMMathChain._evaluate("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )

    return output.strip()


class CustomCalculatorTool(BaseTool):  # inherit from BaseTool
    name = "Calculator"
    description = "useful for when you need to answer questions about math"  #  natural language description tells the agent what the tool does and when to use it
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        try:
            return _evaluate_expression(query.strip())
        except Exception as e:
            return (
                f"Failed to evaluate the expression with error {e}."
                " Please only provide a valid math expression."
            )

  