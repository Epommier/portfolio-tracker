import os
import time
import base64

from typing import List, Literal, Annotated, Optional
from pydantic import BaseModel, Field
from operator import add

from playwright.sync_api import sync_playwright
from rich import print

from langchain.globals import set_verbose
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_openai import AzureChatOpenAI
from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

set_verbose(True)

class Evaluation(BaseModel):
    binary_score: Literal["yes", "no"]
    explanation: str

class RagState(AgentState):
    question: str
    queries: List[str]
    evaluations: Annotated[List[Evaluation], add]

class TokenBalance(BaseModel):
    ticker: str = Field(description="The ticker of the token.")
    amount: float = Field(description="The amount of the token.")
    usd_value: float = Field(description="The USD value of the token.")

class Wallet(BaseModel):
    tokens: Optional[List[TokenBalance]] = Field(description="The tokens that compose the wallet.")

llm = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    temperature=0.0,
    request_timeout=None,
    timeout=None,
    max_retries=3
)

vision_llm = AzureAIChatCompletionsModel(
    endpoint=os.environ["INFERENCE_API_ENDPOINT"],
    api_version=os.environ["INFERENCE_API_VERSION"],
    credential=os.environ["INFERENCE_API_KEY"],
    model_name="Phi-3.5-vision-instruct",
    temperature=0.0
)

json_llm = llm.bind(response_format={"type": "json_object"})

def capture_debank_porfolio(address, output_path):
    with sync_playwright() as p:
        url = f"https://debank.com/profile/{address}"
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.set_viewport_size(viewport_size={ "width": 4880, "height": 2559 })
        time.sleep(10)
        
        # Wallet
        wallet = page.locator("div[class*='TokenWallet_table__']")
        wallet.screenshot(path=f"{output_path}/debank_wallet_{time.time()}.png")

        # Protocols
        protocols = page.locator(f"div[class^='Project_project__']").all()
        for protocol in protocols:
            protocol.screenshot(path=f"{output_path}/debank_protocol_{time.time()}.png")

        browser.close()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        data_string = base64.b64encode(image_file.read()).decode('utf-8')
        padding_len = len(data_string) % 4
        data_string += padding_len * '='
        return data_string

def extract_portfolio_data():
    print(f":runner:[italic green]Extracting portfolio data...[/italic green]")
    #capture_debank_porfolio("0x95c14c058aaffda687780eb60ef8658ef1166578", f"screens/")
    print(f":runner:[italic green]Done[/italic green]")

    print(f"Analyzing wallet data...")
    wallet_image = encode_image(f"screens/debank_wallet.png")
    system_prompt =  """
        # Goal
        Your task is to extract crypto wallet data from the image.
        You should list all the tokens visible on the image, and return them in a list.
        Wrapped tokens and their native representation should appear under their native token name(ie WETH == ETH, WBTC == BTC).
        For each token you should return the name, the amount and the USD value.

        # Rules
        - If no tokens are visible, return an empty list.
        - You should **ONLY** return values extracted from the image, never make up ones.
        - You should **NOT** return any values that are not extracted from the image.

        # Response format
        ```json
        {{
            "tokens": [
                {{
                    "ticker": "BTC",
                    "amount": 0.001,
                    "usd_value": 100000
                }},
                {{
                    "ticker": "ETH",
                    "amount": 1,
                    "usd_value": 2034.45
                }}
            ]
        }}
        ```"""

    prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system_prompt),
        HumanMessagePromptTemplate.from_template(content=[
            #{"type": "text", "text": "Extract the wallet composition from the image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;charset=utf-8;base64,{image_data}"}}
        ]),
    ])

    chain =  prompt | vision_llm | JsonOutputParser(pydantic_object=Wallet)
    wallet_response = chain.invoke({"image_data": wallet_image})
    print(f":runner:[italic green]Done[/italic green]")
    print(wallet_response)

if __name__ == "__main__":
   extract_portfolio_data()