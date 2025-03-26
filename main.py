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

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

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
    #model_name="Phi-3.5-vision-instruct",
    #model_name="Phi-4-multimodal-instruct",
    model_name="Azure AI Vision",
    temperature=0.0,
    top_p=1
)

json_llm = llm.bind(response_format={"type": "json_object"})

def capture_debank_porfolio(address, output_path):
    with sync_playwright() as p:
        url = f"https://debank.com/profile/{address}"
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.route("**/*.{png,jpg,jpeg,svg}", lambda route: route.abort())

        page.goto(url)
        page.set_viewport_size(viewport_size={ "width": 4880, "height": 2559 })
        time.sleep(10)
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        # Wallet
        wallet = page.locator("div[class*='TokenWallet_table__']")
        wallet.screenshot(animations="disabled", type="jpeg", quality=100, path=f"{output_path}/debank_wallet_{timestamp}.jpeg")

        # Protocols
        protocols = page.locator("div[class^='Project_project__']").all()
        for protocol in protocols:
            name = protocol.locator("span[class^='ProjectTitle_protocolLink__']").first
            protocol_name = name.text_content()
            protocol.screenshot(animations="disabled", type="jpeg", quality=100, path=f"{output_path}/debank_protocol_{protocol_name}_{timestamp}.jpeg")

        browser.close()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path):
    with open(image_path, "rb") as img:
        image_data = img.read()

        client = ImageAnalysisClient(
            endpoint=os.environ["VISION_API_ENDPOINT"],
            credential=AzureKeyCredential(os.environ["INFERENCE_API_KEY"])
        )

        result = client.analyze(
            image_data,
            visual_features=[VisualFeatures.READ]
        )

        if result.read is not None:
            words = result.read.as_dict()["blocks"][0]["lines"]
            lines = [words[n:n+4] for n in range(0, len(words), 4)]

            for line in lines:
                full_line = ""
                for word in line:
                    full_line += word["text"] + " "
                print(full_line)
    
        return result;

def extract_portfolio_data():
    print(f":runner:[italic green]Extracting portfolio data...[/italic green]")
    #capture_debank_porfolio("0x95c14c058aaffda687780eb60ef8658ef1166578", f"screens/")
    print(f":runner:[italic green]Done[/italic green]")
    print(f"Analyzing wallet data...")

    ocr_data = analyze_image(f"screens/debank_wallet.jpeg")
    print(ocr_data)

    wallet_image = encode_image(f"screens/debank_wallet.jpeg")
    system_prompt =  """
        # Goal
        Your task is to extract text data from the image.
        You should list all the assets visible on the image, and return them in a list.       
        For each asset you should return the name, the amount and the USD value you see on the image.
        Note that the amount and dollar values are in US format, meaning the [,] character is the thousands separator (ie 29,700 is 29 700).
        Be aware that there is also some numbers in scientific format.

        # Rules
        - If no assets are visible, return an empty list.
        - **ONLY** return numbers and texts you found on the image
        - Numbers should be in decimal format for example "11,456.28" should returned as "11456.28" and "27,000" as "27000"

        # Response format
        See the example below for your expected response format:

        ```json
        {{
            "tokens": [
                {{
                    "ticker": "[token_1_name]",
                    "amount": [token_1_amount],
                    "usd_value": [token_1_usd_value]
                }},
                {{
                    "ticker": "[token_2_name]",
                    "amount": [token_2_amount],
                    "usd_value": [token_2_usd_value]
                }}
            ]
        }}
        ```
        """

    prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=[
            {"type": "text", "text": system_prompt},
        ]),
        HumanMessagePromptTemplate.from_template(template=[
            ##{"type": "text", "text": "Extract the wallet composition from the image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,{image_data}"}}
        ]),
    ])

    chain =  prompt | vision_llm | JsonOutputParser(pydantic_object=Wallet)
    wallet_response = chain.invoke({"image_data": wallet_image})
    print(f":runner:[italic green]Done[/italic green]")
    print(wallet_response)

if __name__ == "__main__":
   extract_portfolio_data()