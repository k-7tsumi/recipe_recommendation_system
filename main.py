import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
model = init_chat_model(openai_model, model_provider="openai")
messages = [
    SystemMessage(content="入力された日本語を英語に訳してください"),
    HumanMessage(content="今日の夜ご飯は秋刀魚の蒲焼です。"),
]

response = model.invoke(messages)

# レスポンスを表示
print(response)
