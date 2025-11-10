from openai import OpenAI
from typing import TypedDict
from src.config import Settings
from src.prompts import RecipeReccomendAgentPrompts
from src.models import ReccomendPlan


class AgentState(TypedDict):
    question: str
    plan: list[str]


class AgentSubGraphState(TypedDict):


class RecipeReccomendAgent:
    def __init__(
        self,
        settings: Settings,
        tools: list = [],
        prompts: RecipeReccomendAgentPrompts = RecipeReccomendAgentPrompts(),
    ) -> None:
        self.tools = tools
        self.tool_maps = {tool.name: tool for tool in tools}
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.prompts = prompts

    # 計画の作成(サブタスクの作成)
    def create_plan(self, state: AgentState) -> dict:
        system_prompt = self.prompts.recipe_curator_system_prompt
        user_prompt = self.prompts.recipe_curator_user_prompt.format(
            question=state["question"],
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # OpenAIへリクエスト
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.settings.openai_model,
                messages=messages,
                response_format=ReccomendPlan,
                temperature=0,
                seed=0,
            )
        except Exception as e:
            raise

        reccomend_plan = response.choices[0].message.parsed

        # 生成した計画を返し、状態を更新する
        return {"plan": reccomend_plan.subtasks}

    # ツール選択
    def select_tools(self, state: AgentSubGraphState) -> dict:
