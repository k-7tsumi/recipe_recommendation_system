from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import TypedDict, Sequence, Annotated
import operator
from src.config import Settings
import logging
from src.prompts import RecipeReccomendAgentPrompts
from src.models import (
    Subtask,
    ReccomendPlan,
    SearchOutput
)
from langchain_core.utils.function_calling import convert_to_openai_tool

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """計画作成ステップの入力状態"""
    question: str
    plan: list[str]
    current_step: int
    last_answer: str


class AgentSubGraphState(TypedDict):
    """ツール選択・実行ステップの入力状態"""
    question: str
    plan: list[str]
    subtask: str
    is_completed: bool
    messages: list[ChatCompletionMessageParam]
    tool_results: Annotated[Sequence[Sequence[SearchOutput]], operator.add]
    challenge_count: int


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
        # OpenAI対応のtool定義に書き換える
        openai_tools = [convert_to_openai_tool(tool) for tool in self.tools]

        if state["challenge_count"] == 0:
            user_prompt = self.prompts.subtask_tool_selection_user_prompt.format(
                question=state["question"],
                plan=state["plan"],
                subtask=state["subtask"],
            )

            messages = [
                {"role": "system", "content": self.prompts.subtask_system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        else:
            # リトライした場合、過去の対話情報にプロンプトを追加する
            messages: list = state["messages"]
            user_retry_prompt = self.prompts.subtask_retry_answer_user_prompt
            user_message = {"role": "user", "content": user_retry_prompt}
            messages.append(user_message)

        try:
            logger.info("OpenAIにリクエストを送信中...")
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                tools=openai_tools,
                temperature=0,
                seed=0,
            )
            logger.info("OpenAIからの応答を正常に受信しました。")
        except Exception as error:
            logger.error(f"OpenAIリクエスト中にエラーが発生しました: {error}")
            raise

        if response.choices[0].message.tool_calls is None:
            raise ValueError("ツール呼び出しはなし")

        ai_message = {
            "role": "assistant",
            "tool_calls": [tool_call.model_dump() for tool_call in response.choices[0].message.tool_calls],
        }

        logger.info("ツールの選択が完了しました")
        messages.append(ai_message)

        return {"messages": messages}
