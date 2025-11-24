from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import TypedDict, Sequence, Annotated
import operator
from src.config import Settings
import logging
from src.prompts import RecipeReccomendAgentPrompts
from src.models import (
    ReccomendPlan,
    SearchOutput,
    Subtask,
    ToolResult,
    Plan,
    AgentResult,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.pregel import Pregel
from langgraph.graph import START, StateGraph

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
        self.tool_map = {tool.name: tool for tool in tools}
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.prompts = prompts

    # 　エージェントの実行
    def run_agent(self, question: str) -> AgentResult:
        app = self.create_graph()

        # TODO： 全ての処理追加後正しい値を入れる
        return AgentResult(
            question="",
            plan=Plan(subtasks=[""]),
            subtasks="",
            answer="",
        )

    # エージェントのメイングラフを作成する
    def create_graph(self) -> Pregel:
        # Stateを引数としてGraphを初期化
        workflow = StateGraph(AgentState)
        # 計画の作成ノードを追加
        workflow.add_node("create_plan", self.create_plan)
        # 実行ステップのノードを追加
        workflow.add_node("execute_subtasks", self._execute_subgraph)

        # エッジを追加
        workflow.add_edge(START, "create_plan")

        app = workflow.compile()
        return app

    # サブグラフの作成
    def _create_subgraph(self) -> Pregel:
        # Stateを引数としてGraphを初期化
        workflow = StateGraph(AgentSubGraphState)
        # 　ツールの選択ノードを追加
        workflow.add_node("select_tools", self.select_tools)
        # ツールの実行ノードを追加
        workflow.add_node("execute_tools", self.execute_tools)

        # ノード間のエッジを追加
        workflow.add_edge("select_tools", "execute_tools")

        app = workflow.compile()
        return app

    def _execute_subgraph(self, state: AgentState):
        subgraph = self._create_subgraph()

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

    # ツールの実行
    def execute_tools(self, state: AgentSubGraphState) -> dict:
        logger.info("ツールの実行を開始しました1。。。")
        messages = state["messages"]
        tool_results = []

        # 最後のメッセージからツール呼び出し情報を取得
        tool_calls = messages[-1]["tool_calls"]

        if tool_calls is None:
            logger.info("Error： ツール呼び出し情報(tool_calls)がありません")
            raise ValueError("ツール呼び出し情報(tool_calls)がありません")

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"]["arguments"]

            tool = self.tool_map[tool_name]
            tool_result_str = tool.invoke(tool_args)
            tool_result = SearchOutput(content=tool_result_str)

            tool_results.append(
                ToolResult(
                    tool_name=tool_name,
                    args=tool_args,
                    results=[tool_result],
                )
            )

            messages.append(
                {
                    "role": "tool",
                    "content": str(tool_result),
                    "tool_call_id": tool_call["id"]
                }
            )

        logger.info("ツールの実行が完了しました")
        return {"messages": messages, "tool_results": [tool_results]}
