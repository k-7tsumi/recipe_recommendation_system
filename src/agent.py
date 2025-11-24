from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import TypedDict, Sequence, Annotated, Literal
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
    ReflectionResult,
)
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.pregel import Pregel
from langgraph.graph import END, START, StateGraph
from langgraph.constants import Send

MAX_CHALLENGE_COUNT = 3

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """計画作成ステップの入力状態"""
    question: str
    plan: list[str]
    current_step: int
    subtask_results: Annotated[Sequence[Subtask], operator.add]
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
    reflection_results: Annotated[Sequence[ReflectionResult], operator.add]
    subtask_answer: str


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
        result = app.invoke(
            {
                "question": question,
                "current_step": 0,
            }
        )

        return AgentResult(
            question=question,
            plan=Plan(subtasks=result["plan"]),
            subtasks=result["subtask_results"],
            answer=result["last_answer"],
        )

    # エージェントのメイングラフを作成する
    def create_graph(self) -> Pregel:
        # Stateを引数としてGraphを初期化
        workflow = StateGraph(AgentState)
        # 計画の作成ノードを追加
        workflow.add_node("create_plan", self.create_plan)
        # 実行ステップのノードを追加
        workflow.add_node("execute_subtasks", self._execute_subgraph)
        # 最終回答の作成ノードを追加
        workflow.add_node("create_last_answer", self.create_last_answer)

        # 計画の作成からスタート
        workflow.add_edge(START, "create_plan")

        # execute_subtasksサブグラフを並列で実行するよう処理
        workflow.add_conditional_edges(
            "create_plan",
            self._should_continue_exec_subtasks,
        )
        workflow.add_edge("execute_subtasks", "create_last_answer")
        workflow.set_finish_point("create_last_answer")

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
        # サブタスク回答作成ノードを追加
        workflow.add_node("create_subtask_answer", self.create_subtask_answer)
        # サブタスク内省の回答ノードを追加
        workflow.add_node("reflect_subtask", self.reflect_subtask)

        # ツール選択からスタート
        workflow.add_edge(START, "select_tools")

        # ノード間のエッジを追加
        workflow.add_edge("select_tools", "execute_tools")
        workflow.add_edge("execute_tools", "create_subtask_answer")
        workflow.add_edge("create_subtask_answer", "reflect_subtask")

        # サブタスク内省の回答ノードの結果から、繰り返し用エッジを追加
        workflow.add_conditional_edges(
            "reflect_subtask",
            self._should_continue_exec_subtask_flow,
            {"continue": "select_tools", "end": END}

        )

        app = workflow.compile()
        return app

    def _execute_subgraph(self, state: AgentState):
        subgraph = self._create_subgraph()

        result = subgraph.invoke(
            {
                "question": state["question"],
                "plan": state["plan"],
                "subtask": state["plan"][state["current_step"]],
                "current_step": state["current_step"],
                "is_completed": False,
                "challenge_count": 0,
            }
        )

        # デバッグ: resultのキーを確認
        logger.info(f"サブグラフ実行結果のキー: {list(result.keys())}")
        for key in result.keys():
            value = result[key]
            value_type = type(value).__name__
            if isinstance(value, list):
                logger.info(f"  {key}: {value_type}(長さ={len(value)})")
            elif isinstance(value, str) and len(value) < 100:
                logger.info(f"  {key}: {value_type} = '{value}'")
            else:
                logger.info(f"  {key}: {value_type}")

        subtask_result = Subtask(
            task_name=result["subtask"],
            tool_results=result["tool_results"],
            reflection_results=result["reflection_results"],
            is_completed=result["is_completed"],
            subtask_answer=result["subtask_answer"],
            challenge_count=result["challenge_count"],
        )

        return {"subtask_results": [subtask_result]}

    # 計画の作成(サブタスクの作成)
    def create_plan(self, state: AgentState) -> dict:
        logger.info("計画の作成処理を開始しました。。。")
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
            logger.info("OpenAIへのリクエスト開始。。。")
            response = self.client.beta.chat.completions.parse(
                model=self.settings.openai_model,
                messages=messages,
                response_format=ReccomendPlan,
                temperature=0,
                seed=0,
            )
            logger.info("OpenAIからの応答を正常に受信しました。")
        except Exception as e:
            logger.error(f"OpenAIリクエスト中にエラーが発生しました: {e}")
            raise

        reccomend_plan = response.choices[0].message.parsed

        # 生成した計画を返し、状態を更新する
        return {"plan": reccomend_plan.subtasks}

    # ツール選択
    def select_tools(self, state: AgentSubGraphState) -> dict:
        logger.info("ツール選択処理を開始しました。。。")
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
        logger.info("ツールの実行を開始しました。。。")
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

    # サブタスク回答を作成する
    def create_subtask_answer(self, state: AgentSubGraphState) -> dict:
        logger.info("サブタスクの回答処理を開始。。。")
        messages = state["messages"]

        try:
            logger.info("OpenAIへのリクエスト開始。。。")
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                temperature=0,
                seed=0,
            )
            logger.info("OpenAIからのレスポンス受け取り完了")
        except Exception as e:
            logger.error(f"OpenAI リクエストエラー： {e}")
            raise

        subtask_answer = response.choices[0].message.content
        ai_message = {"role": "assistant", "content": subtask_answer}
        messages.append(ai_message)

        logger.info("サブタスクの回答の作成完了")

        return {
            "messages": messages,
            "subtask_answer": subtask_answer,
        }

    # サブタスク回答を内省する
    def reflect_subtask(self, state: AgentSubGraphState) -> dict:
        logger.info("サブタスク回答の内省を開始。。。")
        messages = state["messages"]
        user_prompt = self.prompts.subtask_reflection_user_prompt
        messages.append({"role": "user", "content": user_prompt})

        # デバッグ: メッセージの詳細をログ出力
        logger.info(f"メッセージ数: {len(messages)}")
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content")
            content_type = type(content).__name__
            has_tool_calls = "tool_calls" in msg
            has_tool_call_id = "tool_call_id" in msg
            logger.info(
                f"メッセージ[{i}]: role={role}, content_type={content_type}, has_tool_calls={has_tool_calls}, has_tool_call_id={has_tool_call_id}")
            if content is not None and len(str(content)) < 100:
                logger.info(f"  content preview: {content}")

        try:
            logger.info("OpenAIへのリクエストを開始。。。")
            response = self.client.beta.chat.completions.parse(
                model=self.settings.openai_model,
                messages=messages,
                response_format=ReflectionResult,
                temperature=0,
                seed=0,
            )
            logger.info("OpenAIからのレスポンス受け取り完了")
        except Exception as e:
            logger.error(f"OpenAI リクエストエラー： {e}")
            raise

        reflection_result = response.choices[0].message.parsed
        if reflection_result is None:
            raise ValueError("内省の回答がありません")

        messages.append(
            {
                "role": "assistant",
                "content": reflection_result.model_dump_json(),
            }
        )

        update_state = {
            "messages": messages,
            "reflection_results": [reflection_result],
            "challenge_count": state["challenge_count"] + 1,
            "is_completed": reflection_result.is_completed,
        }

        if update_state["challenge_count"] >= MAX_CHALLENGE_COUNT and not reflection_result.is_completed:
            update_state["subtask_answer"] = f"{state['subtask']}の回答が見つかりませんでした。"

        logger.info("サブタスク回答の内省処理完了")
        return update_state

    # 最終回答の作成
    def create_last_answer(self, state: AgentState) -> dict:
        logger.info("最終回答の作成開始。。。")
        system_prompt = self.prompts.create_last_answer_system_prompt

        subtask_results = [(result.task_name, result.subtask_answer)
                           for result in state["subtask_results"]]
        user_prompt = self.prompts.create_last_answer_user_prompt.format(
            question=state["question"],
            plan=state["plan"],
            subtask_results=str(subtask_results)
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            logger.info("OpenAIへのリクエスト開始。。。")
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                temperature=0,
                seed=0,
            )
            logger.info("OpenAIからのレスポンス受け取り完了")
        except Exception as e:
            logger.info(f"OpenAIのリクエストに失敗しました。エラー：{e}")
            raise

        logger.info("最終回答の作成処理が完了しました")

        return {"last_answer": response.choices[0].message.content}

    def _should_continue_exec_subtask_flow(self, state: AgentSubGraphState) -> Literal["end", "continue"]:
        if state["is_completed"] or state["challenge_count"] >= MAX_CHALLENGE_COUNT:
            return "end"
        else:
            return "continue"

    def _should_continue_exec_subtasks(self, state: AgentState) -> list:
        return [
            Send(
                "execute_subtasks",
                {
                    "question": state["question"],
                    "plan": state["plan"],
                    "current_step": idx,
                },
            )
            for idx, _ in enumerate(state["plan"])
        ]
