from pydantic import BaseModel, Field


class ReccomendPlan(BaseModel):
    subtasks: list[str] = Field(..., description="レシピを推薦するためのサブタスクリスト")


class SearchOutput(BaseModel):
    content: str = Field(..., description="検索結果")


class ToolResult(BaseModel):
    tool_name: str = Field(..., description="ツールの名前")
    args: str = Field(..., description="ツールの引数")
    results: list[SearchOutput] = Field(..., description="ツールの結果")


class Plan(BaseModel):
    subtasks: list[str] = Field(..., description="問題を解決するためのサブタスクリスト")


class ReflectionResult(BaseModel):
    advice: str = Field(
        ...,
        description="評価がNGの場合は、別のツールを試す、別の文言でツールを試すなど、なぜNGなのかとどうしたら改善できるかを考えアドバイスを作成してください。\
アドバイスの内容は過去のアドバイスと計画内の他のサブタスクと重複しないようにしてください。\
アドバイスの内容をもとにツール選択・実行からやり直します。",
    )
    is_completed: bool = Field(
        ...,
        description="ツールの実行結果と回答から、サブタスクに対して正しく回答できているかの評価結果",
    )


class Subtask(BaseModel):
    task_name: str = Field(..., description="サブタスクの名前")
    tool_results: list[list[ToolResult]] = Field(..., description="サブタスクの結果")
    reflection_results: list[ReflectionResult] = Field(
        ..., description="サブタスクの評価結果"
    )
    is_completed: bool = Field(..., description="サブタスクが完了しているかどうか")
    subtask_answer: str = Field(..., description="サブタスクの回答")
    challenge_count: int = Field(..., description="サブタスクの挑戦回数")


class AgentResult(BaseModel):
    question: str = Field(..., description="ユーザーの元の質問")
    plan: Plan = Field(..., description="エージェントの計画")
    subtasks: list[Subtask] = Field(..., description="サブタスクのリスト")
    answer: str = Field(..., description="最終的な回答")
