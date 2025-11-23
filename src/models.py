from pydantic import BaseModel, Field


class ReccomendPlan(BaseModel):
    subtasks: list[str] = Field(..., description="レシピを推薦するためのサブタスクリスト")


class SearchOutput(BaseModel):
    content: str = Field(..., description="検索結果")


class ToolResult(BaseModel):
    tool_name: str = Field(..., description="ツールの名前")
    args: str = Field(..., description="ツールの引数")
    results: list[SearchOutput] = Field(..., description="ツールの結果")
