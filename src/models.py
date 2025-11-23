from pydantic import BaseModel, Field


class ReccomendPlan(BaseModel):
    subtasks: list[str] = Field(..., description="レシピを推薦するためのサブタスクリスト")


class SearchOutput(BaseModel):
    file_name: str = Field(..., description="ファイル名")
    content: str = Field(..., description="ファイルの内容")
