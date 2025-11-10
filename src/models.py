from pydantic import BaseModel, Field


class ReccomendPlan(BaseModel):
    subtasks: list[str] = Field(..., description="レシピを推薦するためのサブタスクリスト")
