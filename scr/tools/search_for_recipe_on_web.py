from langchain.tools import tool
from pydantic import BaseModel, Field
from perplexity import Perplexity
from src.config import Settings
import logging


logger = logging.getLogger(__name__)


class SearchQueryInput(BaseModel):
    query: str = Field(description="検索クエリ")


@tool(args_schema=SearchQueryInput)
def search_for_recipe_on_web(query: str) -> str:
    """
    Perplexityを使用してWeb上レシピを検索します。

    Args:
        query: 検索クエリ（例: "簡単なパスタレシピ"）

    Returns:
        str: 検索結果のJSON形式の文字列
    """
    try:
        # 設定を読み込み
        settings = Settings()
        # Perplexityを初期化
        client = Perplexity(api_key=settings.perplexity_api_key)

        logger.info(f"Perplexityで検索を実行中: {query}")

        # Perplexity検索を実行
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": query}
            ],
            model="sonar",
            stream=False,
        )

        logger.info("検索が完了しました。")

        # レスポンスから内容を取得
        content = response.choices[0].message.content

        return content

    except Exception as e:
        logger.error(f"Perplexity検索中にエラーが発生しました: {e}")
        return f"エラーが発生しました: {str(e)}"
