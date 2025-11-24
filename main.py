from src.agent import RecipeReccomendAgent
from src.config import Settings
from src.tools.search_for_recipe_on_web import search_for_recipe_on_web

if __name__ == "__main__":
    # 設定の読み込み
    settings = Settings()
    agent = RecipeReccomendAgent(
        settings=settings,
        tools=[search_for_recipe_on_web],
    )
    question = input("質問を入力してください: ")
    result = agent.run_agent(question)
    print(result.answer)
