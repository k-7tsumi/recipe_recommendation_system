from src.agent import RecipeReccomendAgent
from src.config import Settings

if __name__ == "__main__":
    settings = Settings()
    agent = RecipeReccomendAgent(
        settings=settings,
        tools=[],
    )
    question = input("質問を入力してください: ")
    input_data = {"question": question}
    # 計画の作成
    plan_result = agent.create_plan(state=input_data)
    print(plan_result["plan"])
