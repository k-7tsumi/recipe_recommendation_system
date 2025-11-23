from src.agent import RecipeReccomendAgent
from src.config import Settings
from src.tools.search_for_recipe_on_web import search_for_recipe_on_web

if __name__ == "__main__":
    def _input_data_for_select_tools(question: str, plan_result: dict) -> dict:
        return {
            "question": question,
            "plan": plan_result["plan"],
            "subtask": plan_result["plan"][0],
            "challenge_count": 0,
            "is_completed": False,
        }

    # 設定の読み込み
    settings = Settings()
    agent = RecipeReccomendAgent(
        settings=settings,
        tools=[search_for_recipe_on_web],
    )
    question = input("質問を入力してください: ")
    input_data_for_plan = {"question": question}
    # 計画の作成
    plan_result = agent.create_plan(state=input_data_for_plan)
    print("計画")
    print(plan_result["plan"])

    input_data_for_select_tools = _input_data_for_select_tools(
        question, plan_result)

    # ツールの選択
    select_tool_result = agent.select_tools(state=input_data_for_select_tools)
    print("ツールの選択")
    print(select_tool_result)
