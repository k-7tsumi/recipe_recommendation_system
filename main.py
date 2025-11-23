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
    input_data_for_plan = {"question": question}
    # 計画の作成
    plan_result = agent.create_plan(state=input_data_for_plan)
    print("**計画**")
    print(plan_result["plan"])

    # すべてのサブタスクに対してツール選択と実行を繰り返す
    all_tool_results = []
    for i, subtask in enumerate(plan_result["plan"]):
        print(f"\n**サブタスク {i+1}/{len(plan_result['plan'])}: {subtask}**")

        # サブタスクごとの入力データを作成
        input_data_for_select_tools = {
            "question": question,
            "plan": plan_result["plan"],
            "subtask": subtask,
            "challenge_count": 0,
            "is_completed": False,
        }

        # ツールの選択
        select_tool_result = agent.select_tools(
            state=input_data_for_select_tools)
        print("**ツールの選択**")
        print(select_tool_result)

        # ツールの実行
        input_data_for_execute_tools = {
            "question": question,
            "plan": plan_result["plan"],
            "subtask": subtask,
            "challenge_count": 0,
            "messages": select_tool_result["messages"],
            "is_completed": False,
        }
        tool_results = agent.execute_tools(state=input_data_for_execute_tools)
        print("**ツールの実行**")
        print(tool_results)

        all_tool_results.append(tool_results)

    print("\n**すべてのサブタスクが完了しました**")
    print(f"処理したサブタスク数: {len(all_tool_results)}")
