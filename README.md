# レシピレコメンドシステム

LangChain、LangGraph と OpenAI を活用した AI エージェントベースのレシピ推薦システムです。ユーザーの質問を分析し、サブタスクに分解して計画的に回答を生成します。

## 概要

このシステムは、以下の特徴を持つインテリジェントなレシピ推薦エージェントです：

- **計画ベースの実行**: ユーザーの質問を複数のサブタスクに分解し、段階的に処理
- **Web 検索統合**: Perplexity API を使用して最新のレシピ情報を Web 上から取得
- **反復的な改善**: サブタスクの結果を評価し、必要に応じて再実行
- **LangChain & LangGraph 活用**: LangChain でツール定義と LLM 連携を行い、LangGraph で複雑なワークフローを状態管理とともに実行

## 技術スタック

- **Python**: 3.12.9 以上
- **LangChain**: LLM アプリケーション開発フレームワーク（ツール定義、OpenAI 連携）
- **LangGraph**: エージェントのワークフロー管理
- **OpenAI API**: 大規模言語モデル（LLM）
- **Perplexity API**: Web 検索
- **Pydantic**: データバリデーションと設定管理

## プロジェクト構造

```
recipe_recommendation_system/
├── main.py                      # アプリケーションのエントリーポイント
├── src/
│   ├── agent.py                 # メインのエージェントロジック
│   ├── config.py                # 設定管理
│   ├── models.py                # データモデル（Pydantic）
│   ├── prompts.py               # プロンプトテンプレート
│   └── tools/
│       └── search_for_recipe_on_web.py  # Perplexity検索ツール
├── pyproject.toml               # プロジェクト依存関係
├── .env.sample                  # 環境変数のサンプル
└── README.md                    # このファイル
```

## 環境構築

### 1. 前提条件

- Python 3.12.9 以上
- [uv](https://github.com/astral-sh/uv) (Python パッケージマネージャー)

uv のインストール：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. リポジトリのクローン

```bash
git clone <repository-url>
cd recipe_recommendation_system
```

### 3. 環境変数の設定

`.env.sample`をコピーして`.env`ファイルを作成し、必要な API キーを設定します：

```bash
cp .env.sample .env
```

`.env`ファイルを編集：

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
PERPLEXITY_API_KEY=your_perplexity_api_key
```

必要な API キーの取得先：

- [OpenAI API](https://platform.openai.com/api-keys)
- [Perplexity API](https://www.perplexity.ai/)

### 4. 依存関係のインストール

```bash
uv sync
```

## 使い方

### 方法 1: パイプで入力を渡す

```bash
echo "おすすめのレシピを教えて" | uv run python main.py
```

### 方法 2: 対話的に実行

```bash
uv run python main.py
```

実行後、プロンプトが表示されるので質問を入力してください：

```
質問を入力してください: おすすめのレシピを教えて
```

## AI エージェントのアーキテクチャ

このシステムは、LangGraph を使用した階層的なグラフ構造で実装された AI エージェントです。計画（Plan）→ 実行（Execute）→ 評価（Reflect）のサイクルを通じて、複雑な質問に対して段階的に回答を構築します。

### アーキテクチャの全体像

```
ユーザーの質問
    ↓
[メイングラフ]
    ├─ 計画作成 (create_plan)
    │   └─ 質問を複数のサブタスクに分解
    │
    ├─ サブタスク実行 (execute_subtasks) ※並列実行
    │   └─ 各サブタスクごとにサブグラフを実行
    │       │
    │       [サブグラフ] ※反復実行可能
    │       ├─ ツール選択 (select_tools)
    │       ├─ ツール実行 (execute_tools)
    │       ├─ 回答作成 (create_subtask_answer)
    │       └─ 内省 (reflect_subtask)
    │           ├─ OK → 次のサブタスクへ
    │           └─ NG → ツール選択に戻る（最大3回）
    │
    └─ 最終回答作成 (create_last_answer)
        └─ 全サブタスクの結果を統合
```

### メイングラフの構造

メイングラフは、エージェント全体のワークフローを管理します。

#### 1. 計画作成 (create_plan)

- **役割**: ユーザーの質問を分析し、解決に必要な複数のサブタスクに分解
- **処理内容**:
  - OpenAI の LLM を使用してタスク分解
  - Structured Outputs で構造化されたサブタスクリストを生成
- **出力**: サブタスクのリスト（例: ["簡単なレシピの定義を確認する", "インターネットで簡単なレシピを探す", ...]）

#### 2. サブタスク実行 (execute_subtasks)

- **役割**: 各サブタスクを並列で実行
- **処理内容**:
  - LangGraph の `Send` API を使用して、複数のサブタスクを並列処理
  - 各サブタスクに対してサブグラフを起動
- **特徴**: 効率的な並列実行により、複数のサブタスクを同時に処理

#### 3. 最終回答作成 (create_last_answer)

- **役割**: すべてのサブタスク結果を統合し、ユーザーに返す最終回答を生成
- **処理内容**:
  - 各サブタスクの回答を収集
  - OpenAI の LLM を使用して、一貫性のある自然な回答に統合
- **出力**: ユーザーに提示する最終的なレシピ推薦

### サブグラフの構造（反復的改善サイクル）

各サブタスクは独立したサブグラフとして実行され、品質が満たされるまで反復的に改善されます。

#### 1. ツール選択 (select_tools)

- **役割**: サブタスクを達成するために適切なツールを選択
- **処理内容**:
  - OpenAI の Function Calling を使用
  - 利用可能なツール（`search_for_recipe_on_web` など）から最適なものを選択
  - 複数のツールを同時に選択可能
- **出力**: 実行するツールとその引数

#### 2. ツール実行 (execute_tools)

- **役割**: 選択されたツールを実際に実行
- **処理内容**:
  - Perplexity API を使用して Web 検索を実行
  - 検索結果を構造化されたデータとして保存
- **出力**: ツールの実行結果（検索結果）

#### 3. サブタスク回答作成 (create_subtask_answer)

- **役割**: ツールの実行結果を基に、サブタスクに対する回答を生成
- **処理内容**:
  - OpenAI の LLM を使用
  - ツールの結果を解釈し、サブタスクの質問に対する回答を作成
- **出力**: サブタスクの回答

#### 4. 内省（リフレクション）(reflect_subtask)

- **役割**: 生成された回答の品質を評価し、改善が必要か判断
- **処理内容**:
  - Structured Outputs を使用して評価結果を取得
  - 評価項目: サブタスクが正しく達成されているか
  - 改善アドバイスの生成（NG の場合）
- **出力**:
  - `is_completed`: true（完了）/ false（要改善）
  - `advice`: 改善のためのアドバイス
- **フロー制御**:
  - `is_completed = true` → サブグラフ終了
  - `is_completed = false` → ツール選択に戻る（最大 3 回まで）
  - 3 回試行しても完了しない場合 → 「回答が見つかりませんでした」として終了

### 状態管理

#### AgentState（メイングラフの状態）

```python
{
    "question": str,              # ユーザーの質問
    "plan": list[str],            # サブタスクのリスト
    "current_step": int,          # 現在のステップ
    "last_answer": str,           # 最終回答
    "subtask_results": list[Subtask]  # 各サブタスクの結果（累積）
}
```

#### AgentSubGraphState（サブグラフの状態）

```python
{
    "question": str,                # ユーザーの質問
    "plan": list[str],              # サブタスクのリスト
    "subtask": str,                 # 現在のサブタスク
    "is_completed": bool,           # サブタスク完了フラグ
    "messages": list[Message],      # 会話履歴（累積）
    "tool_results": list[ToolResult],  # ツール実行結果（累積）
    "challenge_count": int,         # 試行回数
    "subtask_answer": str,          # サブタスクの回答
    "reflection_results": list[ReflectionResult]  # 内省結果（累積）
}
```

### 反復的改善の仕組み

このエージェントの特徴は、**内省（リフレクション）による反復的改善**です：

1. **初回実行**: ツール選択 → 実行 → 回答作成 → 内省
2. **品質チェック**: 内省ノードで回答の品質を評価
3. **改善サイクル**:
   - 品質が不十分な場合、改善アドバイスを生成
   - アドバイスを基に、異なるツールや検索クエリで再試行
   - 最大 3 回まで自動的に改善を試みる
4. **完了条件**:
   - 内省で `is_completed = true` が返される
   - または、3 回の試行上限に達する

この仕組みにより、単純な一発実行ではなく、**自己評価と改善を繰り返すことで高品質な回答を生成**します。

### 並列実行の最適化

メイングラフでは、LangGraph の `Send` API を活用して**複数のサブタスクを並列実行**します。これにより：

- 各サブタスクが独立して処理される
- 待ち時間が削減され、全体の実行時間が短縮される
- サブタスク間の依存関係がないため、効率的に処理可能

例えば、4 つのサブタスクがある場合、逐次実行では 4 倍の時間がかかりますが、並列実行により大幅に高速化されます。

## 開発

### ログの確認

エージェントの実行ログは標準出力に出力されます。ログレベルを変更する場合は、`src/agent.py`で設定を調整してください。
