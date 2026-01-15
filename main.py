import pandas as pd
from judge import RAGJudge
import json
from dotenv import load_dotenv
import ast


load_dotenv()


# Read the CSV. The provided `golden_set.csv` has a single column named 'data'
# where each row is a stringified Python dict like "{'question': '...', 'ground_truth': '...'}".
# Parse that into a proper DataFrame with `question` and `ground_truth` columns.
raw_df = pd.read_csv("golden_set.csv")

if 'data' in raw_df.columns:
    def safe_parse(s):
        try:
            # ast.literal_eval can handle Python-style dict strings safely
            return ast.literal_eval(s)
        except Exception:
            # Fallback: try JSON loads (if there are double quotes) or return None
            try:
                return json.loads(s)
            except Exception:
                return None

    parsed = raw_df['data'].apply(safe_parse)
    # Drop rows that couldn't be parsed
    parsed = parsed.dropna()
    # Expand list of dicts/Series into a DataFrame
    df = pd.DataFrame(parsed.tolist())
else:
    # If CSV already has proper columns, use it directly
    df = raw_df

judge = RAGJudge()

def my_rag_app(query):
    return "This is my RAG app response"

## Evaluate
results = []
for _, row in df.iterrows():
    # Be resilient to missing keys
    question = row.get('question') if hasattr(row, 'get') else row['question']
    ground_truth = row.get('ground_truth') if hasattr(row, 'get') else row['ground_truth']
    if pd.isna(question) or pd.isna(ground_truth):
        continue

    actual = my_rag_app(question)
    evaluation = judge.score_response(question, actual, ground_truth)
    results.append({
        "question": question,
        "actual": actual,
        "score": evaluation.get('score') if isinstance(evaluation, dict) else None,
        "reason": evaluation.get('reason') if isinstance(evaluation, dict) else str(evaluation)
    })

## Final report
final_df = pd.DataFrame(results)
final_df.to_csv("evaluation_report.csv", index=False)
print("Evaluation report saved to evaluation_report.csv")