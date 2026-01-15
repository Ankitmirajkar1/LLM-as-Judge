import json
from groq import Groq
from dotenv import load_dotenv


load_dotenv()
class RAGJudge:
    def __init__(self):
        self.client = Groq()
    
    def score_response(self, question, actual_answer, ground_truth):
        evaluation_prompt = f"""
        Rate the 'Actual answer' against the 'Ground truth' on a scale of 1 to 10, where 10 means the actual answer perfectly matches the ground truth, and 1 means it does not match at all.
        - Metrics used to evaluate:
    1. Correctness/Accuracy
    2. Relevance
    3. Completeness
    4. Faithfulness

    Question: {question},
    Ground Truth: {ground_truth},
    Actual Answer: {actual_answer}

    Return JSON: {{"score": 1-10, "reason": "..."}}
    """
        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": evaluation_prompt}
            ],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)