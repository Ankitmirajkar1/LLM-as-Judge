import os
import pandas as pd
from groq import Groq
from pypdf import PdfReader
from dotenv import load_dotenv


load_dotenv()
client= Groq()

def get_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def generate_golden_set(context):
    prompt = f"""
    You are a data generation model. Given the following context, generate 5 high quality examples of a golden set of data.
    Format as JSON {{"data": [{{"question":"...", "ground_truth":"..."}}] }}
    Text : {context}
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", 
             "content": prompt}
        ],
        response_format={"type": "json_object"}
    )

    return pd.read_json(response.choices[0].message.content)

text = get_pdf_text("Rainbow-Bazaar-Return-Refund-&-Cancellation-Policy.pdf")
golden_df = generate_golden_set(text)
golden_df.to_csv("golden_set.csv", index=False)
print("Golden set data generated and saved to golden_set.csv")
