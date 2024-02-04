from dotenv import load_dotenv
from google import generativeai as genai
from extract_artical import get_text_from_link

import os.path
from llama_index import (
    VectorStoreIndex,
    Document
)
from llama_index import ServiceContext
from llama_index.llms import Gemini



load_dotenv()

genai.configure()

model = genai.GenerativeModel('gemini-pro')

artical_link = input("past artical link:")

artical_text = get_text_from_link(artical_link)
# print(artical_text)
response = model.generate_content(f"""
You are a smart assistant designed to come up with meaninful question and answer pair. The question should be to the point and the answer should be as detailed as possible.
Given a piece of text, you must come up with a question and answer pair that can be used to evaluate a QA bot. Do not make up stuff. Stick to the text to come up with the question and answer pair.
When coming up with this question/answer pair, you must respond in the following format:
```
{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}}
```


Everything between the ``` must be valid json.


Please come up with a question/answer pair, in the specified JSON format, for the following text:
----------------
{artical_text}

""")
print(response.text)


