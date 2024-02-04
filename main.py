from dotenv import load_dotenv
from google import generativeai as genai
from extract_artical import get_text_from_link

import os
from llama_index import (
    VectorStoreIndex,
    Document
)
from llama_index import ServiceContext
from llama_index.llms import Gemini


service_context = ServiceContext.from_defaults(llm=Gemini(api_key=os.environ['GOOGLE_API_KEY']),embed_model="local:sentence-transformers/all-MiniLM-L12-v2")



load_dotenv()

genai.configure()

model = genai.GenerativeModel('gemini-pro')

artical_link = input("past artical link:")

artical_text = get_text_from_link(artical_link)
# print(artical_text)
response = model.generate_content(f"""summrize the following: 
'''
{artical_text}
'''
""")
print("Artical summery: \n" + response.text)


documents = [Document(text= artical_text)]
index = VectorStoreIndex.from_documents(documents,service_context=service_context)
query_engine = index.as_query_engine(service_context=service_context)

while True:
    query = input("Q: ")
    response = query_engine.query(query)
    print(response)
