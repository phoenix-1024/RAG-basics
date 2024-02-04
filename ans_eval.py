from dotenv import load_dotenv
from google import generativeai as genai
from extract_artical import get_text_from_link




load_dotenv()

genai.configure()

model = genai.GenerativeModel('gemini-pro')

artical_link = input("paste artical link:")

artical_text = get_text_from_link(artical_link)
# print(artical_text)
qa_dict = {
    "question": "What are some of the limitations of Google Gemini, and how do they affect its accessibility and usability?",
    "answer": "Google Gemini has certain limitations, including:\n\n* **Limited accessibility:** Currently, full access to Gemini is restricted to developers and enterprise customers on Google Cloud platforms like Vertex AI and Generative AI Studio. This limits the general public's ability to experience its capabilities.\n\n* **Technical knowledge required:** Using Gemini requires expertise in coding and AI concepts. The complex interfaces and APIs can be intimidating for those without significant technical knowledge.\n\n* **Bias and fairness:** Like any AI model, Gemini can inherit biases from the data it's trained on. Addressing these biases is crucial to ensure fair and ethical use of its outputs.\n\n* **Explainability and transparency:** While Gemini offers explanation capabilities, they might not be perfect or easily interpretable by everyone. The reasoning behind its outputs might still be opaque to some users.\n\n* **Data requirements and computational cost:** Running advanced models like Gemini requires significant computational resources and access to large datasets. This can limit its scalability and accessibility for wider use.\n\n* **Common sense and world knowledge:** Gemini lacks common sense and real-world experience, which can lead to misinterpretations or limitations in tasks requiring such knowledge.\n\n* **Creativity and originality:** Gemini's creations are primarily based on its training data and might struggle with entirely original concepts or ideas not previously encountered.\n\n* **Ethical considerations:** The powerful capabilities of Gemini raise ethical concerns regarding potential misuse or manipulation. Guidelines and safeguards are necessary to ensure responsible development and deployment of such AI models.\n\nThese limitations affect the accessibility and usability of Google Gemini by creating barriers for non-technical users and limiting its scalability and applicability in certain contexts."
}
response = model.generate_content(f"""
You are a smart assistant designed to Judge a users Answer to Questions. 
The answer should be to the point and as detailed as possible.
Rate the users answer on a scale of 1 to 5.
Give the reasoning bhind your rating.

Use the following text as a referance.
----------------
{artical_text}
----------------
Question:{qa_dict['question']}
Answer:{qa_dict['answer']}

""")
print("Rating \n" + response.text)



