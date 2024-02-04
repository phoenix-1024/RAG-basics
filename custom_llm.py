import google.generativeai as genai
import os

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])



class Custom_gemini():
    model = genai.GenerativeModel('gemini-pro')

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
    ) -> str:

        response = self.model.generate_content(prompt)
        return response.text
