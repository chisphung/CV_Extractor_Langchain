from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.documents import Document

class CVExtractor:
    def __init__(self, llm):
        self.prompt = PromptTemplate.from_template("""
        Given the following CV text, extract the following information in JSON format:
        - Full Name
        - Email
        - Phone Number
        - Education (school, degree, years)
        - Work Experience (company, title, years, description)
        - Skills
        - Certifications

        CV Text:
        {cv_chunk}

        Return the result as JSON:
        """)
        self.chain = LLMChain(llm=llm, prompt=self.prompt)

    def extract(self, docs: list[Document]):
        extracted = []
        for doc in docs:
            result = self.chain.run({"cv_chunk": doc.page_content})
            extracted.append(result)
        return extracted
