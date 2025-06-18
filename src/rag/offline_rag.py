from queue import Full
import re
from tokenize import Name
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.prompts import PromptTemplate


class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()
    
    def parse(self, text: str) -> str:
        return self.extract_answer(text)
    
    
    def extract_answer(self,
                       text_response: str,
                       pattern: str = r"Answer:\s*(.*)"
                       ) -> str:
        
        match = re.search(pattern, text_response, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            return answer_text
        else:
            return text_response


class Offline_RAG:
    def __init__(self, llm) -> None:
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            """Extract the following information from this CV text:

            {context}

            Return in JSON format containing:
            - Full Name
            - Email and Contact
            - Education (institution, degree, year)
            - Work Experience (company, role, duration, achievements)
            - Skills
            - Certifications"""
        )
        self.str_parser = Str_OutputParser()

    
    def get_chain(self, retriever):
        chain = RunnableMap({
            "context": retriever | RunnableLambda(self.format_docs),
            "question": RunnablePassthrough()
        }) | self.prompt | self.llm | self.str_parser
        return chain

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)