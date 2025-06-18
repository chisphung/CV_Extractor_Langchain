from typing import List
import re
import json
from langchain_core.output_parsers import StrOutputParser


def recursive_extract(text, pattern, default_answer):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        assistant_text = match.group(1).strip()
        return recursive_extract(assistant_text, pattern, assistant_text)
    else:
        return default_answer


class Str_OutputParser(StrOutputParser):
    def __init__(self) -> None:
        super().__init__()
    
    def parse(self, text: str) -> str:
        return self.extract_answer(text)
    
    
    def extract_answer(self,
                       text_response: str,
                       patterns: List[str] = [r'\nAssistant:(.*)', r'\nAI:(.*)'],
                       default = "Sorry, I am not sure how to help with that."
                       ) -> str:

        input_text = text_response
        default_answer = default
        for pattern in patterns:
            output_text = recursive_extract(input_text, pattern, default_answer)
            if output_text != default_answer:
                input_text = output_text
                default_answer = output_text
        
        return output_text

def json_clean(raw_blocks):
    cleaned_results = []
    
    for block in raw_blocks:
        json_str = re.sub(r"^```json\s*|\s*```$", "", block.strip(), flags=re.MULTILINE)
        
        try:
            data = json.loads(json_str)
            cleaned_results.append(data)
        except json.JSONDecodeError as e:
            print(f"Failed to parse: {e}")
    
    return cleaned_results