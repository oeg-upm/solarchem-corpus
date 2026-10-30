from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
import argparse
from typing import List
import time
from pydantic import BaseModel, Field


query = """
Please help me analysis if there are mentions in the provided text. There are 7 different types of mentions. 

catalyst: The catalyst used in the experiment, often is a chemical compound
co-catalyst: The co-catalyst used in the experiment, often is a chemical compound
light source: The type of light source used in the experiment. Possible choices are UV, Solar, UV-Vis, Monochromatic, Solar Simulator
lamp: The type of lamp used in the experiment. Possible choices are Fluorescent, Mercury, 'Halogen', 'Mercury-Xenon', 'LED', 'Tungsten', 'Xenon', 'Tungsten-Halide', 'Solar Simulator'
reaction medium: The type of the medium of the reactor used in the experiment. Possible choices are 'Liquid', 'Gas'
reactor type: The type of the reactor, possible choices are 'Slurry', 'Fixed-bed', 'Optical Fiber', 'Monolithic', 'Membrane', 'Fluidised-bed'
operation mode: The mode of operation in the experiment. Possible choices are 'Batch', 'Continuous', 'Batch/Continuous'

Please help me analysis if there are mentions in the provided text. If so, please indicate the answer for each mention in the provided text. And please indicate which sentence is the source of the mention as well. 

If there are any of these types are mentioned, please answer the question as type, answer and source from the context.
"""

class Answer(BaseModel):
    category: str = Field(description="The category of this answer")
    answer: str = Field(description="The actual selection or inference of the answer")
    source: str = Field(description="The original source of the answer")

class Answers(BaseModel):
    analysis: str = Field(description="The thinking process")
    answers: List[Answer] = Field(description="list of answers which contains the category, answer and source context of the answer")

class SolarAnno():
    def __init__(self, user_key, file_dir, model_id):
        os.environ["GOOGLE_API_KEY"] = "AIzaSyD17BOuVn_T6_grylA4ZUnsmYLJc3lUldE"
        self.file_dir = file_dir
        self.parser = PydanticOutputParser(pydantic_object=Answers)
        self.llm = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\nContext: {context}",
            input_variables=["query", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )
        self.chain = prompt | self.llm
        
    def get_data(self):
        if self.file_dir[-4:] == "json":
            f = open(self.file_dir, "rb")
            data = json.load(f)
        else:
            data = None
        return data
    
    def execute(self, data):
        if data == None:
            raise TypeError("Bro, data misssssssed")
        res = {}
        for item in data:
            if item["title"] == "Doi":
                res["paper_doi"] = item["content"]
                doi = item["content"]
            elif item["title"] == "Article_Title":
                res["paper_title"] = item["content"]
        res["paragraphs"] = []
        for item in data:
            if item["title"] in ["Abstract", "Experimental", "Results_and_Discussion", "Conclusions"]:
                # print(f"Paragraph for Annotation: {item["title"]}")
                print(item["title"])
                temp = {
                    "paragraph_text": item["content"],
                    "annotations": []
                }
                answers = self.chain.invoke({"query": query, "context": item["content"]})
                print(answers)
                try:
                    parser_answers = self.parser.parse(answers.content)
                    print("paerser")
                    print(parser_answers.answers)
                    for answer in parser_answers.answers:
                        temp_answer = dict(answer)
                        temp_answer["annotator"] = "hybrid annotation"
                        temp["annotations"].append(temp_answer)
                    print("Annotation is Done")
                except:
                    temp_answer = {}
                    print("Annotation Failed")
                    temp["annotations"].append(temp_answer)
                res["paragraphs"].append(temp)
                time.sleep(30)
        return res

    

                
def get_parser():
    parser = argparse.ArgumentParser(description="Solar Annotation Pipeline")
    parser.add_argument('--user_key', default="xxx", help="Gemini Key", type=str)
    parser.add_argument('--model_id', default = "gemini-2.0-flash", help="gemini ai model reference", type=str)
    parser.add_argument('--file_dir', help="the location of the file being annotated", type=str)
    parser.add_argument('--save_file_dir', default="None", help="gemini-2.0-flash", type=str)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    save_file_dir = args_dict["save_file_dir"]
    del args_dict["save_file_dir"]
    annotator = SolarAnno(**args_dict)
    data = annotator.get_data()
    res = annotator.execute(data)
    if save_file_dir == "None":
        pass
    else:
        with open(save_file_dir, "w") as f:
            json.dump(res, f)

        
main()
