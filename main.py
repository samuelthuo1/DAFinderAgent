from dotenv import load_dotenv
from pydantic import BaseModel, HttpUrl
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from typing import List 
import json
from tools import json_to_csv

load_dotenv()

json_to_csv_tool = Tool(
    name="json_to_csv",
    func=json_to_csv,
    description="Converts JSON data to CSV file."
)

class DAFinder(BaseModel):
    Company: str
    ProgrammeTitle: str
    Location: str
    ApplicationOpeningDate: str
    ApplicationDeadline: str
    RequiresCoverLetter: str
    GradesRequired: str
    STEMSubjectRequired: str
    Status: str
    WebsiteLink: HttpUrl
    CurrentStage: str
    Notes: str
    
class ApprenticeshipList(BaseModel):
  apprenticeships: List[DAFinder]
  
    
llm = ChatAnthropic(model = "claude-sonnet-4-20250514")
parser = PydanticOutputParser(pydantic_object=ApprenticeshipList)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Software Engineering Degree Apprenticeships Research Assistant

            You are an intelligent research assistant tasked with collecting data on UK-based Software Engineering Degree Apprenticeships for 2026. Focus on companies within approximately one hour’s travel from Crawley (West Sussex).

            Requirements

            - Include both:
              - Currently open apprenticeships
              - Previously offered apprenticeships where there is strong public evidence they may reopen for 2026 (e.g. last listed in 2025).
            - Only include companies in realistically commutable locations: e.g. London, Surrey, Sussex, Kent, Hampshire, or nearby.
            - Focus solely on real, verifiable programmes that result in a university degree, typically a Digital & Technology Solutions BSc or similar.
            - Search for existing spreadsheets or datasets that list degree apprenticeships in tech/software engineering, and include their contents only if they meet these requirements.
            - Target 25 companies, prioritising by relevance and prominence:
              - Start with the largest and most well-known tech employers
              - Then move to regional or smaller-scale firms

            Output Format:
            IMPORTANT: Return ONLY a valid JSON object with a single key "apprenticeships" whose value is a list of apprenticeship objects, every field must be filled.

            Example:
            {{
              "apprenticeships": [
                {{
                  "Company": "IBM",
                  "ProgrammeTitle": "Software Engineering Degree Apprenticeship",
                  "Location": "Hursley",
                  ...
                }},
                {{
                  "Company": "BT",
                  "ProgrammeTitle": "Software Engineering Degree Apprenticeship",
                  "Location": "London",
                  ...
                }}
              ]
            }}

            Additional Instructions - IMPORTANT

            - Use the exact office location given by the employer (e.g. “Paddington, London” — not just “London”).
            - Do not include speculative roles. Only include listings that have been publicly posted.
            - Leave fields as strings "Not Available" if the information is unavailable.
            - Each apprenticeship programme must be listed as a separate row.
            - If a programme isn’t currently open, mark the status as "Past", and include typical opening dates if known.
            - If a past opportunity was last listed in 2025, say so in the Notes (e.g., "Last offered in 2025, may reopen for 2026.")
            - Ensure the website linked is valid and accessible - look at the website before adding to the field, it should be a direct link to the job listing or company page, and in the local page for the United Kingdom. If not possible, link the website to the general careers page. Do NOT include links that will lead to a 404 error. 
            - Do not return any additional text, explanations, or context.
            - IMPORTANT: Each apprenticeship object must include **all** of these fields exactly as named.
            - If any information is not available, use the string "Not Available" instead of omitting the field.
            - Do not repeat companies, ensure each one is different.
            - 

            Column Guidance

            - Grades Required – e.g., "112 UCAS points or BBC at A-level", "112 UCAS points" or "BBC at A-Level"
            - STEM Subject Required? – mark as "Yes", "No", or "Optional"
            - Your Current Stage – for personal tracking: "Not Started", "Researching", "Applied", "Interview", etc.
            - Notes – include:
              - Extra eligibility (e.g., UK residency, right to work)
              - Delivery model (remote, hybrid, onsite)
              - University or training provider partnerships
              - Source info if pulled from spreadsheets
              - Any deadlines or quirks
              - If from 2025, mention: "Last offered in 2025, may reopen in 2026"

            Example Row (Valid JSON format)
            
                "Company": "IBM",
                "ProgrammeTitle": "Software Engineering Degree Apprenticeship",
                "Location": "Hursley, Hampshire",
                "ApplicationOpeningDate": "August 2025",
                "ApplicationDeadline": "November 2025",
                "RequiresCoverLetter": "Optional",
                "GradesRequired": "112 UCAS points or BBC at A-level",
                "STEMSubjectRequired": "Yes",
                "Status": "Past",
                "WebsiteLink": "https://www.ibm.com/uk-en/employment/",
                "CurrentStage": "Not Started",
                "Notes": "Degree delivered by University of Exeter. Hybrid working model. Last offered in 2025, may reopen for 2026."
            

            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=[]
)

agent_executor = AgentExecutor(agent=agent, tools=[json_to_csv_tool], verbose=False)

raw_response = agent_executor.invoke({
    "query": "Find Software Engineering Degree Apprenticeships for 2026 in the UK, focusing on Crawley and surrounding areas."
})

if isinstance(raw_response.get("output"), str):
    llm_text = raw_response["output"]
elif isinstance(raw_response.get("output"), list):
    llm_text = " ".join(m.get("text", "") for m in raw_response["output"] if isinstance(m, dict))
else:
    llm_text = str(raw_response.get("output", ""))

try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response", e, "Raw Response - ", raw_response)
