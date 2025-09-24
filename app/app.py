from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import os, json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
# Mandatory model and config
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")


app = FastAPI(title="OSS Note 1804812 Assessment & Remediation Prompt")

# --- SNIPPET HELPER ---
def snippet_at(text: str, start: int, end: int) -> str:
    s = max(0, start - 60)
    e = min(len(text), end + 60)
    return text[s:e].replace("\n", "\\n")

# ---- Strict input models ----
class select_item(BaseModel):
    table: str
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: Optional[str] = None
    snippet: Optional[str] = None 

    @field_validator("used_fields", "suggested_fields", mode="before")
    @classmethod
    def no_none(cls, v):
        return [x for x in v if x]

class NoteContext(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    code: Optional[str] = ""
    mb_txn_usage: List[select_item] = Field(default_factory=list)


# ---- Planner summary ----
def summarize_context(ctx: NoteContext) -> dict:
    return {
        "unit_program": ctx.pgm_name,
        "unit_include": ctx.inc_name,
        "unit_type": ctx.type,
        "unit_name": ctx.name,
        "mb_txn_usage": [item.model_dump() for item in ctx.mb_txn_usage]
    }

# ---- LangChain prompt ----
SYSTEM_MSG = """You are a senior ABAP expert. Output ONLY JSON as response.
You are an ABAP upgrade advisor. Output ONLY valid JSON as response.
- For every provided .mb_txn_usage[] with a non-empty "suggested_statement":
  - Write a bullet point using ONLY the "suggested_statement" field as the corrective action.
  - If "snippet" is non-empty, insert it (as ABAP code/text) before/after the suggested_statement where it fits.
  - Do not reference or require code outside "snippet".
  - Omit mb_txn_usage without a suggested_statement.
- Cover ALL mb_txn_usage with suggested_statement.
Return JSON (and nothing else) with:
{{
  "assessment": "<summary of  issues>",
  "llm_prompt": "<bulleted list with all actions as described above>"
}}
""".strip()

USER_TEMPLATE = """
You are evaluating a system context related to SAP OSS Note 1804812 (MB* obsolescence). We provide:
- system context
Instructions:
1. Write a summary ("assessment").
2. For every finding containing a non-empty suggested_statement, add a bullet in "llm_prompt":
    - Use the "suggested_statement" field as the action text.
    - Do NOT include any "snippet" content in the output (use it only as background to refine the bullets).
    - No Bullet point should be exact duplicate.
    - Each Bullet point Should clearly explains , each action Item.
    - Skip any findings without a suggested_statement.
Return valid JSON:
{{
  "assessment": "<concise impact paragraph>",
  "llm_prompt": "<bullet list of actionable suggestions>"
}}

Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {type}
- Unit name: {name}

System context:
{context_json}
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE),
])

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
parser = JsonOutputParser()
chain = prompt | llm | parser

def llm_assess(ctx: NoteContext):
    ctx_json = json.dumps(summarize_context(ctx), ensure_ascii=False, indent=2)
    return chain.invoke({"context_json": ctx_json,"pgm_name": ctx.pgm_name,
    "inc_name": ctx.inc_name,
    "type": ctx.type,
    "name": ctx.name})

@app.post("/assess-1804812")
async def assess_note_context(ctxs: List[NoteContext]):
    results = []
    for ctx in ctxs:
        try:
            llm_result = llm_assess(ctx)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

        results.append({
            "pgm_name": ctx.pgm_name,
            "inc_name": ctx.inc_name,
            "type": ctx.type,
            "name": ctx.name,
            "code": "",  # Assuming no actual ABAP code is passed/analyzed in this API
            "assessment": llm_result.get("assessment", ""),
            "llm_prompt": llm_result.get("llm_prompt", "")
        })

    return results

@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}