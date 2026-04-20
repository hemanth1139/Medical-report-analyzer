import json
from google import genai
from google.genai import types
from utils.knowledge_base import retrieve
from config import GEMINI_MODEL, CHAT_SYSTEM_PROMPT

analyze_report_tool = types.FunctionDeclaration(
    name="analyze_report",
    description="Analyze the uploaded medical report and return structured findings, risk level, and recommendations",
)

search_kb_tool = types.FunctionDeclaration(
    name="search_knowledge_base",
    description="Search the medical knowledge base to answer questions about test values, diseases, or medical terms",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(type=types.Type.STRING, description="The medical question or term to search for")
        },
        required=["query"]
    )
)

def run_agent(user_message, report_content, analysis, risk_level, kb_index, kb_facts, api_key):
    client = genai.Client(api_key=api_key)
    tools = [types.Tool(function_declarations=[analyze_report_tool, search_kb_tool])]
    context = f"Report Analysis:\n{json.dumps(analysis, indent=2)}\nRisk Level: {risk_level}"
    messages = [
        types.Content(role="user", parts=[types.Part.from_text(text=f"{CHAT_SYSTEM_PROMPT}\n{context}")]),
        types.Content(role="user", parts=[types.Part.from_text(text=user_message)])
    ]
    
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=messages,
        config=types.GenerateContentConfig(tools=tools, temperature=0.2)
    )
    
    if resp.candidates and resp.candidates[0].content.parts and resp.candidates[0].content.parts[0].function_call:
        fc = resp.candidates[0].content.parts[0].function_call
        if fc.name == "analyze_report":
            result = f"Analysis retrieved: {json.dumps(analysis)}"
        elif fc.name == "search_knowledge_base":
            query = fc.args["query"]
            chunks = retrieve(query, kb_index, kb_facts)
            result = "Knowledge Base Results:\n" + "\n".join(chunks)
        else:
            result = "Unknown tool."
            
        messages.append(resp.candidates[0].content)
        messages.append(types.Content(role="user", parts=[
            types.Part.from_function_response(name=fc.name, response={"result": result})
        ]))
        
        final_resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=messages,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        return final_resp.text
        
    return resp.text
