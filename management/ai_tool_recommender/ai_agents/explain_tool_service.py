"""Service for generating tool explanations."""

import json

from langchain_openai import ChatOpenAI


async def generate_tool_explanation(json_object, query):
    """Generate a detailed explanation of the configuration."""
    prompt_template = """
You are an AI assistant helping a non-technical user understand and implement a personalized AI and automation workflow. The user has described a challenge or task they want to solve, and you've received a set of recommended tools (from Agent 1) in a structured JSON format.
Your job is to:
- Explain what each tool is, in simple human terms
- Show how to use each one to solve the user's challenge
- Connect the tools into a clear step-by-step workflow
- Ensure clarity by filling in any missing or unclear info using trusted online sources
Query to solve: {query}
Tool Data (JSON):
{json_object}
Instructions:
1. For each tool listed in the "nodes" array:
   - Refer to the tool by its **label (name)**, not ID or “node”.
   - Briefly describe what this tool is for (in layman's terms).
   - If the description is vague or missing, **simulate looking it up online** using the website or keywords provided. Fill in missing context to make it understandable.
   - Use a simple numbered guide:
     1. Start with this…
     2. Set this up…
     3. Connect it to… etc.
   - Clearly explain any configuration based on features, category, or tags.
2. For any linked tools (via “edges”):
   - Explain how each tool passes data or action to the next one.
   - Use simple metaphors or business-friendly language (e.g., “This tool gathers info, then sends it to the next tool to process it”).
3. Present a clear overall picture:
   - Summarize the goal of the full workflow.
   - Describe how all the tools work together to achieve the user’s objective.
4. Add real-world value:
   - Mention any standout benefits or tips you found during your online research.
   - Recommend one small action the user can do now (like testing one of the tools).
5. End by asking the user:
   - “Would you like to add anything or need something clarified?”
Tone: Be friendly, confident, professional, and helpful. Speak like a smart assistant who’s there to make complex things easy and actionable.
Begin your explanation:
    """

    # Format the JSON object into the prompt
    prompt = prompt_template.format(
        json_object=json.dumps(json_object, indent=2), query=query
    )

    # Initialize OpenAI model
    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4000)

    # Invoke the model with the prompt
    response = await llm.ainvoke(prompt)

    return response.content
