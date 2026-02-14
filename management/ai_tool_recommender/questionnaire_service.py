"""Questionnaire Service for dynamic 3-phase workflow generation through questions."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from asgiref.sync import sync_to_async

from .ai_agents.core.llm import get_shared_llm

logger = logging.getLogger(__name__)


class QuestionnaireService:
    """Service to manage 4-phase questionnaire-based workflow generation.

    Phase 1: Discovery Part 1 - Initial understanding (3 questions)
    Phase 2: Discovery Part 2 - Deeper exploration (4 questions)
    Phase 3: Discovery Part 3 - Final details (3 questions)
    Phase 4: Refinement - Generate refined query and confirm/refine with user (1 question)
    """

    # Phase configuration
    PHASE_CONFIG = {
        1: {"name": "Intent Identification", "num_questions": 3, "order": 1},
        2: {"name": "Gathering Context", "num_questions": 4, "order": 2},
        3: {"name": "Next Steps and Timeline", "num_questions": 3, "order": 3},
        4: {"name": "Refine Query", "num_questions": 1, "order": 4},
    }

    def __init__(self):
        """Initialize the questionnaire service."""
        self.llm = get_shared_llm()

    async def generate_phase_questions(
        self,
        phase: int,
        user_query: str,
        previous_phases_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate questions for a specific phase based on user query and previous phase data.

        Args:
            phase: Phase number (1, 2, 3, or 4)
            user_query: User's initial query
            previous_phases_data: Data from previous phases for context

        Returns:
            List of question dictionaries for this phase
        """
        try:
            phase_config = self.PHASE_CONFIG.get(phase)
            if not phase_config:
                raise ValueError(f"Invalid phase: {phase}")

            num_questions = phase_config["num_questions"]
            phase_name = phase_config["name"]

            logger.info(
                f"📝 Generating {num_questions} questions for Phase {phase}: {phase_name}"
            )

            if phase == 1:
                prompt = self._get_phase_1_prompt(user_query, num_questions)
            elif phase == 2:
                prompt = self._get_phase_2_prompt(
                    user_query, previous_phases_data, num_questions
                )
            elif phase == 3:
                prompt = self._get_phase_3_prompt(
                    user_query, previous_phases_data, num_questions
                )
            elif phase == 4:
                prompt = self._get_phase_4_prompt(
                    user_query, previous_phases_data, num_questions
                )
            else:
                raise ValueError(f"Unknown phase: {phase}")

            response = await self.llm.generate_response(prompt)
            result = await self.llm.parse_json_response(response)

            if not result or "questions" not in result:
                logger.error("❌ LLM didn't return valid questions format")
                raise ValueError("Failed to generate questions - LLM response invalid")

            questions = result["questions"][:num_questions]

            logger.info(
                f"✅ Generated {len(questions)} questions for Phase {phase}: {phase_name}"
            )

            return questions

        except Exception as e:
            logger.error(
                f"❌ Error generating Phase {phase} questions: {e}", exc_info=True
            )
            raise

    def _get_phase_1_prompt(self, user_query: str, num_questions: int) -> str:
        """Generate prompt for Phase 1: Discovery."""
        return f"""
You are an expert business consultant helping someone solve their workflow challenges.

USER'S REQUEST: "{user_query}"

Your task: Generate EXACTLY {num_questions} personalized discovery questions that feel like a natural conversation with a helpful consultant.

ANALYSIS FIRST:
- What domain/industry does this seem to be about? (sales, marketing, HR, operations, etc.)
- What type of problem are they facing? (automation, organization, communication, etc.)
- What's the scope? (personal productivity, team workflow, company-wide system)
- What's their likely experience level? (beginner, intermediate, expert)

QUESTION GENERATION RULES:
1. Each question should be SPECIFIC to their domain and problem
2. Use NATURAL, conversational language - like talking to a friend
3. Reference their actual request in the questions when relevant
4. Ask follow-up style questions that build on their initial request
5. Include WHY you're asking each question (helps build trust)
6. Use industry-appropriate terminology but keep it accessible
7. Make questions feel progressive - each builds on understanding

CRITICAL - DO NOT USE GENERIC TEMPLATES:
❌ BAD: "What specific challenge or pain point are you trying to solve?"
✅ GOOD: "I can see you're dealing with [specific issue from their request]. What's the biggest headache this is causing for your [team/business/workflow] right now?"

❌ BAD: "Who will be using or benefiting from this solution?"
✅ GOOD: "When you get this [specific solution] working, who on your team will be the main people using it day-to-day?"

QUESTION FOCUS AREAS (adapt to their specific context):
- INTENT: What exactly they want to achieve (be specific to their domain)
- PROBLEM: The pain points they're experiencing (use their language/context)
- OUTCOME: What success looks like for them (measurable, specific)
- USERS: Who's involved and affected (roles, team size, stakeholders)
- TIMELINE: When they need this (urgency, constraints, deadlines)

EXAMPLES GENERATION (CRITICAL):
- Create 3 realistic ANSWER EXAMPLES that someone might SAY IN RESPONSE to the question
- These are SUGGESTED ANSWERS, NOT questions - NEVER generate questions (e.g., "What tools do you use?" is WRONG)
- Examples should be DECLARATIVE statements (e.g., "I want to automate payroll processing" is CORRECT)
- Use domain-specific language and scenarios
- Make examples feel authentic, not generic

Return JSON with this structure:
{{
  "questions": [
    {{
      "id": "discovery_intent",
      "question": "Natural, conversational question specific to their request that explores what they want to build/achieve",
      "type": "open_text",
      "required": true,
      "examples": ["Domain-specific example 1", "Domain-specific example 2", "Domain-specific example 3"],
      "tool_examples": [],
      "reason": "Brief explanation of why this helps you assist them better"
    }}
  ]
}}

REMEMBER:
- Reference their actual request: "{user_query}"
- Make it feel like a conversation, not an interrogation
- Use their domain/industry context
- Each question should feel necessary and helpful
- NO generic business-speak or consultant jargon

Generate {num_questions} personalized discovery questions that feel natural and specific to their request.
"""

    def _get_phase_2_prompt(
        self, user_query: str, previous_phases_data: Dict[str, Any], num_questions: int
    ) -> str:
        """Generate prompt for Phase 2: Tool Exploration."""
        phase1_answers = previous_phases_data.get("phase_1", {}).get("answers", {})

        # Extract context from Phase 1
        context = []
        for answer_data in phase1_answers.values():
            context.append(
                f"Q: {answer_data.get('question', '')}\nA: {answer_data.get('answer', '')}"
            )

        context_str = "\n\n".join(context) if context else "No previous context"

        return f"""
You are continuing a conversation as a helpful business consultant. You've already learned about their goals, now you need to understand their current situation.

ORIGINAL REQUEST: "{user_query}"

WHAT WE'VE LEARNED SO FAR:
{context_str}

PHASE 2 OBJECTIVE: Understand their current setup, tools, and processes to identify gaps and opportunities.

ANALYSIS TASK:
Based on their answers above, analyze:
- What domain/industry are they in?
- What type of work/processes are they trying to improve?
- What's their likely current tech stack based on their role/industry?
- What are the typical pain points in this domain?
- What integrations would be common in their situation?

QUESTION GENERATION RULES:
1. Reference their specific answers from Phase 1 in your questions
2. Ask about tools/processes relevant to their domain
3. Use conversational language that builds on what they've told you
4. Make each question feel like a natural follow-up
5. Include specific tool suggestions that make sense for their situation
6. Ask about pain points in the context of what they're trying to achieve

CRITICAL - MAKE IT CONVERSATIONAL:
❌ BAD: "What tools do you currently use?"
✅ GOOD: "You mentioned [reference their goal/problem]. What tools or systems are you using right now to handle [specific process]?"

❌ BAD: "How do you currently do this?"
✅ GOOD: "Walk me through how you typically [specific process they mentioned] today - what's your current workflow?"

TOOL SUGGESTIONS STRATEGY:
- Suggest 4-6 tools that are actually relevant to their industry/domain
- Don't suggest random tools - make them contextually appropriate
- Consider their team size and complexity level from Phase 1

QUESTION FOCUS AREAS (adapt to their context):
- CURRENT TOOLS: What they're using now (be specific to their domain)
- CURRENT PROCESS: How they do things today (reference their specific workflow)
- PAIN POINTS: What's not working (in context of their goals)
- INTEGRATIONS: What needs to connect (based on their current tools/processes)
- CONSTRAINTS: Limitations they face (budget, team, technical skills)

EXAMPLES GENERATION (CRITICAL):
- Create 3 realistic ANSWER EXAMPLES that someone might SAY IN RESPONSE to the question
- These are SUGGESTED ANSWERS, NOT questions - NEVER generate questions (e.g., "What tools do you use?" is WRONG)
- Examples should be DECLARATIVE statements (e.g., "I'm using Slack and Trello for project management" is CORRECT)
- Use domain-specific language and scenarios
- Make examples feel authentic, not generic

Return JSON with this structure:
{{
  "questions": [
    {{
      "id": "tools_current_stack",
      "question": "Conversational question that references their Phase 1 answers and asks about current tools in their specific context",
      "type": "open_text",
      "required": true,
      "examples": ["Realistic example 1 for their domain", "Realistic example 2", "Realistic example 3"],
      "tool_examples": ["Tool1 relevant to their domain", "Tool2", "Tool3", "Tool4"],
      "reason": "Brief explanation of why this helps"
    }}
  ]
}}

REMEMBER:
- Build on their Phase 1 answers - reference them directly
- Make questions feel like natural conversation progression
- Suggest tools that actually make sense for their situation
- Use their language and context from previous answers
- Keep it friendly and non-technical

Generate {num_questions} contextual questions that build naturally on what they've already shared.
"""

    def _get_phase_3_prompt(
        self, user_query: str, previous_phases_data: Dict[str, Any], num_questions: int
    ) -> str:
        """Generate prompt for Phase 3: Final Requirements."""
        phase1_answers = previous_phases_data.get("phase_1", {}).get("answers", {})
        phase2_answers = previous_phases_data.get("phase_2", {}).get("answers", {})

        # Extract context from Phase 1 and Phase 2
        all_context = []

        # Phase 1 context
        for answer_data in phase1_answers.values():
            all_context.append(
                f"Q: {answer_data.get('question', '')}\nA: {answer_data.get('answer', '')}"
            )

        # Phase 2 context
        for answer_data in phase2_answers.values():
            all_context.append(
                f"Q: {answer_data.get('question', '')}\nA: {answer_data.get('answer', '')}"
            )

        context_str = "\n\n".join(all_context) if all_context else "No previous context"

        return f"""
You are wrapping up a comprehensive discovery conversation as a business consultant. You now understand their goals and current situation - time to nail down the final requirements.

ORIGINAL REQUEST: "{user_query}"

COMPLETE CONVERSATION HISTORY:
{context_str}

PHASE 3 OBJECTIVE: Gather final technical and business requirements to ensure the solution fits perfectly.

ANALYSIS TASK:
Based on everything they've shared:
- What integrations would be critical for their workflow?
- What constraints (budget, team, technical) should we be aware of?
- What specific requirements haven't been covered yet?
- What would make or break this solution for them?

QUESTION GENERATION RULES:
1. Reference specific tools/processes they mentioned in previous phases
2. Ask about integrations that would actually matter for their workflow
3. Probe for constraints that could affect tool selection
4. Identify any deal-breakers or must-haves they haven't mentioned
5. Keep questions practical and business-focused
6. Make each question feel essential, not just "nice to know"

CRITICAL - MAKE IT SPECIFIC TO THEIR SITUATION:
❌ BAD: "What other tools or systems does this need to work with?"
✅ GOOD: "You mentioned using [specific tools from Phase 2]. Which of these would need to connect with your new [solution type] to make your workflow seamless?"

❌ BAD: "Are there any constraints we should know about?"
✅ GOOD: "Given that you're a [team size] team working on [their goal], what's realistic in terms of budget and how technical your team can get?"

INTEGRATION FOCUS:
- Reference the specific tools they mentioned using
- Ask about data flow between systems
- Consider their current pain points when asking about connections

CONSTRAINT FOCUS:
- Budget reality check based on their team size/company
- Technical expertise level of their team
- Timeline constraints
- Compliance or security requirements if relevant

EXAMPLES GENERATION (CRITICAL):
- Create 3 realistic ANSWER EXAMPLES that someone might SAY IN RESPONSE to the question
- These are SUGGESTED ANSWERS, NOT questions - NEVER generate questions (e.g., "What's your budget?" is WRONG)
- Examples should be DECLARATIVE statements (e.g., "We need it to integrate with Salesforce and HubSpot" is CORRECT)
- Use domain-specific language and scenarios
- Make examples feel authentic, not generic

Return JSON with this structure:
{{
  "questions": [
    {{
      "id": "tools_integrations",
      "question": "Specific question about integrations that references their current tools and workflow",
      "type": "open_text",
      "required": true,
      "examples": ["Realistic integration example 1", "Example 2", "Example 3"],
      "tool_examples": ["Integration tool 1 relevant to their setup", "Tool 2", "Tool 3", "Tool 4"],
      "reason": "Brief explanation of why this matters for their success"
    }}
  ]
}}

REMEMBER:
- This is the final discovery phase - make every question count
- Reference their specific situation from previous answers
- Focus on practical constraints and requirements
- Help them think through what they might have missed
- Keep it conversational but thorough

Generate {num_questions} targeted final requirement questions based on their complete conversation history.
"""

    def _get_phase_4_prompt(
        self, user_query: str, previous_phases_data: Dict[str, Any], num_questions: int
    ) -> str:
        """Generate prompt for Phase 4: Refinement."""
        # Collect all previous answers
        all_answers = []
        for phase_num in [1, 2, 3]:
            phase_data = previous_phases_data.get(f"phase_{phase_num}", {})
            answers = phase_data.get("answers", {})
            for answer_data in answers.values():
                all_answers.append(
                    f"Q: {answer_data.get('question', '')}\nA: {answer_data.get('answer', '')}"
                )

        conversation_context = (
            "\n\n".join(all_answers) if all_answers else "No previous context"
        )

        return f"""
You are a senior business analyst who has just completed an in-depth discovery session with a client. Now you must create a comprehensive problem and workflow analysis that will guide the selection of perfect tools and creation of an optimal workflow.

ORIGINAL REQUEST: "{user_query}"

COMPLETE DISCOVERY CONVERSATION:
{conversation_context}

PHASE 4 OBJECTIVE: Create a DEEP, COMPREHENSIVE analysis that captures not just what they want, but WHY they want it, HOW they'll use it, and WHAT problems they're solving.

CRITICAL ANALYSIS FRAMEWORK:
You must analyze the conversation from multiple angles:

1. **Problem Deep-Dive**: What is the REAL problem beneath their surface request?
   - Core pain points (be specific about each)
   - Root causes (why do these problems exist?)
   - Impact on their business/work (quantify if possible)
   - Urgency and priority

2. **User Context Analysis**: Who are they and what's their situation?
   - Technical proficiency level (how technical are they?)
   - Team size and structure
   - Industry and domain specifics
   - Current tool stack (what they already use)
   - Budget and resource constraints

3. **Workflow Requirements**: How should the solution work?
   - Current process flow (how they do things now)
   - Desired process flow (how they want to do things)
   - Integration points (what must connect to what)
   - Automation opportunities (what can be automated)
   - Manual steps that must remain

4. **Tool Requirements**: What characteristics must tools have?
   - Must-have features (non-negotiable)
   - Nice-to-have features (would enhance solution)
   - Tool categories needed (CRM, automation, analytics, etc.)
   - Specific tools they mentioned wanting
   - Integration capabilities required

5. **Success Criteria**: How will we know this works?
   - Measurable outcomes they expect
   - Timelines and deadlines
   - Key metrics to track
   - Definition of "done"

REFINED QUERY STRUCTURE (MARKDOWN FORMAT):
Your refined query should be a comprehensive analysis document with the following structure:

## Problem Analysis
[2-3 paragraphs explaining the core problem, its root causes, impact, and why it matters]

## Current Situation
**Tools in Use:** [List actual tools they mentioned]
**Current Process:** [Detailed description of how they do things now]
**Pain Points:** [Specific, detailed pain points]
**Constraints:** [Budget, technical, timeline, team constraints]

## Desired Workflow Solution
[3-4 paragraphs describing the ideal workflow that will solve their problem]
- Describe the flow from start to finish
- Explain how it addresses each pain point
- Include integration points
- Mention automation opportunities

## Tool Requirements & Suggestions
**Essential Tool Categories:**
- [Category 1]: [Why needed and what it should do]
- [Category 2]: [Why needed and what it should do]
- [Category 3+]: [Continue for all categories]

**Specific Tools Mentioned:**
- [If they mentioned specific tools, list them and why they want them]

**Integration Requirements:**
- [What needs to connect with what]

## Implementation Considerations
**Users:** [Who will use this]
**Timeline:** [When they need it]
**Technical Level:** [Their technical proficiency]
**Success Metrics:** [How to measure success]

## Workflow Design Guidance
[2-3 paragraphs providing guidance on how the workflow should be structured]
- Should it be linear or parallel?
- Where are the decision points?
- What are the critical paths?
- How should data flow between tools?

WRITING GUIDELINES:
1. Write as if you're a non-technical user explaining to another human - avoid jargon
2. Use THEIR words and phrases from the conversation
3. Be EXTREMELY specific - names, numbers, tools, processes
4. Explain the "why" behind every requirement
5. Include context from their industry/domain
6. Think like a problem solver, not just a requirements gatherer
7. Write in markdown format but keep it readable and flowing
8. Aim for 500-800 words total - be comprehensive but concise

CONFIRMATION QUESTION:
Create a natural, conversational question that:
- Summarizes that you've created a comprehensive analysis
- Invites them to review it
- Makes it easy to request changes
- Feels like a colleague checking in

EXAMPLES GENERATION (CRITICAL):
- Create 3 realistic ANSWER EXAMPLES that someone might SAY IN RESPONSE
- These are SUGGESTED ANSWERS, NOT questions
- Examples: "Yes, that's perfect", "Can you add [specific thing]", "Actually, change [specific detail]"
- Make them feel authentic to their situation

Return JSON with this structure:
{{
  "refined_query": "COMPREHENSIVE MARKDOWN ANALYSIS (500-800 words) using the structure above",
  "questions": [
    {{
      "id": "refinement_confirmation",
      "question": "Natural question presenting the analysis and asking for feedback",
      "type": "open_text",
      "required": true,
      "examples": ["Yes, that captures everything perfectly", "Can you add [realistic specific]", "Change [realistic specific detail]"],
      "tool_examples": [],
      "reason": "Ensure the analysis is complete and accurate before tool selection",
      "is_refined": true,
      "refined_query": "SAME MARKDOWN CONTENT as above"
    }}
  ]
}}

CRITICAL SUCCESS FACTORS:
- Make this feel like a consultant's analysis, not a form response
- Use their exact words and specific details from the conversation
- Explain the problem from multiple angles
- Provide clear guidance for tool selection and workflow design
- Be comprehensive enough that someone could build a solution from this alone
- Write in a flowing, narrative style that's easy to read

Create an analysis that shows deep understanding of their problem and provides clear direction for solving it.
"""

    async def _generate_phase_4_with_refined_query(
        self, user_query: str, previous_phases_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate Phase 4 questions and refined query.

        Args:
            user_query: Original user query
            previous_phases_data: Data from previous phases

        Returns:
            Dict with 'refined_query' and 'questions'
        """
        try:
            prompt = self._get_phase_4_prompt(user_query, previous_phases_data, 1)
            response = await self.llm.generate_response(prompt)
            result = await self.llm.parse_json_response(response)

            if not result:
                raise ValueError("Failed to parse Phase 4 response")

            return {
                "refined_query": result.get("refined_query", ""),
                "questions": result.get("questions", []),
            }

        except Exception as e:
            logger.error(
                f"❌ Error generating Phase 4 with refined query: {e}", exc_info=True
            )
            raise

    async def initialize_questionnaire(self, user_query: str) -> Dict[str, Any]:
        """
        Initialize a new 4-phase questionnaire for a conversation session.

        Args:
            user_query: User's initial query

        Returns:
            Initialized questionnaire dictionary with Phase 1 questions
        """
        try:
            # Generate Phase 1 questions
            phase_1_questions = await self.generate_phase_questions(
                phase=1, user_query=user_query, previous_phases_data=None
            )

            # Initialize the complete phase structure
            questionnaire = {
                "current_phase": 1,
                "original_query": user_query,
                "started_at": datetime.now().isoformat(),
                "completed_at": None,
                "phase_1": {
                    "name": "Intent Identification",
                    "status": "in_progress",
                    "questions": phase_1_questions,
                    "answers": {},
                    "current_question_index": 0,
                    "started_at": datetime.now().isoformat(),
                    "completed_at": None,
                },
                "phase_2": {
                    "name": "Gathering Context",
                    "status": "pending",
                    "questions": [],
                    "answers": {},
                    "current_question_index": 0,
                    "started_at": None,
                    "completed_at": None,
                },
                "phase_3": {
                    "name": "Next Steps and Timeline",
                    "status": "pending",
                    "questions": [],
                    "answers": {},
                    "current_question_index": 0,
                    "started_at": None,
                    "completed_at": None,
                },
                "phase_4": {
                    "name": "Refine Query",
                    "status": "pending",
                    "questions": [],
                    "answers": {},
                    "refined_query": None,
                    "current_question_index": 0,
                    "started_at": None,
                    "completed_at": None,
                },
                "workflow_info": {
                    "intent": None,  # "tool" or "workflow" - 10%
                    "outcome": None,  # 10%
                    "problem": None,  # 10%
                    "users": None,  # 10%
                    "timeline": None,  # 10%
                    "current_tools": [],  # 10%
                    "current_process": None,  # 10%
                    "pain_points": [],  # 10%
                    "integrations": [],  # 10%
                    "constraints": [],  # 10%
                    "refined_query": None,  # Not counted in progress (added at Phase 4)
                },
                "progress_percentage": 0,  # 0-100% based on workflow_info fields filled
            }

            logger.info(
                f"✅ Initialized 4-phase questionnaire - Starting Phase 1: Intent Identification"
            )
            return questionnaire

        except Exception as e:
            logger.error(f"❌ Error initializing questionnaire: {e}")
            raise

    async def process_answer(
        self, questionnaire_json: Dict[str, Any], user_message: str
    ) -> Tuple[Dict[str, Any], bool, bool]:
        """
        Process user's answer to the current question and handle phase transitions.

        Args:
            questionnaire_json: Current questionnaire state
            user_message: User's answer message

        Returns:
            Tuple of (updated_questionnaire, phase_complete, all_complete)
        """
        try:
            current_phase = questionnaire_json.get("current_phase", 1)
            phase_key = f"phase_{current_phase}"
            phase_data = questionnaire_json[phase_key]

            current_index = phase_data.get("current_question_index", 0)
            questions = phase_data.get("questions", [])

            if current_index >= len(questions):
                logger.warning(f"⚠️ No more questions in Phase {current_phase}")
                return questionnaire_json, True, False

            current_question = questions[current_index]

            # Special handling for Phase 4 refinement
            if (
                current_phase == 4
                and current_question.get("id") == "refinement_confirmation"
            ):
                return await self._process_refinement_answer(
                    questionnaire_json, user_message, phase_data, current_question
                )

            # Extract and validate answer
            answer = await self._extract_answer_from_message(
                user_message, current_question
            )

            # Save answer in current phase
            phase_data["answers"][current_question["id"]] = {
                "answer": answer,
                "answered_at": datetime.now().isoformat(),
                "question": current_question["question"],
            }

            # Move to next question in this phase
            phase_data["current_question_index"] = current_index + 1

            # Update progress percentage after each answer
            questionnaire_json[
                "progress_percentage"
            ] = self.calculate_progress_percentage(questionnaire_json)

            logger.info(
                f"✅ Phase {current_phase} - Answer saved for question {current_index + 1}/{len(questions)}"
            )

            # Check if current phase is complete
            phase_complete = current_index + 1 >= len(questions)

            if phase_complete:
                phase_data["status"] = "complete"
                phase_data["completed_at"] = datetime.now().isoformat()
                logger.info(f"🎉 Phase {current_phase}: {phase_data['name']} COMPLETE!")

                # Extract information from completed phase
                await self._extract_phase_information(questionnaire_json, current_phase)

                # Check if all phases are complete
                if current_phase == 4:
                    questionnaire_json["completed_at"] = datetime.now().isoformat()
                    logger.info("🎉🎉🎉 ALL PHASES COMPLETE! Ready to generate workflow!")
                    return questionnaire_json, True, True

            return questionnaire_json, phase_complete, False

        except Exception as e:
            logger.error(f"❌ Error processing answer: {e}", exc_info=True)
            raise

    async def _process_refinement_answer(
        self,
        questionnaire_json: Dict[str, Any],
        user_message: str,
        phase_data: Dict[str, Any],
        current_question: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool, bool]:
        """
        Process Phase 3 refinement answer with change detection and iteration.

        Args:
            questionnaire_json: Current questionnaire state
            user_message: User's response
            phase_data: Phase 3 data
            current_question: Current refinement question

        Returns:
            Tuple of (updated_questionnaire, phase_complete, all_complete)
        """
        try:
            # Analyze if user is confirming or requesting changes
            current_refined_query = phase_data.get("refined_query", "")

            prompt = f"""
            Current refined query: "{current_refined_query}"

            User's response: "{user_message}"

            Analyze if the user is:
            A) CONFIRMING the refined query (e.g., "yes", "perfect", "looks good", "correct")
            B) REQUESTING CHANGES (e.g., "change X to Y", "add Z", "no, I want...")

            Return JSON:
            {{
              "action": "confirm" or "change",
              "changes_requested": "Description of changes if action is 'change', otherwise null"
            }}

            Return ONLY valid JSON.
            """

            response = await self.llm.generate_response(prompt)
            analysis = await self.llm.parse_json_response(response)

            action = analysis.get("action", "confirm")
            changes_requested = analysis.get("changes_requested")

            logger.info(
                f"📊 Refinement analysis: action={action}, changes={changes_requested}"
            )

            if action == "confirm":
                # User confirmed - complete Phase 3
                phase_data["answers"][current_question["id"]] = {
                    "answer": user_message,
                    "answered_at": datetime.now().isoformat(),
                    "question": current_question["question"],
                    "confirmed": True,
                }

                phase_data["current_question_index"] = 1  # Move past the question
                phase_data["status"] = "complete"
                phase_data["completed_at"] = datetime.now().isoformat()
                questionnaire_json["completed_at"] = datetime.now().isoformat()

                logger.info("✅ User confirmed refined query - Phase 3 COMPLETE!")
                return questionnaire_json, True, True

            else:
                # User wants changes - regenerate refined query
                logger.info(f"🔄 User requested changes: {changes_requested}")

                # Save the change request
                if "refinement_iterations" not in phase_data:
                    phase_data["refinement_iterations"] = []

                phase_data["refinement_iterations"].append(
                    {
                        "iteration": len(phase_data["refinement_iterations"]) + 1,
                        "user_feedback": user_message,
                        "changes_requested": changes_requested,
                        "previous_query": current_refined_query,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

                # Generate new refined query incorporating changes
                new_refined_query = await self._regenerate_refined_query(
                    questionnaire_json, current_refined_query, changes_requested
                )

                # Generate new dynamic examples based on the updated refined query
                # Example 1 stays "Yes, that's perfect", examples 2 & 3 are dynamic
                new_examples = await self._generate_dynamic_examples(new_refined_query)

                # Update the refined query
                phase_data["refined_query"] = new_refined_query
                questionnaire_json["workflow_info"]["refined_query"] = new_refined_query

                # Update the question to ask for confirmation again
                phase_data["questions"][0]["question"] = (
                    f"I've updated it based on your feedback. Here's the new version:\n\n"
                    f"**{new_refined_query}**\n\n"
                    f"Does this better capture what you're looking for?"
                )

                # Update examples - example 1 stays same, examples 2 & 3 are contextual
                phase_data["questions"][0]["examples"] = new_examples

                # DON'T move to next question - stay on the same question for re-confirmation
                # phase_data["current_question_index"] stays at 0

                logger.info(
                    f"✅ Regenerated refined query: {new_refined_query[:100]}..."
                )
                logger.info(f"✅ Regenerated examples: {new_examples}")
                return (
                    questionnaire_json,
                    False,
                    False,
                )  # Not complete yet, need confirmation

        except Exception as e:
            logger.error(f"❌ Error processing refinement answer: {e}", exc_info=True)
            # Fallback: treat as confirmation
            phase_data["answers"][current_question["id"]] = {
                "answer": user_message,
                "answered_at": datetime.now().isoformat(),
                "question": current_question["question"],
            }
            phase_data["current_question_index"] = 1
            phase_data["status"] = "complete"
            phase_data["completed_at"] = datetime.now().isoformat()
            questionnaire_json["completed_at"] = datetime.now().isoformat()
            return questionnaire_json, True, True

    async def _generate_dynamic_examples(self, refined_query: str) -> list:
        """
        Generate dynamic, contextual examples based on the refined query content.
        All examples should feel natural and specific to their project.

        Args:
            refined_query: The refined query to generate examples for

        Returns:
            List of 3 example strings [confirmation, specific_change, specific_addition]
        """
        try:
            prompt = f"""
You are helping generate realistic response examples for someone reviewing their project brief.

PROJECT BRIEF:
"{refined_query}"

Generate 3 realistic examples of how someone might respond to this project brief:

1. CONFIRMATION: A natural way to confirm this is accurate (not just "Yes, that's perfect")
2. SPECIFIC CHANGE: A realistic change they might want based on the actual content
3. SPECIFIC ADDITION: A realistic addition they might want based on their situation

REQUIREMENTS:
- Use natural, conversational language
- Reference actual details from their project brief
- Make examples feel authentic - like real responses
- Be specific to their industry/situation
- Avoid generic business language

ANALYSIS FIRST:
- What industry/domain is this?
- What specific tools/processes are mentioned?
- What timeline/constraints are mentioned?
- What might they realistically want to adjust?

EXAMPLE QUALITY:
❌ BAD: "Change X to Y" or "Add Z requirement"
✅ GOOD: "Actually, make the timeline 6 months instead of 3 - we want to do this right"
✅ GOOD: "This looks great! I think we should also include mobile access since our team travels a lot"

Return JSON:
{{
  "confirmation_example": "Natural confirmation that sounds like a real person",
  "change_example": "Specific, realistic change based on their actual project details",
  "addition_example": "Specific, realistic addition that makes sense for their situation"
}}

Make each example sound like something a real person in their situation would actually say.
"""

            response = await self.llm.generate_response(prompt)
            result = await self.llm.parse_json_response(response)

            confirmation_example = result.get(
                "confirmation_example", "Yes, that captures exactly what we need"
            )
            change_example = result.get(
                "change_example",
                "Actually, let's extend the timeline to be more realistic",
            )
            addition_example = result.get(
                "addition_example", "We should also consider mobile access for our team"
            )

            dynamic_examples = [confirmation_example, change_example, addition_example]

            logger.info(f"✅ Generated contextual examples: {dynamic_examples}")
            return dynamic_examples

        except Exception as e:
            logger.error(f"❌ Error generating dynamic examples: {e}", exc_info=True)
            # Fallback to contextual examples
            return [
                "Yes, that captures exactly what we need",
                "Actually, let's extend the timeline to be more realistic",
                "We should also consider mobile access for our team",
            ]

    async def _regenerate_refined_query(
        self,
        questionnaire_json: Dict[str, Any],
        current_refined_query: str,
        changes_requested: str,
    ) -> str:
        """
        Regenerate refined query incorporating user's requested changes.

        Args:
            questionnaire_json: Full questionnaire state
            current_refined_query: Current refined query
            changes_requested: What the user wants to change

        Returns:
            New refined query string
        """
        try:
            # Get all phase data for context
            phase_1_answers = questionnaire_json.get("phase_1", {}).get("answers", {})
            phase_2_answers = questionnaire_json.get("phase_2", {}).get("answers", {})

            all_context = []
            for phase_num, answers in [(1, phase_1_answers), (2, phase_2_answers)]:
                for answer_data in answers.values():
                    all_context.append(
                        f"Q: {answer_data.get('question', '')}\nA: {answer_data.get('answer', '')}"
                    )

            context_str = "\n\n".join(all_context)

            prompt = f"""
You are a business consultant updating a project brief based on client feedback.

ORIGINAL REQUEST: "{questionnaire_json.get('original_query', '')}"

DISCOVERY CONVERSATION:
{context_str}

CURRENT PROJECT BRIEF:
"{current_refined_query}"

CLIENT FEEDBACK:
{changes_requested}

TASK: Update the project brief to incorporate their feedback while maintaining all the valuable context from our discovery conversation.

UPDATE RULES:
1. Make the requested changes thoughtfully
2. Keep all relevant details from the original brief
3. Maintain their specific language and context
4. Ensure the updated brief is still comprehensive and actionable
5. Use the same professional but accessible tone
6. Keep the markdown structure clean and organized

CRITICAL SUCCESS FACTORS:
- Address their specific feedback directly
- Don't lose important details from the original brief
- Make sure the updated version still captures their complete needs
- Use their industry context and terminology
- Keep it specific to their situation (team size, tools, constraints, etc.)

Return JSON:
{{
  "refined_query": "UPDATED MARKDOWN PROJECT BRIEF that incorporates their feedback while maintaining all valuable context"
}}

Create an updated project brief that addresses their feedback while keeping everything else that was accurate.
"""

            response = await self.llm.generate_response(prompt)
            result = await self.llm.parse_json_response(response)

            new_query = result.get("refined_query", current_refined_query)

            logger.info(f"✅ Generated new refined query: {new_query[:100]}...")
            return new_query

        except Exception as e:
            logger.error(f"❌ Error regenerating refined query: {e}", exc_info=True)
            return current_refined_query  # Fallback to current query

    async def transition_to_next_phase(
        self, questionnaire_json: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Transition to the next phase and generate questions.

        Args:
            questionnaire_json: Current questionnaire state

        Returns:
            Updated questionnaire with next phase questions
        """
        try:
            current_phase = questionnaire_json.get("current_phase", 1)
            next_phase = current_phase + 1

            if next_phase > 4:
                logger.warning("⚠️ No more phases after Phase 4")
                return questionnaire_json

            logger.info(
                f"🔄 Transitioning from Phase {current_phase} to Phase {next_phase}"
            )

            # Generate questions for next phase
            phase_key = f"phase_{next_phase}"

            if next_phase == 4:
                # For Phase 4, the LLM returns both refined_query and questions
                logger.info("🔄 Generating Phase 4 questions and refined query...")
                response = await self._generate_phase_4_with_refined_query(
                    questionnaire_json["original_query"], questionnaire_json
                )
                next_phase_questions = response.get("questions", [])
                refined_query = response.get("refined_query", "")

                logger.info(
                    f"📝 Generated {len(next_phase_questions)} Phase 4 questions"
                )
                logger.info(f"📝 Refined query length: {len(refined_query)} chars")

                if not next_phase_questions:
                    raise ValueError("Phase 4 generation returned no questions")

                # Generate dynamic examples based on the refined query
                # Example 1 is always "Yes, that's perfect", examples 2 & 3 are dynamic
                dynamic_examples = await self._generate_dynamic_examples(refined_query)

                # Update the question with dynamic examples
                if next_phase_questions and len(next_phase_questions) > 0:
                    next_phase_questions[0]["examples"] = dynamic_examples
                    logger.info(
                        f"✅ Updated Phase 4 question with dynamic examples: {dynamic_examples}"
                    )

                # Update Phase 4 data
                questionnaire_json[phase_key]["questions"] = next_phase_questions
                questionnaire_json[phase_key]["refined_query"] = refined_query
                questionnaire_json["workflow_info"]["refined_query"] = refined_query
            else:
                next_phase_questions = await self.generate_phase_questions(
                    phase=next_phase,
                    user_query=questionnaire_json["original_query"],
                    previous_phases_data=questionnaire_json,
                )

                if not next_phase_questions:
                    raise ValueError(
                        f"Phase {next_phase} generation returned no questions"
                    )

                questionnaire_json[phase_key]["questions"] = next_phase_questions

            # Update phase status
            questionnaire_json[phase_key]["status"] = "in_progress"
            questionnaire_json[phase_key]["started_at"] = datetime.now().isoformat()

            # Update current phase
            questionnaire_json["current_phase"] = next_phase

            phase_name = self.PHASE_CONFIG[next_phase]["name"]
            logger.info(f"✅ Transitioned to Phase {next_phase}: {phase_name}")

            return questionnaire_json

        except Exception as e:
            logger.error(f"❌ Error transitioning to next phase: {e}", exc_info=True)
            logger.error(
                f"❌ Questionnaire state: phase={questionnaire_json.get('current_phase')}"
            )
            raise

    async def _extract_phase_information(
        self, questionnaire_json: Dict[str, Any], completed_phase: int
    ) -> None:
        """
        Extract structured information from a completed phase using LLM.

        Args:
            questionnaire_json: Current questionnaire state
            completed_phase: The phase that was just completed
        """
        try:
            phase_key = f"phase_{completed_phase}"
            phase_data = questionnaire_json[phase_key]
            answers = phase_data.get("answers", {})

            # Build context from answers
            answer_context = []
            for answer_data in answers.values():
                answer_context.append(
                    f"Q: {answer_data['question']}\nA: {answer_data['answer']}"
                )
            context_str = "\n\n".join(answer_context)

            if completed_phase == 1:
                # Extract intent, outcome, problem, users, timeline from Phase 1
                prompt = f"""
                PHASE 1 ANSWERS:
                {context_str}

                Extract the following information:
                1. intent: Is the user looking for a "tool" or a "workflow"?
                2. outcome: What is their desired outcome/goal?
                3. problem: What problem/pain point are they trying to solve?
                4. users: Who will be using/benefiting from this?
                5. timeline: When do they need this?

                Return JSON:
                {{
                  "intent": "tool" or "workflow",
                  "outcome": "Brief description of desired outcome",
                  "problem": "Core problem they're solving",
                  "users": "Who will use this",
                  "timeline": "When they need it"
                }}

                Return ONLY valid JSON.
                """

                response = await self.llm.generate_response(prompt)
                extracted = await self.llm.parse_json_response(response)

                questionnaire_json["workflow_info"]["intent"] = extracted.get("intent")
                questionnaire_json["workflow_info"]["outcome"] = extracted.get(
                    "outcome"
                )
                questionnaire_json["workflow_info"]["problem"] = extracted.get(
                    "problem"
                )
                questionnaire_json["workflow_info"]["users"] = extracted.get("users")
                questionnaire_json["workflow_info"]["timeline"] = extracted.get(
                    "timeline"
                )

                logger.info(
                    f"✅ Extracted from Phase 1: Intent={extracted.get('intent')}, Outcome={extracted.get('outcome')}, Problem={extracted.get('problem')}"
                )

            elif completed_phase == 2:
                # Extract tools and process from Phase 2
                prompt = f"""
                PHASE 2 ANSWERS:
                {context_str}

                Extract the following information:
                1. current_tools: List of tools/platforms they currently use
                2. current_process: How they currently do this
                3. pain_points: What frustrates them about current setup

                Return JSON:
                {{
                  "current_tools": ["tool1", "tool2"],
                  "current_process": "Brief description",
                  "pain_points": ["pain1", "pain2"]
                }}

                Return ONLY valid JSON.
                """

                response = await self.llm.generate_response(prompt)
                extracted = await self.llm.parse_json_response(response)

                questionnaire_json["workflow_info"]["current_tools"] = extracted.get(
                    "current_tools", []
                )
                questionnaire_json["workflow_info"]["current_process"] = extracted.get(
                    "current_process"
                )
                questionnaire_json["workflow_info"]["pain_points"] = extracted.get(
                    "pain_points", []
                )

                logger.info(
                    f"✅ Extracted from Phase 2: {len(extracted.get('current_tools', []))} tools, {len(extracted.get('pain_points', []))} pain points"
                )

            elif completed_phase == 3:
                # Extract integrations and constraints from Phase 3
                prompt = f"""
                PHASE 3 ANSWERS:
                {context_str}

                Extract the following information:
                1. integrations: What systems need to work together
                2. constraints: Budget, team, technical limitations

                Return JSON:
                {{
                  "integrations": ["system1", "system2"],
                  "constraints": ["constraint1", "constraint2"]
                }}

                Return ONLY valid JSON.
                """

                response = await self.llm.generate_response(prompt)
                extracted = await self.llm.parse_json_response(response)

                questionnaire_json["workflow_info"]["integrations"] = extracted.get(
                    "integrations", []
                )
                questionnaire_json["workflow_info"]["constraints"] = extracted.get(
                    "constraints", []
                )

                logger.info(
                    f"✅ Extracted from Phase 3: {len(extracted.get('integrations', []))} integrations, {len(extracted.get('constraints', []))} constraints"
                )

            elif completed_phase == 4:
                # Extract final refined query from Phase 4
                refined_query = phase_data.get("refined_query")
                if refined_query:
                    questionnaire_json["workflow_info"]["refined_query"] = refined_query
                    logger.info(
                        f"✅ Extracted from Phase 4: Refined query = {refined_query}"
                    )

        except Exception as e:
            logger.error(f"❌ Error extracting phase information: {e}", exc_info=True)
            # Don't raise - extraction is optional

    async def _extract_answer_from_message(
        self, user_message: str, question: Dict[str, Any]
    ) -> Any:
        """
        Extract structured answer from user's message.

        Args:
            user_message: User's response
            question: Question being answered

        Returns:
            Extracted answer (string, list, or dict depending on question type)
        """
        try:
            question_type = question.get("type", "open_text")

            # For open_text, use the message as-is (cleaned up)
            if question_type == "open_text":
                return user_message.strip()

            # For choice questions, use LLM to extract the choice
            elif question_type in ["single_choice", "multiple_choice"]:
                prompt = f"""
                Question: {question["question"]}
                Question Type: {question_type}
                Available Options: {question.get("options", [])}
                User Response: "{user_message}"

                Extract the user's choice from their response.

                If question type is "multiple_choice", return a JSON array of selected options.
                If "single_choice", return a single string with the selected option.

                Return ONLY the extracted answer, no explanation.
                """

                response = await self.llm.generate_response(prompt)
                return response.strip()

            else:
                # Default: return as text
                return user_message.strip()

        except Exception as e:
            logger.error(f"Error extracting answer: {e}")
            return user_message.strip()  # Fallback to raw message

    def is_phase_complete(self, questionnaire_json: Dict[str, Any], phase: int) -> bool:
        """
        Check if a specific phase is complete.

        Args:
            questionnaire_json: Questionnaire state
            phase: Phase number to check

        Returns:
            True if phase is complete
        """
        try:
            phase_key = f"phase_{phase}"
            phase_data = questionnaire_json.get(phase_key, {})
            return phase_data.get("status") == "complete"

        except Exception as e:
            logger.error(f"Error checking phase completion: {e}")
            return False

    def is_questionnaire_complete(self, questionnaire_json: Dict[str, Any]) -> bool:
        """
        Check if all phases are complete.

        Args:
            questionnaire_json: Questionnaire state

        Returns:
            True if all 4 phases are answered
        """
        try:
            return (
                self.is_phase_complete(questionnaire_json, 1)
                and self.is_phase_complete(questionnaire_json, 2)
                and self.is_phase_complete(questionnaire_json, 3)
                and self.is_phase_complete(questionnaire_json, 4)
            )

        except Exception as e:
            logger.error(f"Error checking questionnaire completion: {e}")
            return False

    def get_next_question(
        self, questionnaire_json: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get the next unanswered question in the current phase.

        Args:
            questionnaire_json: Questionnaire state

        Returns:
            Next question dict or None if no more questions in current phase
        """
        try:
            current_phase = questionnaire_json.get("current_phase", 1)
            phase_key = f"phase_{current_phase}"
            phase_data = questionnaire_json.get(phase_key, {})

            current_index = phase_data.get("current_question_index", 0)
            questions = phase_data.get("questions", [])

            if current_index < len(questions):
                return questions[current_index]

            return None

        except Exception as e:
            logger.error(f"Error getting next question: {e}")
            return None

    def build_enriched_search_query(self, questionnaire_json: Dict[str, Any]) -> str:
        """
        Build comprehensive search query from refined query or all phase answers.

        Args:
            questionnaire_json: Questionnaire with answers

        Returns:
            Enriched search query string
        """
        try:
            # First check if we have a refined query from Phase 3
            refined_query = questionnaire_json.get("workflow_info", {}).get(
                "refined_query"
            )
            if refined_query:
                logger.info("📝 Using refined query from Phase 3")
                return refined_query

            # Otherwise build from workflow_info
            workflow_info = questionnaire_json.get("workflow_info", {})
            original_query = questionnaire_json.get("original_query", "")

            query_parts = [f"Original Goal: {original_query}"]

            if workflow_info.get("intent"):
                query_parts.append(f"Intent: {workflow_info['intent']}")

            if workflow_info.get("outcome"):
                query_parts.append(f"Desired Outcome: {workflow_info['outcome']}")

            if workflow_info.get("current_tools"):
                tools_str = ", ".join(workflow_info["current_tools"])
                query_parts.append(f"Current Tools: {tools_str}")

            if workflow_info.get("current_process"):
                query_parts.append(
                    f"Current Process: {workflow_info['current_process']}"
                )

            if workflow_info.get("requirements"):
                req_str = ", ".join(workflow_info["requirements"])
                query_parts.append(f"Requirements: {req_str}")

            enriched_query = " | ".join(query_parts)

            logger.info(f"📝 Built enriched query from workflow_info")
            logger.debug(f"Enriched query: {enriched_query}")

            return enriched_query

        except Exception as e:
            logger.error(f"Error building enriched query: {e}")
            return questionnaire_json.get("original_query", "")  # Fallback

    def get_progress_message(
        self, questionnaire_json: Dict[str, Any]
    ) -> Tuple[str, int]:
        """
        Get a progress message for the user showing phase and question progress.

        Args:
            questionnaire_json: Questionnaire state

        Returns:
            Tuple of (progress_message, total_questions_answered)
        """
        try:
            current_phase = questionnaire_json.get("current_phase", 1)
            phase_key = f"phase_{current_phase}"
            phase_data = questionnaire_json.get(phase_key, {})

            current_index = phase_data.get("current_question_index", 0)
            phase_questions = phase_data.get("questions", [])
            phase_name = phase_data.get("name", f"Phase {current_phase}")

            # Count total questions answered across all phases
            total_answered = 0
            for p in [1, 2, 3, 4]:
                p_key = f"phase_{p}"
                p_data = questionnaire_json.get(p_key, {})
                total_answered += len(p_data.get("answers", {}))

            # Total questions across all phases
            total_questions = 0
            for p in [1, 2, 3, 4]:
                p_key = f"phase_{p}"
                p_data = questionnaire_json.get(p_key, {})
                total_questions += len(p_data.get("questions", []))

            message = f"Phase {current_phase}: {phase_name} - Question {current_index + 1} of {len(phase_questions)}"

            return message, total_answered

        except Exception as e:
            logger.error(f"Error generating progress message: {e}")
            return "Phase 1: Discovery", 0

    def calculate_progress_percentage(self, questionnaire_json: Dict[str, Any]) -> int:
        """
        Calculate progress percentage based on questions answered.

        Phase 1: 3 questions (0-30%)
        Phase 2: 4 questions (30-70%)
        Phase 3: 3 questions (70-100%)
        Phase 4: 1 question (stays at 100%)

        Args:
            questionnaire_json: Current questionnaire state

        Returns:
            Progress percentage (0-100)
        """
        # Count total questions answered across all phases
        total_answered = 0

        # Phase 1: 3 questions (each = 10%)
        phase_1 = questionnaire_json.get("phase_1", {})
        phase_1_answered = len(phase_1.get("answers", {}))
        total_answered += phase_1_answered

        # Phase 2: 4 questions (each = 10%)
        phase_2 = questionnaire_json.get("phase_2", {})
        phase_2_answered = len(phase_2.get("answers", {}))
        total_answered += phase_2_answered

        # Phase 3: 3 questions (each = 10%)
        phase_3 = questionnaire_json.get("phase_3", {})
        phase_3_answered = len(phase_3.get("answers", {}))
        total_answered += phase_3_answered

        # Phase 4: 1 question (doesn't add to percentage, stays at 100%)
        # We don't count Phase 4 as it's just confirmation

        # Each question = 10% (10 questions total in Phase 1 + Phase 2 + Phase 3)
        percentage = min(total_answered * 10, 100)

        logger.info(
            f"📊 Progress: {total_answered}/10 questions answered = {percentage}% "
            f"(Phase 1: {phase_1_answered}/3, Phase 2: {phase_2_answered}/4, Phase 3: {phase_3_answered}/3)"
        )

        return percentage

    async def _analyze_user_response(
        self, user_message: str, current_question: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze if user needs clarification or is providing an answer."""
        try:
            prompt = f"""
            You are analyzing a user's response to a questionnaire question.

            Current Question: "{current_question.get('question', '')}"
            User's Response: "{user_message}"

            Determine if the user is:
            A) PROVIDING AN ANSWER to the question
            B) ASKING FOR CLARIFICATION or more information about the question

            Return JSON with this structure:
            {{
                "needs_clarification": true/false,
                "explanation": "A clear, direct explanation for the USER about what the question means and why it's being asked (if clarification needed). Address the user directly in second person ('you'), not third person.",
                "confidence": 0.0-1.0
            }}

            CRITICAL: If needs_clarification is true, the explanation must:
            - Be written directly TO the user (use "you", "your", not "the user")
            - Be helpful and conversational
            - Explain what information is being asked for
            - Give examples if helpful
            - Be 1-2 sentences maximum
            - DO NOT start with "I'm asking" or "Let me explain" - just explain directly

            Example good explanation: "This question is about the specific challenges or pain points your HR department is facing, like high turnover, slow hiring, or difficulty tracking candidates. This helps me recommend the right tools for your needs."

            Example bad explanation: "The user's response does not specify the challenges..." (NO third person!)
            Example bad explanation: "I'm asking about..." (NO "I'm asking" prefix!)
            """
            response = await self.llm.generate_response(prompt)
            analysis = await self.llm.parse_json_response(response)
            return {
                "needs_clarification": analysis.get("needs_clarification", False),
                "explanation": analysis.get("explanation", ""),
                "confidence": analysis.get("confidence", 0.5),
            }
        except Exception as e:
            logger.error(f"Error analyzing user response: {e}")
            return {"needs_clarification": False, "explanation": "", "confidence": 0.0}

    async def save_refined_query_to_db(
        self,
        refined_query: str,
        workflow_id: str,
        conversation_session,
        request_user,
        questionnaire_json: Dict[str, Any],
    ):
        """
        Save the refined query to the database immediately after generation.

        Args:
            refined_query: The refined query text
            workflow_id: Workflow UUID
            conversation_session: ConversationSession object
            request_user: User object
            questionnaire_json: Complete questionnaire data
        """
        try:
            from .models import RefinedQuery

            workflow_info = questionnaire_json.get("workflow_info", {})

            # Use update_or_create to handle both new and existing workflows
            refined_query_obj, created = await sync_to_async(
                RefinedQuery.objects.update_or_create
            )(
                workflow_id=workflow_id,
                defaults={
                    "user": request_user,
                    "session": conversation_session,
                    "original_query": conversation_session.original_query,
                    "refined_query": refined_query,
                    "workflow_info": workflow_info,
                },
            )

            action = "created" if created else "updated"
            logger.info(
                f"✅ Refined query {action} in database for workflow {workflow_id}"
            )
            logger.info(f"📝 Refined query: {refined_query[:100]}...")

            return refined_query_obj

        except Exception as e:
            logger.error(
                f"❌ Error saving refined query to database: {e}", exc_info=True
            )
            return None
