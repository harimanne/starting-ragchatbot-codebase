import json
import anthropic
from openai import OpenAI
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Claude (Anthropic) or a local Ollama LLM"""

    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- Use `list_courses` when the user asks what courses are available, what topics are covered, or wants to browse the catalog
- Use `search_course_content` for questions about specific course content or detailed educational materials
- Use `get_course_outline` for any question asking for a course outline, structure, or lesson list
- **Up to 2 sequential tool calls per query**: You may call a tool, observe its result, then call a second tool if needed before answering
- After receiving tool results, synthesize all gathered information into a single, comprehensive response
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Use the appropriate tool first, then answer
- **Outline queries**: Call `get_course_outline` and return the course title, course link, and a numbered list of every lesson with its title
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(
        self,
        api_key: str,
        model: str,
        backend: str = "anthropic",
        ollama_url: str = "http://localhost:11434/v1",
        ollama_model: str = "llama3.1",
    ):
        self.backend = backend
        self.model = model if backend == "anthropic" else ollama_model

        if backend == "anthropic":
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        else:
            self.ollama_client = OpenAI(base_url=ollama_url, api_key="ollama")

        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        if self.backend == "anthropic":
            return self._generate_anthropic(
                query, conversation_history, tools, tool_manager
            )
        else:
            return self._generate_ollama(
                query, conversation_history, tools, tool_manager
            )

    # ── Anthropic path ───────────────────────────────────────────────────────

    def _generate_anthropic(self, query, conversation_history, tools, tool_manager):
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )
        messages = [{"role": "user", "content": query}]
        return self._run_agentic_loop(messages, system_content, tools, tool_manager)

    def _run_agentic_loop(self, messages, system, tools, tool_manager, max_rounds=2):
        for _ in range(max_rounds):
            api_params = {**self.base_params, "messages": messages, "system": system}
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            response = self.anthropic_client.messages.create(**api_params)

            if response.stop_reason != "tool_use" or tool_manager is None:
                return self._extract_text(response)

            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_blocks:
                return self._extract_text(response)

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in tool_blocks:
                try:
                    result = tool_manager.execute_tool(block.name, **block.input)
                except Exception as e:
                    result = f"Tool execution failed: {str(e)}"
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )
            messages.append({"role": "user", "content": tool_results})

        # max_rounds tool executions done — synthesize without tools
        final_response = self.anthropic_client.messages.create(
            **{**self.base_params, "messages": messages, "system": system}
        )
        return self._extract_text(final_response)

    @staticmethod
    def _extract_text(response) -> str:
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    # ── Ollama path ──────────────────────────────────────────────────────────

    def _generate_ollama(self, query, conversation_history, tools, tool_manager):
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 800,
        }
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"

        response = self.ollama_client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and tool_manager:
            return self._handle_ollama_tool_execution(choice, messages, tool_manager)
        return choice.message.content

    def _handle_ollama_tool_execution(self, choice, messages, tool_manager):
        messages.append(choice.message)

        for tc in choice.message.tool_calls:
            args = json.loads(tc.function.arguments)
            result = tool_manager.execute_tool(tc.function.name, **args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        final_response = self.ollama_client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=800
        )
        return final_response.choices[0].message.content

    @staticmethod
    def _to_openai_tools(anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic tool format to OpenAI/Ollama format."""
        openai_tools = []
        for t in anthropic_tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                }
            )
        return openai_tools
