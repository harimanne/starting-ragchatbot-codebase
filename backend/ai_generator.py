import json
import anthropic
from openai import OpenAI
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Claude (Anthropic) or a local Ollama LLM"""

    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **One search per query maximum**
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str, backend: str = "anthropic",
                 ollama_url: str = "http://localhost:11434/v1", ollama_model: str = "llama3.1"):
        self.backend = backend
        self.model = model if backend == "anthropic" else ollama_model

        if backend == "anthropic":
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
        else:
            self.ollama_client = OpenAI(base_url=ollama_url, api_key="ollama")

        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }

    def generate_response(self, query: str,
                          conversation_history: Optional[str] = None,
                          tools: Optional[List] = None,
                          tool_manager=None) -> str:
        if self.backend == "anthropic":
            return self._generate_anthropic(query, conversation_history, tools, tool_manager)
        else:
            return self._generate_ollama(query, conversation_history, tools, tool_manager)

    # ── Anthropic path (unchanged logic) ────────────────────────────────────

    def _generate_anthropic(self, query, conversation_history, tools, tool_manager):
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history else self.SYSTEM_PROMPT
        )
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content
        }
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        response = self.anthropic_client.messages.create(**api_params)

        if response.stop_reason == "tool_use" and tool_manager:
            return self._handle_anthropic_tool_execution(response, api_params, tool_manager)
        return response.content[0].text

    def _handle_anthropic_tool_execution(self, initial_response, base_params, tool_manager):
        messages = base_params["messages"].copy()
        messages.append({"role": "assistant", "content": initial_response.content})

        tool_results = []
        for block in initial_response.content:
            if block.type == "tool_use":
                result = tool_manager.execute_tool(block.name, **block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        final_response = self.anthropic_client.messages.create(**{
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        })
        return final_response.content[0].text

    # ── Ollama path ──────────────────────────────────────────────────────────

    def _generate_ollama(self, query, conversation_history, tools, tool_manager):
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history else self.SYSTEM_PROMPT
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query}
        ]

        kwargs = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": 800}
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
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })

        final_response = self.ollama_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=800
        )
        return final_response.choices[0].message.content

    @staticmethod
    def _to_openai_tools(anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic tool format to OpenAI/Ollama format."""
        openai_tools = []
        for t in anthropic_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {})
                }
            })
        return openai_tools
