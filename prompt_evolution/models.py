from enum import StrEnum
import ollama
from ollama import ChatResponse
from pydantic import BaseModel

class ModelNames(StrEnum):
    qwen_math = "qwen2-math:1.5b"
    gemma3 = "gemma3:4b"
    qwen3 = "qwen3:4b"

MODEL_NAME = ModelNames.gemma3


class LLM: 
    def __init__(self, developer_message: str, model_name: str = MODEL_NAME, memory = None):
        self.developer_message = developer_message
        self.model_name = model_name
        self.memory = memory

    def _format_prompt(self, role: str, content: str) -> dict[str, str]:
        return {"role": role, "content": content}

    @property
    def developer_prompt(self) -> dict[str, str]:
        if self.memory:
            # TODO Add logic to append memory to the developer prompt
            return self._format_prompt(role="developer", content=self.developer_message)

        return self._format_prompt(role="developer", content=self.developer_message)

    def call(self, user_message: str, format: BaseModel | None = None) -> ChatResponse:
        user_prompt = self._format_prompt(role="user", content=user_message)
        model_input = [self.developer_prompt, user_prompt]
        return ollama.chat(model=self.model_name, messages=model_input, format=format.model_json_schema())

class Answer(BaseModel):
    answer: int

class Solver(LLM):
    def __init__(self, developer_message):
        super().__init__(developer_message=developer_message)
        self.answers = None


    def solve(self, problems: list[str]) -> Answer: 
        user_message = f"/no_think solve these problems: {problems}"
        response = self.call(user_message=user_message, format=Answer)
        
        return Answer.model_validate_json(response.message.content)
        


# class Evolver(LLM):
#     def __init__(self, developer_message, memory=None):
#         super().__init__(developer_message=developer_message, memory=memory)

#     def update_prompt(solver_)


#     def update_prompt(self, solver_prompt, marked_homework) -> solver_prompt:
#         self.call()


