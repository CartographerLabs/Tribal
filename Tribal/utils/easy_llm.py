import re
from typing import Any, Dict, List, Tuple, Type, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, logging as hf_logging
from pydantic import BaseModel, Field, create_model, RootModel
import json
import random
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Suppress unnecessary warnings
hf_logging.set_verbosity_error()

UNSLOTH_MODELS = ["unsloth/Llama-3.1-Storm-8B-bnb-4bit"]

class EasyLLM:
    def __init__(self, max_new_tokens: int = 350, model_name: str = None) -> None:
        self.max_new_tokens = max_new_tokens
        if model_name is None:
            model_name = random.choice(UNSLOTH_MODELS)
            print(f"No model chosen, model {model_name} selected.")
        
        self.model_name = model_name
        self.dialogue: List[dict] = []
        self._device: str = "cuda"
        self.model = None
        self.tokenizer = None

    def _load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if self.model is None or self.tokenizer is None:
            is_4bit = '4bit' in self.model_name.lower()
            is_8bit = '8bit' in self.model_name.lower()

            if is_4bit or is_8bit:
                if is_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type='nf4',
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.bfloat16,
                    )

                max_memory = {
                    0: "10GiB",  # Adjust based on GPU memory availability
                    "cpu": "50GiB"  # Allocate more for offloading
                }

                device_map = "auto"

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map=device_map,
                    max_memory=max_memory,
                    offload_folder="offload",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map='auto',
                )

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        return self.model, self.tokenizer

    def _unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        torch.cuda.empty_cache()  # Clears GPU memory
        torch.cuda.ipc_collect()  # Forces release of GPU memory

    def _generate_dialogue_response(self, messages: List[dict]) -> str:
        self._load_model()

        chat_template = getattr(self.tokenizer, 'chat_template', None)
        if chat_template:
            input_data = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
            if isinstance(input_data, dict):
                input_ids = input_data["input_ids"].to(self._device)
                attention_mask = input_data.get("attention_mask", torch.ones_like(input_ids)).to(self._device)
            elif isinstance(input_data, torch.Tensor):
                input_ids = input_data.to(self._device)
                attention_mask = torch.ones_like(input_ids).to(self._device)
            else:
                raise TypeError(f"Unexpected type for input_data: {type(input_data)}")
        else:
            prompt = self.format_messages(messages)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self._device)
            attention_mask = torch.ones_like(input_ids).to(self._device)

        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_tokens = generated_ids[:, input_ids.shape[-1]:]
        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        # Clean up
        del input_ids, attention_mask, generated_ids, generated_tokens
        torch.cuda.empty_cache()

        self._unload_model()

        return decoded.strip()

    def ask_question(self, question: str, reset_dialogue: bool = True) -> str:
        if reset_dialogue:
            self.reset_dialogue()

        self._load_model()

        chat_template = getattr(self.tokenizer, 'chat_template', None)
        roles = self.extract_roles_from_template(chat_template) if chat_template else []
        message_roles = self.get_message_roles(roles)

        if message_roles['user']:
            self.dialogue.append({"role": message_roles['user'], "content": question})
        else:
            self.dialogue.append({"content": question})

        messages_for_model = self.dialogue.copy()

        result = self._generate_dialogue_response(messages_for_model)

        if message_roles['assistant']:
            self.dialogue.append({"role": message_roles['assistant'], "content": result})
        else:
            self.dialogue.append({"content": result})

        if reset_dialogue:
            self.reset_dialogue()

        self._unload_model()

        result = result.replace("json", "").replace("\n", " ").replace("   ", " ").replace("  ", " ")

        try:
            return json.loads(result)
        except:
            try:
                return self.parse_llm_json(result)
            except:
                preamble, *resp = result.split(":")
                resp = "".join(resp)
                return json.loads(resp)

    def reset_dialogue(self) -> None:
        self.dialogue = []

    def extract_roles_from_template(self, chat_template: str) -> List[str]:
        roles = re.findall(r'message\["role"\]\s*==\s*"([^"]+)"', chat_template)
        return list(set(roles))

    def get_message_roles(self, roles: List[str]) -> Dict[str, str]:
        if not roles:
            return {'user': 'user', 'assistant': 'assistant'}

        user_role = assistant_role = None

        for role in roles:
            role_lower = role.lower()
            if 'user' in role_lower and not user_role:
                user_role = role
            elif 'assistant' in role_lower and not assistant_role:
                assistant_role = role
            elif 'system' in role_lower and not assistant_role:
                assistant_role = role

        user_role = user_role or 'user'
        assistant_role = assistant_role or 'assistant'

        return {'user': user_role, 'assistant': assistant_role}

    def format_messages(self, messages: List[dict]) -> str:
        formatted = ""
        for message in messages:
            content = message["content"]
            role = message.get("role")
            if role:
                formatted += f"{role}: {content}\n"
            else:
                formatted += f"{content}\n"
        return formatted.strip()

    def parse_llm_json(self, llm_response: str) -> dict:
        llm_response = llm_response.replace("\\", "")
        match = re.search(r'```.*?({.*?}).*?```', llm_response, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to decode JSON: {e}")
        else:
            raise Exception(f"Failed to parse JSON from: '{llm_response}'")

    def generate_json_prompt(self, schema: Type[BaseModel], query: str) -> str:
        """
        Generates a JSON prompt based on a given schema and query.

        Args:
            schema (Type[BaseModel]): The Pydantic schema to structure the output.
            query (str): The query for which the JSON response is to be generated.

        Returns:
            str: The JSON-formatted prompt.
        """
        # Ensure schema is a class (Type[BaseModel])
        if not isinstance(schema, type) or not issubclass(schema, BaseModel):
            raise TypeError("The schema argument must be a Pydantic model class.")

        # Initialize the JSON output parser with your schema
        parser = JsonOutputParser(pydantic_object=schema)

        # Create a prompt template with the format instructions injected
        prompt = PromptTemplate(
            template="Answer the following query in JSON format according to the provided schema:\n{format_instructions}\n\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        # Generate the response using the model and parser
        response = prompt.format(query=query)

        return response

    @staticmethod
    def parse_field(field_name: str, field_value: Any) -> tuple:
        if isinstance(field_value, str):
            return (str, Field(description=field_value))
        elif isinstance(field_value, int):
            return (int, Field(description=field_value))
        elif isinstance(field_value, bool):
            return (bool, Field(description=field_value))
        elif isinstance(field_value, list):
            if field_value and isinstance(field_value[0], dict):
                return (List[EasyLLM.generate_pydantic_model_from_json_schema(f"{field_name}_item", field_value[0])], ...)
            return (List[Any], Field(description="List of values"))
        elif isinstance(field_value, dict):
            return (EasyLLM.generate_pydantic_model_from_json_schema(field_name.capitalize(), field_value), ...)
        else:
            return (Any, Field(description="Unknown field type"))

    @staticmethod
    def generate_pydantic_model_from_json_schema(schema_name: str, json_schema: Union[str, Dict[str, Any], List[Any]]) -> Type[BaseModel]:
        if isinstance(json_schema, str):
            json_schema = json.loads(json_schema)

        if isinstance(json_schema, dict):
            fields = {
                field_name: EasyLLM.parse_field(field_name, field_value)
                for field_name, field_value in json_schema.items()
            }
            return create_model(schema_name, **fields)

        elif isinstance(json_schema, list):
            if json_schema and isinstance(json_schema[0], dict):
                return RootModel[List[EasyLLM.generate_pydantic_model_from_json_schema(f"{schema_name}_item", json_schema[0])]]
            else:
                return RootModel[List[Any]]

        else:
            raise ValueError("The provided JSON schema must be a dictionary, list, or valid JSON string.")
