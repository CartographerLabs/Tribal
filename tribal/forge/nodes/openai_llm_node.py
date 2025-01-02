import time
from tribal.forge.base_nodes import MESSAGE_FORMAT, BaseProcessorNode
from openai import OpenAI, RateLimitError
import json
from tribal.forge.managers.alert_manager import AlertManager
from tribal.forge.managers.log_manager import LogManager

class OpenAILLMNode(BaseProcessorNode):
    def __init__(self, broadcast_manager, api_key, prompt, received_dict_key_to_process, return_keys):
        super().__init__("OpenAI Node", broadcast_manager)
        self.api_key = api_key
        self.prompt = prompt
        self.received_dict_key_to_process = received_dict_key_to_process
        self.return_keys = return_keys if isinstance(return_keys, list) else return_keys.split(',')

    def _call_openai_with_retry(self, client, prompt, text_to_process, function_definition, max_retries=10, wait_time=5):
        retries = 0
        while retries < max_retries:
            try:
                response = client.chat.completions.create(
                    model="gpt-4-0613",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text_to_process}
                    ],
                    functions=[function_definition],
                    function_call={"name": function_definition["name"]}
                )
                return response
            except RateLimitError as e:
                if retries == max_retries - 1:
                    raise e
                LogManager().log(f"Rate limit reached, retrying in {wait_time} seconds... (Attempt {retries + 1}/{max_retries})")
                time.sleep(wait_time)
                retries += 1

    def _process_and_merge(self, api_key, prompt, input_dict, dict_key, mandatory_keys):
        if dict_key not in input_dict:
            raise KeyError(f"The key '{dict_key}' is not present in the input dictionary.")

        if not isinstance(mandatory_keys, list):
            raise TypeError("The mandatory_keys parameter must be a list.")

        client = OpenAI(api_key=api_key)
        text_to_process = input_dict[dict_key]

        json_schema = {
            "type": "object",
            "properties": {
                key: {
                    "type": "string",
                    "description": f"The extracted {key}."
                } for key in mandatory_keys
            },
            "required": mandatory_keys
        }

        function_definition = {
            "name": "process_text",
            "description": "Process the provided text and extract structured data.",
            "parameters": json_schema
        }

        try:
            response = self._call_openai_with_retry(client, prompt, text_to_process, function_definition)

            function_call = getattr(response.choices[0].message, "function_call", None)
            if not function_call or not hasattr(function_call, "arguments"):
                raise ValueError("No function call arguments found in the response.")

            extracted_data = json.loads(function_call.arguments)
            merged_dict = {**input_dict, **extracted_data}

            LogManager().log(f"LLM Response: {str(response)}")
            return merged_dict

        except Exception as e:
            AlertManager().send_alert(f"Error processing OpenAI Request: {e}", self.name)
            return None

    def _process_broadcast(self, act_message):
        try:
            message = act_message[MESSAGE_FORMAT["MESSAGE"]]
            response = self._process_and_merge(
                self.api_key,
                self.prompt,
                message,
                self.received_dict_key_to_process,
                self.return_keys
            )
            if response is None:
                return

            message_to_send = self._construct_message(self.name, response)
            self.send_broadcast(message_to_send, self.name)

        except KeyError as e:
            LogManager().log(f"Missing key in message: {e}", style="bold yellow")
            raise RuntimeError(f"Error processing broadcast: Missing key in message: {e}")
        except Exception as e:
            LogManager().log(f"Unexpected error processing broadcast: {e}", style="bold yellow")
            raise RuntimeError(f"Error processing broadcast: {e}")
