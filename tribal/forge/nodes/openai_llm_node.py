from tribal.forge.base_nodes import MESSAGE_FORMAT, BaseProcessorNode
from openai import OpenAI
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

    def _process_and_merge(self, api_key, prompt, input_dict, dict_key, mandatory_keys):
        """
        Processes text from a dictionary value using OpenAI API with a prompt, and returns
        the merged dictionary that includes the extracted data following a JSON schema.

        Parameters:
            api_key (str): OpenAI API key.
            prompt (str): The prompt to be used by the OpenAI model.
            input_dict (dict): A dictionary containing the input data.
            dict_key (str): The key in the dictionary whose value will be processed.
            mandatory_keys (list): A list of key names to add into the schema as mandatory.

        Returns:
            dict: The merged dictionary combining the input dictionary and the generated data.
        """
        if dict_key not in input_dict:
            raise KeyError(f"The key '{dict_key}' is not present in the input dictionary.")

        if not isinstance(mandatory_keys, list):
            raise TypeError("The mandatory_keys parameter must be a list.")

        # Initialize the OpenAI client
        client = OpenAI(api_key=api_key)

        # Extract the text to process
        text_to_process = input_dict[dict_key]

        # Dynamically construct the JSON schema
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

        # Define the function schema for the OpenAI API
        function_definition = {
            "name": "process_text",
            "description": "Process the provided text and extract structured data.",
            "parameters": json_schema
        }

        # Call the OpenAI API
        try:
            response = client.chat.completions.create(
                model="gpt-4-0613",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text_to_process}
                ],
                functions=[function_definition],
                function_call={"name": function_definition["name"]}  # Explicitly enforce function call
            )

            # Safely access function_call
            function_call = getattr(response.choices[0].message, "function_call", None)
            if not function_call or not hasattr(function_call, "arguments"):
                raise ValueError("No function call arguments found in the response.")

            extracted_data = json.loads(function_call.arguments)

            # Merge the extracted data with the input dictionary
            merged_dict = {**input_dict, **extracted_data}


            LogManager().log(f"LLM Response: {str(response)}")

            return merged_dict

        except Exception as e:
            AlertManager().send_alert(f"Error processing OpenAI Request: {e}", self.name)

    def _process_broadcast(self, act_message):
        """
        Handles the broadcast message, processes it using OpenAI API, and sends a response.

        Parameters:
            act_message (dict): The received broadcast message containing data to process.
        """
        try:
            # Extract the message from the act_message
            message = act_message[MESSAGE_FORMAT["MESSAGE"]]

            # Process and merge the data using OpenAI
            response = self._process_and_merge(
                self.api_key,
                self.prompt,
                message,
                self.received_dict_key_to_process,
                self.return_keys
            )

            # Construct and send the processed message
            message_to_send = self._construct_message(self.name, response)
            self.send_broadcast(message_to_send, self.name)

        except KeyError as e:
            logging.error(f"Missing key in message: {e}")
            raise RuntimeError(f"Error processing broadcast: Missing key in message: {e}")
        except Exception as e:
            logging.error(f"Unexpected error processing broadcast: {e}")
            raise RuntimeError(f"Error processing broadcast: {e}")
