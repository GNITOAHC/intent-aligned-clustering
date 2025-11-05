from ollama import Client
from openai import OpenAI
import http.client
import os


class OuterMedusaLLM:
    def __init__(self, model="gpt-oss-20b"):
        self.model = model
        self.cum_prompt_tokens = 0
        self.cum_completion_tokens = 0

        from dotenv import load_dotenv

        load_dotenv()
        self.OUTER_MEDUSA_ENDPOINT = os.environ.get("OUTER_MEDUSA_ENDPOINT", "")
        self.OUTER_MEDUSA_API_KEY = os.environ.get("OUTER_MEDUSA_API_KEY", "")

        if self.OUTER_MEDUSA_ENDPOINT == "":
            raise ValueError("OUTER_MEDUSA_CLIENT is not set")
        if self.OUTER_MEDUSA_API_KEY == "":
            raise ValueError("OUTER_MEDUSA_API_KEY is not set")

        # Parse and extract host[:port] from OUTER_MEDUSA_ENDPOINT if it is a URL
        from urllib.parse import urlparse

        parsed_endpoint = urlparse(
            self.OUTER_MEDUSA_ENDPOINT
            if self.OUTER_MEDUSA_ENDPOINT.startswith("http")
            else "https://" + self.OUTER_MEDUSA_ENDPOINT
        )
        host = (
            parsed_endpoint.netloc if parsed_endpoint.netloc else parsed_endpoint.path
        )
        # Remove any trailing slashes from host
        self.host = host.rstrip("/")

        conn = self._get_connection()
        try:
            response, response_data = self._get_response(
                conn,
                "GET",
                "/v1/models",
            )
            if response.status != 200:
                raise ValueError(
                    f"Request failed: {response.status} {response.reason} - {response_data}"
                )

        finally:
            conn.close()

        print(f"Connected to Outer Medusa client at {self.OUTER_MEDUSA_ENDPOINT}")

    def change_model(self, model: str):
        import json

        conn = self._get_connection()
        try:
            response, response_data = self._get_response(conn, "GET", "/v1/models")
            if response.status != 200:
                raise ValueError(
                    f"Request failed: {response.status} {response.reason} - {response_data}"
                )
        finally:
            conn.close()

        data = json.loads(response_data)
        models = data.get("data", [])
        if model in [m["id"] for m in models]:
            self.model = model
            return
        else:
            raise ValueError(f"Model {model} not found in available models")

    def _get_connection(self):
        return http.client.HTTPSConnection(self.host)

    def _get_response(self, conn, method, path, headers=None, body=None):
        if headers is None:
            headers = {
                "Authorization": f"Bearer {self.OUTER_MEDUSA_API_KEY}",
                "Content-Type": "application/json",
            }
        conn.request(method, path, body=body, headers=headers)
        response = conn.getresponse()
        response_data = response.read().decode()
        return response, response_data

    def generate(self, prompt: str, sys_prompt: str | None = None):
        import json

        path = "/v1/chat/completions"
        payload = {"model": self.model, "messages": []}
        if sys_prompt:
            payload["messages"].append({"role": "system", "content": sys_prompt})
        payload["messages"].append({"role": "user", "content": prompt})

        conn = self._get_connection()
        try:
            response, response_data = self._get_response(
                conn,
                "POST",
                path,
                body=json.dumps(payload),
            )
            if response.status != 200:
                raise ValueError(
                    f"Request failed: {response.status} {response.reason} - {response_data}"
                )
            data = json.loads(response_data)
            # Response shape: { "choices": [{ "message": { "content": ... } }], "usage": {"prompt_tokens": ..., "completion_tokens": ... } }
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            prompt_eval_count = data.get("usage", {}).get("prompt_tokens", 0)
            eval_count = data.get("usage", {}).get("completion_tokens", 0)

            # Update cumulative token counts
            self.cum_prompt_tokens += prompt_eval_count
            self.cum_completion_tokens += eval_count

            return content, prompt_eval_count, eval_count
        finally:
            conn.close()


class OpenAILLM:
    def __init__(
        self,
        model="gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
    ):
        self.model = model
        self.cum_prompt_tokens = 0
        self.cum_completion_tokens = 0

        from dotenv import load_dotenv

        load_dotenv()
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
        if OPENAI_API_KEY == "":
            raise ValueError("OPENAI_API_KEY is not set")

        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=base_url)

        print("Connected to OpenAI")

    def change_model(self, model: str):
        self.model = model

    def generate(self, prompt: str, sys_prompt: str | None = None):
        request_msg = []
        if sys_prompt and sys_prompt != "":
            request_msg.append({"role": "system", "content": sys_prompt})
        request_msg.append({"role": "user", "content": prompt})

        c = self.client.chat.completions.create(
            model=self.model,
            messages=request_msg,
        )

        if c.usage and c.usage.completion_tokens:
            self.completion_tokens += c.usage.completion_tokens

        if c.usage and c.usage.prompt_tokens:
            self.prompt_tokens += c.usage.prompt_tokens

        return (
            str(c.choices[0].message.content),
            c.usage.prompt_tokens if c.usage and c.usage.prompt_tokens else 0,
            c.usage.completion_tokens if c.usage and c.usage.completion_tokens else 0,
        )


class OllamaLLM:
    def __init__(self, model="gpt-oss:20b"):
        self.model = model

        from dotenv import load_dotenv

        load_dotenv()
        OLLAMA_CLIENT = os.environ.get("OLLAMA_CLIENT", "http://localhost:11434")

        print(f"Connecting to Ollama client at {OLLAMA_CLIENT}")
        self.client = Client(host=OLLAMA_CLIENT)
        print("Connected")

    def generate(self, prompt: str, sys_prompt: str | None = None):
        messages: list[dict] = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat(
            model=self.model,
            messages=messages,
            keep_alive=0,
        )
        prompt_eval_count = response.get("prompt_eval_count", 0)
        eval_count = response.get("eval_count", 0)
        return str(response.message.content), prompt_eval_count, eval_count


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    # llm = OllamaLLM()
    llm = OuterMedusaLLM()
    test_sys_prompt = "You are a helpful assistant."
    test_prompt = "Why is the sky blue?"
    response, prompt_eval_count, eval_count = llm.generate(test_prompt, test_sys_prompt)

    # print("Response:", response)
    # print("Prompt Eval Count:", prompt_eval_count)
    # print("Eval Count:", eval_count)

    llm.change_model("gpt-oss-120b")
