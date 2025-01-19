# Llama-3-with-Hugging-Face-Transformers


## Features

- **Model Support:** Works with Llama-3–8B-Instruct and Llama-3–8B.
- **Quantization:** Reduces GPU memory requirements (~8 GB).
- **Customizable Chatbot:** Interactive chatbot with user and assistant roles.
- **Extensibility:** Easily deployable via Flask or FastAPI.

---

## Requirements

### **Hardware Requirements**
- **Minimum GPU Memory:** 8 GB (with quantization).
- **Recommended GPU Memory:** 20 GB (without quantization).

### **Software Requirements**
- Python 3.8+
- PyTorch with CUDA support
- Hugging Face Transformers, Accelerate, and BitsAndBytes

---

## Installation

### **1. Create a Virtual Environment (Recommended)**
```bash
python -m venv env_name
source env_name/bin/activate  # Linux/Mac
env_name\Scripts\activate  # Windows
```

### **2. Install Required Packages**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install accelerate transformers bitsandbytes
```

### **3. Download the Model**
1. Create a Hugging Face account: [Hugging Face](https://huggingface.co/).
2. Generate an access token: [Access Token](https://huggingface.co/settings/tokens).
3. Log in to Hugging Face:
    ```bash
    huggingface-cli login
    ```
4. Accept the model's terms: [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
5. Download the model:
    ```bash
    huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --exclude "original/*" --local-dir meta-llama/Meta-Llama-3-8B-Instruct
    ```

---

## Usage

### **Implementation Code**
Save the following code as `llama3_chatbot.py`:

```python
import torch
import transformers

class Llama3:
    def __init__(self, model_path):
        self.model_id = model_path
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids(""),
        ]

    def get_response(
        self, query, message_history=[], max_tokens=4096, temperature=0.6, top_p=0.9
    ):
        user_prompt = message_history + [{"role": "user", "content": query}]
        prompt = self.pipeline.tokenizer.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"][len(prompt):]
        return response, user_prompt + [{"role": "assistant", "content": response}]

    def chatbot(self, system_instructions=""):
        conversation = [{"role": "system", "content": system_instructions}]
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the chatbot. Goodbye!")
                break
            response, conversation = self.get_response(user_input, conversation)
            print(f"Assistant: {response}")

if __name__ == "__main__":
    bot = Llama3("your-model-path-here")
    bot.chatbot()
```

### **4. Run the Chatbot**
```bash
python llama3_chatbot.py
```

---

## Notes

- **Model Path:** Replace `your-model-path-here` with the path to the downloaded Llama-3 model.
- **Exit Chatbot:** Type `exit` or `quit` to close the chatbot.
- **Deployment:** For a production setup, consider using Flask or FastAPI to create an API endpoint.

---

## License
This repository is open-source and licensed under the MIT License.
