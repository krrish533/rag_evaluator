import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load env variables
load_dotenv(dotenv_path="../config/api_keys.env")
# load_dotenv()


# Azure OpenAI credentials
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

def generate_summary(text):
    prompt = f"Summarize the following text in 2–3 lines:\n\n{text[:3000]}"  # limit to stay under token limits
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=256
    )
    return response.choices[0].message.content.strip()

def main():
    input_dir = Path("../data/chunks")
    output_dir = Path("../data/summaries/gpt-4o")
    output_dir.mkdir(parents=True, exist_ok=True)

    for txt_file in input_dir.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8").strip()
        if not text:
            print(f"⚠️ Skipping {txt_file.name} — empty text.")
            continue

        print(f"📝 Summarizing {txt_file.name}...")
        summary = generate_summary(text)

        summary_result = {
            "filename": txt_file.name,
            "summary": summary
        }

        output_path = output_dir / f"{txt_file.stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary_result, f, indent=2)

        print(f"✅ Saved summary to {output_path.name}")

if __name__ == "__main__":
    main()
