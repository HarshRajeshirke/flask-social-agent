import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. Initialization and Setup ---
load_dotenv()
# FIX: Corrected __name__ (double underscores)
app = Flask(__name__)

# --- 2. LangChain and AI Model Setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.8
)
parser = JsonOutputParser()

# The main prompt template that uses all user inputs
prompt_template = """
You are a creative and expert social media assistant. Your task is to generate a post based on the user's specifications.
*Topic:* "{topic}"
*Platform:* {platform}
*Tone:* **{tone}**
*Task Details:*
1. Analyze the topic provided.
2. Write a single, engaging post that is {word_count_instruction}.
3. The post must be tailored for the specified platform and strictly adhere to the specified tone.
4. Suggest 3-5 relevant hashtags.
*Output Format:*
You MUST return a single, valid JSON object with two keys: "post" (string) and "hashtags" (an array of strings, starting with '#').
"""
prompt = ChatPromptTemplate.from_template(template=prompt_template)
chain = prompt | llm | parser

# --- 3. Flask Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {}

    # NEW: Smart dictionary for platform-specific lengths
    platform_lengths = {
        "Twitter": "concise and under 280 characters",
        "Instagram": "visually engaging and around 100-150 words",
        "LinkedIn": "professional and informative (around 150-200 words)",
        "Facebook": "friendly and descriptive (between 100 and 250 words)",
        "General": "between 50 and 100 words" # Default length
    }

    if request.method == "POST":
        # Store all form inputs (Word Count input is REMOVED from collection)
        form_data = {
            "topic": request.form.get("topic"),
            "tone": request.form.get("tone"),
            "platform": request.form.get("platform"),
        }

        if form_data["topic"]:
            # Automatically get the correct instruction based on platform selection
            word_count_instruction = platform_lengths.get(form_data["platform"], platform_lengths["General"])
            
            # Run the AI chain
            result = chain.invoke({
                "topic": form_data["topic"],
                "platform": form_data["platform"],
                "tone": form_data["tone"],
                "word_count_instruction": word_count_instruction # Pass the automatically determined instruction
            })
            
    # Ensure form_data is always passed so inputs are remembered
    return render_template("index.html", result=result, form_data=form_data)

# --- 4. Run the Application ---
if __name__ == "__main__":
    app.run(debug=True)