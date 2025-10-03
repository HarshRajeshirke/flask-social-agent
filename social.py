import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# --- 1. Initialization ---
# This loads your API key from the .env file for local testing
# On Render, it will use the Environment Variable you set up
load_dotenv()
app = Flask(__name__)

# --- 2. LangChain Setup ---
# THIS IS THE CORRECTED LINE:
# It explicitly tells LangChain to use the Google API key from the environment variables.
# This is the most reliable way to ensure it works both locally and when deployed.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.8
)
parser = JsonOutputParser()

# The prompt now includes the {word_count_instruction} placeholder
prompt_template = """
You are a creative and expert social media assistant. Your task is to generate a post based on the user's specifications.

*Topic:* "{topic}"
*Platform:* {platform}
*Tone:* **{tone}**

*Task Details:*
1. Analyze the topic provided.
2. Write a single, engaging post {word_count_instruction}.
3. The post must be tailored for the specified platform and strictly adhere to the specified tone.
4. Suggest 3-5 relevant hashtags.

*Output Format:*
You MUST return a single, valid JSON object with two keys:
- "post": A string containing the generated post.
- "hashtags": An array of strings, where each string is a hashtag (starting with '#').
"""
prompt = ChatPromptTemplate.from_template(template=prompt_template)
chain = prompt | llm | parser

# --- 3. Flask Web Routes ---
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        topic = request.form.get("topic")
        tone = request.form.get("tone")
        platform = request.form.get("platform")
        # Get the word_count value from the form
        word_count = request.form.get("word_count")

        if topic:
            # Create the word count instruction for the AI
            word_count_instruction = f"of about {word_count} words" if word_count else "between 50 and 100 words"
            
            # Run the chain, now including the word count instruction
            result = chain.invoke({
                "topic": topic,
                "platform": platform,
                "tone": tone,
                "word_count_instruction": word_count_instruction
            })
            
    return render_template("index.html", result=result)

# --- 4. Run the Application ---
if __name__ == "__main__":
    app.run(debug=True)
