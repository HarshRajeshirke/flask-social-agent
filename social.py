import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
app = Flask(__name__)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.8
)
parser = JsonOutputParser()

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
You MUST return a single, valid JSON object with two keys: "post" (string) and "hashtags" (an array of strings, starting with '#').
"""
prompt = ChatPromptTemplate.from_template(template=prompt_template)
chain = prompt | llm | parser

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_data = {}

    if request.method == "POST":
        form_data = {
            "topic": request.form.get("topic"),
            "tone": request.form.get("tone"),
            "platform": request.form.get("platform"),
            "word_count": request.form.get("word_count")
        }

        if form_data["topic"]:
            word_count_instruction = f"of about {form_data['word_count']} words" if form_data['word_count'] else "between 50 and 100 words"
            
            result = chain.invoke({
                "topic": form_data["topic"],
                "platform": form_data["platform"],
                "tone": form_data["tone"],
                "word_count_instruction": word_count_instruction
            })
            
    return render_template("index.html", result=result, form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True)