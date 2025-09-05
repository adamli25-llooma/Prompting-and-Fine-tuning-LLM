#streamlit run your_script.py [-- script args] 
# maybe run this in terminal?
# or this: python -m streamlit run your_script.py
#magic commands? dont have to run like st.write to write something?
"""
# My first app
Here's our first attempt at using data to create a table:


import streamlit as st
import pandas as pd
import numpy as np
st.write("Here's our first attempt at using data to create a table:")
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40],
  'sigma column': ["oi", "anri-~chan", 45, 20]
})

dataframe = np.random.randn(10, 20)
st.dataframe(dataframe)

df
"""
import streamlit as st
from openai import OpenAI
import time
import requests
import base64
from io import BytesIO
from PIL import Image
import json

flux_key = "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"

# Initialize client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

# Set page config
st.set_page_config(page_title="Deepseek Prompt-Testing Chatbot", layout="centered")

# Custom dark theme styling, 
st.markdown("""
    <style>
    body { background-color: #000000; color: white; }
    .stTextInput > div > div > input {
        background-color: #111111;
        color: white;
        border: 1px solid #333;
    }
    .response-box {
        background-color: #111;
        padding: 1em;
        border-radius: 0.5em;
        margin-top: 1em;
        border: 1px solid #333;
    }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown("<h1 style='text-align: center;'>Deepseek Prompt-Testing Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>How can I help you today? (swimming oriented) </h3>", unsafe_allow_html=True)

# Prompt suggestions
with st.expander("Try asking something like..."):
    st.markdown("- Which number is larger, 9.11 or 9.8?")
    st.markdown("- How many 'r's are in 'strawberry'?")
    st.markdown("- What is a fast 100 free time for a 15-year-old?")

selected_model = st.selectbox("Choose model:", ["Deepseek (Text Reasoning)", "Llama (Image Identifier)", "Flux (Image Generator)"])
user_prompt = st.selectbox("Choose topic:", ["Swimmer🏊‍♂️", "BallKnower🏀", "Detective🔍", "None"])

# User input
user_input = st.text_input("Type your question below:", "")

# File uploader
uploaded_file = st.file_uploader("Optional: Upload a file (text or image)", type=["txt", "md", "csv", "json", "py", "png"])


if "sys_prompt" not in st.session_state:

    d_history = [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional swimming coach, sports analyst, and educator."
                            "Your job is to provide deeply researched, thoughtful, and accurate insights about competitive swimming. "
                            "You should prioritize Olympic history, technical swimming mechanics, and global performance data. "
                            "Think carefully, step by step, and do not stop until the user’s question is completely resolved. "
                            "If unsure, explain your reasoning openly instead of guessing. "
                            "Be concise."
                            "After you have finished your thinking out loud, make sure to specify in your reponse (as to help the user) that you have started actually answering their question by starting a new line with 'Actual Response:' and then generating below it."
                            )
                    }

                ]
    
    st.session_state["sys_prompt"] = d_history

if "basketball_prompt" not in st.session_state:
    st.session_state["basketball_prompt"] = [ {
        "role": "system", "content": (
            "Your job is to provide deeply researched, thoughtful, and accurate insights about basketball at all levels of play."
            "You should prioritize NBA and international history, player comparisons, advanced metrics, in-game strategy, and player development techniques."
            "Think carefully, step by step, and do not stop until the user’s question is completely resolved."
            "If unsure, explain your reasoning openly instead of guessing."
            "Be concise."
            "After you have finished your thinking out loud, make sure to specify in your response (as to help the user) that you have started actually answering their question by starting a new line with 'Actual Response:' and then generating below it.")
    }]

if "detective_prompt" not in st.session_state:
    st.session_state["detective_prompt"] = [{
    "role": "system", "content": (
        "You are a professional detective, investigative analyst, and logical problem solver."
        "Your job is to carefully examine clues, evaluate evidence, and provide well-reasoned, accurate insights to solve mysteries or puzzles."
        "You should prioritize logical deduction, step-by-step reasoning, psychological profiling, timeline construction, and pattern recognition."
        "Think critically and do not jump to conclusions—every assumption must be justified."
        "If unsure, explain your reasoning and what further information would be needed."
        "Be concise, precise, and methodical in your thought process."
        "After you have finished your thinking out loud, make sure to specify in your response (as to help the user) that you have started actually answering their question by starting a new line with 'Actual Response:' and then generating below it.")
}]

if 'no_prompt' not in st.session_state:
    st.session_state["no_prompt"] = []

if "follow_up" not in st.session_state: 
    st.session_state["follow_up"] = True

if "convo" not in st.session_state:
    st.session_state["convo"] = []



def click_f():
    st.session_state["follow_up"] = True

def click_nf():
    st.session_state["follow_up"] = False





# Button to trigger completion
if st.button("Ask"):
    if user_input.strip():
        # Show output box
        with st.container():
            st.markdown("<div class='response-box'><b>Response:</b><br>", unsafe_allow_html=True)
            response_placeholder = st.empty()

            full_response = ""

            #ERROR list index for Actual Response doesn't always work
            #BUG with default being no saving so first message isnt kept (have to run twice before?)
            #saving works, two question error. save messaage as something else, not the full prompt
            #UI help for models, indicating image reader or generator or reasoning
        
            if selected_model == "Deepseek (Text Reasoning)" :
                # Call NVIDIA DeepSeek API

                #could change topic function, append to history, two different states variables
                #play around with no prompts, diff prompts, testing
                #select system prompt/type of model dropdown
                #swimming + other prompts title on page

                #message1 = st.session_state["sys_prompt"] #reference issue?
                
                #print('WOW')
             #  print(message1) #before
             #message1.append(user_q)
               #print([st.session_state["sys_prompt"][0], user_q])
               # print(" ")
             #  print(st.session_state["sys_prompt"])
                #print("END")

                user_q = {"role": "user", "content": user_input}
                if user_prompt == "Swimmer🏊‍♂️":
                    state_prompt = "sys_prompt"

                if user_prompt == "BallKnower🏀":
                    state_prompt = "basketball_prompt"

                if user_prompt == "Detective🔍":
                    state_prompt = "detective_prompt"
                
                if user_prompt == "None":
                    state_prompt = "no_prompt"
                
                if user_prompt == None:
                    st.write("you didnt choose a prompt!!")

                 #think i should use choices method?

                completion = client.chat.completions.create(
                    model="deepseek-ai/deepseek-r1",
                    messages = st.session_state[state_prompt] + st.session_state["convo"], #are prompts the same?
                    temperature=0.6,
                    top_p=0.7,
                    max_tokens=4096,
                    stream=True
                )
                
                #how am i using convo?
                
                print(" ")
                print("HI")
                print()
                print(" ")


                #APPEND COMPLETIONS INSTEAD? into content or system
               #print(completion.choices[0].message.role)
        
        

                #make sure to separate full response -> if u doesn't want to show reasoning, 
                with st.spinner("🤔 Model is thinking..."):
                    for chunk in completion:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            #response_placeholder.markdown(full_response, unsafe_allow_html=True)

                    #can do thinking loop, multiple models
                    #add model is thinking ui 
                    #better understand CSS, different themes etc
                    #can i ask a followup question that will utilize knowledge from previous response
                        #send it first message with system prompt + response and new question
                    tailored_response = full_response.split("Actual Response:")[1]
                    reasoning = full_response.split("Actual Response:")[0]
                    print(reasoning)        
                    looped_response = tailored_response.split(" ")
                    final_response = ''
                    for word in looped_response: 
                        final_response += (word + " ")
                        time.sleep(0.09)
                        response_placeholder.markdown(final_response, unsafe_allow_html=True)


                    st.markdown("</div>", unsafe_allow_html=True)

                st.session_state["convo"].append({"role": "system", "content": "ONLY use this information when user references past questions. For example if user asks with questions like 'he' pull from here: " + full_response})

                st.write("Would You Like to Continue the Conversation?")
                st.write("Click 'Yes' or 'No'")
                st.button("Yes", on_click = click_f) #make button better/more clear (after generation)
                st.button("No", on_click = click_nf)

                if st.session_state["follow_up"]:
                    st.write("message history is being kept!")
                    #st.session_state["convo"].append(user_q)

                if st.session_state["follow_up"] == False:
                    st.session_state["convo"] = [] #does this work? making a list then appending the dict
                    st.write("message history is not being kept")




            if selected_model == "Llama (Image Identifier)":

                invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
                stream = True

                if uploaded_file is not None:
                    image_bytes = uploaded_file.read()
                    image_b64 = base64.b64encode(image_bytes).decode()

                    assert len(image_b64) < 180_000, \
                    "To upload larger images, use the assets API (see docs)"
                    

                    headers = {
                    "Authorization": "Bearer " + flux_key,
                    "Accept": "text/event-stream" if stream else "application/json"
                    }

                    payload = {
                    "model": 'meta/llama-4-scout-17b-16e-instruct',
                    "messages": [ #maybe give context for specific photos (like swim?)
                        {
                        "role": "user",
                        "content": f'What is in this image? <img src="data:image/png;base64,{image_b64}" />'
                        }
                    ],
                    "max_tokens": 512,
                    "temperature": 1.00,
                    "top_p": 1.00,
                    "stream": stream
                    }
                response = requests.post(invoke_url, headers=headers, json=payload, stream=True)

                full_response = ""
                response_placeholder = st.empty()

                with st.spinner("🦙 Llama is analyzing the image..."):
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode("utf-8")
                            if line_str.startswith("data: "):
                                line_str = line_str[6:]

                            if line_str.strip() == "[DONE]":
                                break

                            try:
                                json_data = json.loads(line_str)
                                delta = json_data["choices"][0]["delta"].get("content", "")
                                full_response += delta
                                response_placeholder.markdown(full_response, unsafe_allow_html=True)
                            except Exception as e:
                                print(f"Error parsing stream chunk: {e}")
                st.markdown("</div>", unsafe_allow_html=True)



            
            if selected_model == "Flux (Image Generator)": #only about 20 generations left for images
                invoke_url = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.1-schnell"

                headers = {
                    "Authorization": "Bearer " + flux_key,
                    "Accept": "application/json",
                }

                payload = {
                    "prompt": user_input or "coffee shop interior",
                    "width": 1024,
                    "height": 1024,
                    "seed": 0,
                    "steps": 4
                }

                with st.spinner("🖼️ Generating image..."):
                    response = requests.post(invoke_url, headers=headers, json=payload)
                    response.raise_for_status()

                    response_body = response.json()
                    #st.json(response_body) , helps debug
                    # Get the base64 image string from the response
                    base64_image = response_body["artifacts"][0]["base64"]

                    # Decode the base64 string
                    image_data = base64.b64decode(base64_image)

                    # Convert to a PIL Image
                    image = Image.open(BytesIO(image_data))

                    # Show the image in Streamlit
                    st.image(image, caption="Generated by Flux", use_container_width=True)

                
            #bug, model prints this even when i choose llama, both statements print weird, maybe this code is not right (just say if model selected w)
            if (selected_model != "Deepseek" or selected_model != "Llama" or selected_model != "Flux"): 
                print("what the helly model did u select dawgers, go back and choose one fr no cap ts pmo frfr")

    else:
        st.warning("Please enter a prompt.")
