from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
import requests
import os
import streamlit as st


load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# img2text
# let the image understand the scenario based on the photo
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]['generated_text']
    
    return text

# llm
# generate a short story
def generate_story(scenario):
    template = """
    You are a story teller.
    You can generate a short story based on a simple narrative.
    The story should be no more than 20 words.
    
    CONTEXT: {scenario}
    STORY:
    """
    
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
    
    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)
    
    story = story_llm.invoke(input=scenario)
    
    story_text = story.get('text', '')
    
    return story_text

# text to speech
# generate a voice based on the text
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    
    response = requests.post(API_URL, headers=headers, data=message)
    with open('audio.flac', 'wb') as audio_file:
        audio_file.write(response.content)
    
# main function
def main():
    
    st.set_page_config(page_title="image2audio story", page_icon="📖")
    
    st.header("Image2Audio Story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    
    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as f:
            f.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)
        
        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
            
        st.audio("audio.flac")


if __name__ == '__main__':
    main()