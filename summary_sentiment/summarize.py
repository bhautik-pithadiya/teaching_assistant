from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import runnable
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Initialize the openai language model
openai = OpenAI(
    model_name="gpt-3.5-turbo-instruct", 
    openai_api_key=api_key)

template = PromptTemplate(
    input_variables=["text"],
    template='''
    Summarize the following text: {text}
    ''',
)

# @runnable
chain = template | openai


def summarize(text):    
    """
    Summarizes the given text using a language model.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.
    """
    return chain.invoke({"text":text})  

if __name__ == "__main__":
    print(summarize('''
                    Hallo, hallo, Sie haben Nicole Liev erreicht. Es tut mir leid, ich kann Ihren Anruf nicht entgegennehmen. Wenn Sie eine Nachricht entlassen möchten, werde ich mich so schnell wie möglich bei Ihnen melden. Vielen Dank, bitte sprechen Sie Ihre Nachricht nach dem Signalton. Nachdem Sie Ihre Nachricht aufgezeichnet haben, können Sie auflegen oder die 1 für weitere Optionen drücken.". "Hallo Petta, ich habe gerade Ihren Lebenslauf erhalten. Könnten Sie mir bitte Ihre Gehaltserwartungen für das Jahr mitteilen, damit ich Ihnen den RTR senden kann und wir das bestätigen können. Danach werde ich Ihr Profil einreichen. Bitte teilen Sie mir also Ihre Gehaltserwartungen mit, damit ich den RTR weitergeben kann. Vielen Dank und einen schönen Tag.'''))