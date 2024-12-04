from fastapi import FastAPI, Request,File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors  import CORSMiddleware
from tempfile import NamedTemporaryFile
from fastapi.responses import JSONResponse,FileResponse
from fastapi import HTTPException
from pathlib import Path
import logging
from datetime import datetime
# from summary_sentiment import summarize,sentiment
from summary_sentiment.summarize import summarize
import uuid, os
import shutil
import time
from datetime import timedelta
from src import diarize
from src import *
import json
import pytz


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

origins = [
    "http://localhost",
    "http://0.0.0.0",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://ec2-44-208-238-252.compute-1.amazonaws.com:5500"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



templates = Jinja2Templates(directory="templates/")
app.mount("/static", StaticFiles(directory="static"), name="static")


logger.info("            Loading Diarize Models")
whisper_model, msdd_model, punct_model = diarize.init_models()

# logger.info("            Loading Summarization Model")
# summ_model = summarize.Model(model_dict = "summary_sentiment/MEETING-SUMMARY-BART-LARGE-XSUM-SAMSUM-DIALOGSUM-AMI")
# summ_model = summarize.Model()

# logger.info("            Loading Sentiment Model")
# sentiToken, sentiModel = sentiment.load_sentiment_model()
logger.info("            Model Loading Complete")

json_file_path = "static/data/results/result.json"

@app.get("/")
def form_post(request: Request):
    try : 
        result = "Type a number"
        return templates.TemplateResponse('index.html', context={'request': request, 'result': result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat_history")
async def get_chat_history():
    
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as file:
            data = json.load(file)
        return JSONResponse(content=data)
    else:
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    
@app.get('/get_audios/{id}')
async def get_audio(id):
    audio_path =  f"static/data/audios/{id}.wav"
    
    if os.path.exists(audio_path):
          
        return FileResponse(audio_path)
    else:
        return JSONResponse(content={"error": "File not found"}, status_code=404)
@app.post("/")
def form_post(audioFile: UploadFile = File(...)):
    india_timezone = pytz.timezone('Asia/Kolkata')
    utc_now = datetime.now(pytz.utc)
    startTime = time.time()
    unique_id = str(uuid.uuid4())
    destination_path  = f'static/data/audios/{unique_id}.wav'
    try:
        if audioFile:
            with NamedTemporaryFile(delete=True) as temp:
                with open(temp.name, 'wb') as temp_file:
                    temp_file.write(audioFile.file.read())
                
                transcript = diarize.process(temp.name,whisper_model,msdd_model, punct_model)
                
                # saving to data/audios/....wav
                shutil.copyfile(temp.name, destination_path) 
            
            logger.info("            Now Summarizing Convesations")
            # text = transcript
            
            generated_summary =summarize(transcript)
            
            if generated_summary!="":
                logger.info("            Summary Generated.")
                
            # logger.info("            Sentiment Analysis")
            
            # generated_sentiment = sentiment.inference(generated_summary,sentiToken,sentiModel)
            logger.info("            Analysis Done.")
            try:
                with open(json_file_path, 'r') as file:
                    try:
                        chat_history = json.load(file)
                    except json.JSONDecodeError:
                        chat_history = []
            except FileNotFoundError:
                chat_history = []
            
            endTime = time.time()
            currentTime = utc_now.astimezone(india_timezone)
            response = {"id" :unique_id,
                        'Transcript': transcript,
                        "Summary":generated_summary,
                        # 'Sentiment':generated_sentiment,
                        "TimeTaken":str(timedelta(seconds=endTime - startTime)),
                       "DateTime": currentTime.strftime('%Y-%m-%d %H:%M:%S') }
            
            chat_history.insert(0,response)
            
            with open(json_file_path, "w") as file:
                json.dump(chat_history, file)
             

            return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn 
    uvicorn.run(app,port=8051,host="0.0.0.0")