from flask import Flask, render_template, request,send_file,redirect, url_for, session #Vish
import random
import os
import datetime
import openai

import uuid
from collections import defaultdict

from waitress import serve
from werkzeug.datastructures import MultiDict
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback # 07/08
from langchain.chains import RetrievalQA   # 07/08
import gspread  # For Excel
from oauth2client.service_account import ServiceAccountCredentials # For Excel
from langchain.chains import ConversationalRetrievalChain   #Vish

app = Flask(__name__)
app.secret_key = "key"

#For Excel
scope = ['https://www.googleapis.com/auth/spreadsheets', "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json',scope)
client = gspread.authorize(creds)
sheet = client.open('Data101').sheet1
#For Excel
openai.api_key = 'API_KEY_HERE'
os.environ["OPENAI_API_KEY"] = 'API_KEY_HERE'


#conversation = []
conversation = {}
chat_history = {} 

DOCUMENT_FOLDER = 'Folder_Location'

USERS = {
    'Stores username/password as a key/value pair'
}

#Now
def generate_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
#Now

@app.route('/')
def index():
    conversation.clear()
    #Now
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    #Now
    print("session: ",session['session_id'])    
    if 'user_role' not in request.args:
        return render_template('login.html')
    elif request.args['user_role'] == 'teacher':
        chat_history[session['session_id']] = []
        return render_template('index.html', messages=conversation, user_role='teacher')
    else:
        chat_history[session['session_id']] = []
        return render_template('index.html', messages=conversation, user_role='student')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    print("Login: ",username)
    if USERS.get(username) == password:
        session['user_role'] = request.form.get('username')
        return redirect(url_for('index', user_role=request.form.get('username')))
    else:
        return redirect(url_for('index'))
    

@app.route('/submit', methods=['POST'])
def submit():
    #conversation.clear()
    session_id = session['session_id']
    if session_id not in conversation:
        conversation[session_id] = []
    conversation[session_id].clear()
    input_text = request.form.get('input-field')
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    user_role = session.get('user_role')
    user_message = {
        'type': 'user-message',
        'text': input_text,
        'sentTime': current_time
    }
    conversation[session_id].append(user_message)
    response = chatbot_response(input_text, user_role)
    sheet.insert_row([input_text,response],2)
    chatbot_message = {
        'type': 'chatbot',
        'text': response,
        'sentTime': current_time
    }
    conversation[session_id].append(chatbot_message) 
    return render_template('index.html', messages=conversation[session_id], user_role=user_role)
    

def chatbot_response(user_input, user_role):
    answer = ""
    with open('SyllabusChat_5Jul.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 250,
        chunk_overlap  = 50,
        length_function = count_tokens,
    )

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(text_splitter.create_documents([text]),embeddings)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.7,model_name='gpt-3.5-turbo'), db.as_retriever(),return_source_documents=True)
    result = qa({"question": user_input, "chat_history": chat_history[session['session_id']]})
    if(len(chat_history[session['session_id']])==3):
        chat_history[session['session_id']].pop(0)
    chat_history[session['session_id']].append((user_input, result['answer']))
    answer=result['answer'] 

    #print(chat_history)
    return answer

@app.route('/upload', methods=['POST'])
def upload():
    user = request.form.get('user_role')
    #print(user)
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"

    # Save the uploaded file to the document folder
    file.save(os.path.join(DOCUMENT_FOLDER, file.filename))
    
    return render_template('index.html', user_role=user)

@app.route('/download')
def download():
    filename = 'SyllabusChat_5Jul.txt'  
    user = request.args.get('user_role')
    file_path = os.path.join(DOCUMENT_FOLDER, filename)
    print(file_path)
    return send_file(file_path, as_attachment=True)
    #response = send_file(file_path, as_attachment=True)
    #response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    #response.headers['Pragma'] = 'no-cache'
    #response.headers['Expires'] = '0'
    #return redirect(url_for('home'))

@app.route('/logout', methods=['POST'])
def logout():
    chat_history.pop(session['session_id'])
    session.pop("session_id",None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 80, threaded = True)
