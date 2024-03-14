# -*- coding: utf-8 -*-
from gevent import monkey; monkey.patch_all()
from datetime import datetime
import uuid
from flask import Flask, request, jsonify, send_file, abort, current_app, send_from_directory,  Response
from flask_cors import CORS
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
import openai
from openai import OpenAI
import os
import qdrant_client
import json
import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer
from azure.cognitiveservices.speech.audio import AudioOutputConfig
from qdrant_client import QdrantClient, models
from threading import Timer
import boto3
from boto3.dynamodb.conditions import Key
import logging
from queue import Queue
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from typing import Any, Dict, List, Optional, Union
from langchain.schema import AgentAction, AgentFinish, LLMResult
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
import threading
import time 
from flask_socketio import SocketIO, emit
import eventlet


openai.api_key = os.environ['OPENAI_API_KEY']
qdrant_api_key = os.environ['QDRANT_KEY']
qdrant_url_key = os.environ['QDRANT_URL']

speech_key = os.environ['SPEECH_KEY']
service_region = os.environ['SERVICE_REGION_KEY']

app = Flask(__name__)
CORS(app)
# 全てのオリジンからの接続を許可する場合
socketio = SocketIO(app, async_mode='gevent', cors_allowed_origins="*")
message_queue = Queue()

person_id="ID-002"
company_id="001"

points=""

student=""

feature=""

# Qdrantクライアントの生成を関数外に移動
llm = OpenAI()

embeddings = OpenAIEmbeddings()

client = qdrant_client.QdrantClient(
    qdrant_url_key,
    api_key=qdrant_api_key,
)

global_contentss = []

 #以下は、リアルタイム返信
logging.basicConfig(level=logging.INFO)  # INFOレベル以上のログを出力

logger = logging.getLogger(__name__)

dynamodb = boto3.resource('dynamodb', region_name='ap-northeast-1')
table2 = dynamodb.Table('conversations')

azure_speech_key = os.environ['SPEECH_KEY']
azure_service_region = os.environ['SERVICE_REGION_KEY']

message_queue = []  # クライアントのリスナーを追跡


db_path = 'conversation_database.db'

buffered_text = ""
url_root = "https://selftalk.onrender.com"

# AWS DynamoDBへの接続設定
table = dynamodb.Table('maindatabase')

def text_to_speech2(text):
    output_filename = f"text2output_{datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')}_{uuid.uuid4().hex}.wav"
    output_dir = 'audio'  # 音声ファイルを保存するディレクトリ
    output_path = os.path.join(output_dir, output_filename)  # 音声ファイルの保存パス

    # ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_service_region)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # SSMLを使用してテキストを音声に変換
    ssml_string = f"""<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
                    xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="ja-JP">
                    <voice name="ja-JP-DaichiNeural">
                        <mstts:express-as style="customerservice" styledegree="3">
                            {text}
                        </mstts:express-as>
                    </voice>
                  </speak>"""
    result = synthesizer.speak_ssml_async(ssml_string).get()

    # ファイル削除のためのタイマーを設定（2分後に削除）
    Timer(90, delete_file, args=[output_path]).start()

    return result, output_filename

def dummy_callback(token):
    global buffered_text
    buffered_text += token
    print(f'callback>> {token}', end='')

    if token.endswith("。") or token.endswith("?") or token.endswith("!") or token.endswith("？") or token.endswith("！"):
        print("\nToken ends with a period. Generating audio...")
        text = buffered_text
        buffered_text = ""  # バッファをクリア
        result, output_filename = text_to_speech2(text)
        print("Message successfully added to queue.")
        audio_url = url_root + '/audio/' + output_filename
        socketio.emit('audio_url', {'url': audio_url})
        print("Message send")

def openai_qa_univ(query, talk,callback_streaming=None):
    print(f"Received query for QA: {query}")
    llm = get_chain(callback_streaming)
    messages = [
    SystemMessage(
        content="You are a helpful assistant."
    ),
    HumanMessage(
        content=f"""あなたはプロの進路アドバイザーです。「場面」における、あなたの「役割」を、「ルール１」、「ルール２」、「ルール３」、「ルール４」、「ルール５」、「ルール６」、「ルール７」に従い遂行し、「ゴール」を達成してください。

#条件

「場面」：キャリアに悩む学生が、あなたのところへ相談に来ている。

「役割」：学生と日常会話をする。

「ゴール」：会話履歴の一番最後の行の、「あなた：」のセリフを埋める。（このセリフは、映画の台本として使用する。）

「ルール１」：会話は、日常会話なので、セリフは多くても数行に抑える。

「ルール２」：この後も会話は続くので、会話は絶対に完結させないように自然なセリフを考える。

「ルール３」：あなたは、学生が少しでもキャリアについてイメージが湧くように、深掘りを重視し、多様な話を展開する。

「ルール４」：以下の＃会話履歴は、あなたと学生の過去の会話なので、最大限会話の内容を考慮し、同じ内容を聞く質問をしない。
             
「ルール４」：回答は「あなた：」より後ろのセリフのみ出力する。

「ルール５」：あなたは以下の#性格を持っている。

「ルール６」：以下の#口調のように、セリフは若い男性の喋り口調に最大限似せる。（「お前」という言葉は決して使わない）

「ルール７」：学生の名前は「りゅうせい」なので、名前を呼ぶときに使用する。

#口調
これかなり面白いよ！（フレンドリーな話し方）

#性格
・学生のことが好きで、学生の力になりたい
・学生のことを知り、興味を引き出したい
・学生の興味の深掘りがしたい
・学生のキャリアパスを具体的に示してあげたい
・必要なスキルについて具体的に教えてあげたい
・学生の会話をリードしてあげたい
・具体的な職業や職種のアドバイスをしてあげたい

#学生の情報
・キャリアについてイメージできていない
・とにかく相談したい

＃会話履歴

{talk}

学生:{query}

あなた："""
    ),
]
    response = llm.invoke(messages)
    print(f"LLM response: {response}")
    return response

@socketio.on('textuniv')
def handle_messageuniv(data):
    query = data.get('text')
    user_id = "001"
    if not query:
        return "No text provided", 400
    try:
       # DynamoDBから最新の会話履歴を取得
        talk = get_latest_conversation(user_id) if get_latest_conversation(user_id) else ""
        ai_message = openai_qa_univ(query, talk, dummy_callback)
        # AIMessage オブジェクトの content 属性からテキスト内容を抽出
        response_text = ai_message.content if ai_message and hasattr(ai_message, "content") else "応答を取得できませんでした。"
        fix= f"{talk}\n学生: {query}\nあなた: {response_text}"
        # DynamoDBへの保存
        save_conversation(user_id, query, response_text, fix)
        # クライアントにレスポンステキストを送信
        socketio.emit('response', {'response': response_text})
        # 応答テキストをクライアントに返す
        return jsonify({"content": response_text}), 200
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500

def delete_file(filename):
    """指定されたファイルを削除する関数、ファイルが存在する場合のみ"""
    if os.path.exists(filename):  # ファイルが存在するかチェック
        try:
            os.remove(filename)
            print(f"Deleted file: {filename}")
        except Exception as e:
            print(f"Error deleting file: {filename}, {e}")
    else:
        print(f"File does not exist, no need to delete: {filename}")

def get_latest_conversation(user_id):
    try:
        response = table2.query(
            KeyConditionExpression=Key('conversationId').eq(user_id),
            ScanIndexForward=False,  # 最新の項目を最初にする
            Limit=1
        )
        if response['Items']:
            # 'conversation'キーの値のみを返す
            return response['Items'][0]['conversation']
        else:
            return None
    except Exception as e:
        logger.error(f"Failed to get latest conversation for user {user_id}: {str(e)}")
        return None

def save_conversation(user_id, query, final_response, conversation):
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        response = table2.put_item(
            Item={
                'conversationId': user_id,  # ユーザーIDを直接会話IDとして使用
                'timestamp': timestamp,  # このタイムスタンプを使って最新のレコードを識別
                'query': query,
                'final_response': final_response,
                'conversation': conversation
            }
        )
        logger.info(f"Conversation for user {user_id} saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save conversation for user {user_id}: {str(e)}")


def get_chain(callback_streaming=None):
    callback_manager = CallbackManager([MyCustomCallbackHandler(callback_streaming)])
    llm = ChatOpenAI(temperature=0, model_name="gpt-4-0125-preview", callback_manager=callback_manager, streaming=True)
    return llm

class MyCustomCallbackHandler(BaseCallbackHandler):
    def __init__(self, callback):
        self.callback = callback
    """Custom CallbackHandler."""
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        """Run on new LLM token. Only available when streaming is enabled."""
        if self.callback is not None:
            self.callback(token)
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when LLM errors."""
    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Run when chain starts running."""
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when chain errors."""
    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Run when tool starts running."""
    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""
    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> Any:
        """Run when tool errors."""
    def on_text(self, text: str, **kwargs: Any) -> Any:
        """Run on arbitrary text."""
    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""

@app.route('/audio/<filename>')
def audio_file(filename):
    """音声ファイルを提供するルート"""
    return send_from_directory('audio', filename)

@app.route('/')
def index():
    return 'Server is running!'

if __name__ == '__main__':
    socketio.run(app, debug=True)