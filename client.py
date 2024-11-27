# Copyright 2023 MOSEC Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import websockets
import wave
import base64
import sys
import asyncio

import re
import numpy as np 
from scipy.io.wavfile import write
WS_URL = 'ws://172.27.11.12:9501/ws'  # Replace with your WebSocket URL
import pyaudio

def inner_func(audio:int,name:str):
    print(audio)
    print(name)

def func_args(**kwargs)->dict:
    inner_func(**kwargs)

def test_args():
    func_args(audio = 1,name = "Kay")


def test3():
    from models.dsso_util import CosUploader
    import torchaudio
    file_a = "../2.tingjian/test_set/temp_resample_concated.wav"
    #waveform,sr = torchaudio.load("../2.tingjian/test_set/temp_resample_concated.wav",normalize=True)
    #print(waveform.shape)
    #print(sr)
    uploader = CosUploader(0)
    url1 = uploader.upload_audio(file_a)
    print(url1)


def test2():
    from models.server_conf import ServerConfig
    from myapp.ai_meeting_chatbot import AI_Meeting_Chatbot
    data = {"project_name":"ai_meeting_assistant_chatbot",
            "task_id":"fia967c_2024_06_2517_21_39_165_d",
            "audio_url":"/home/tione/notebook/lskong2/projects/AI_Server_V3/temp/results/ai_meeting_results/fia967c_2024_06_2517_21_39_165/ori.wav",
            #"audio_url":"./temp/voice20240124.m4a",
            #"audio_url":"./temp/luoxiang.wav",
            "task_type":1,
            "task_state":0,
            "lang":"zh",
            "recognize_speakers":1,
            "speaker_num":3 
            }
    global_conf = ServerConfig("./myapp/conf.yaml")
    model = AI_Meeting_Chatbot(global_conf)
    model.dsso_init(data)
    response,_ = model.dsso_forward_http(data)
    print(response)


def test1():
    from myapp.vits_tts_en import vits_tts
    from models.server_conf import ServerConfig
    data = {"project_name":"vits_tts",
            "text":"The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves.",
            "gender":1,  #0 woman, 1 man
            }
    global_conf = ServerConfig("./myapp/conf.yaml")
    model = vits_tts(global_conf)
    model.dsso_init()
    response,_ = model.dsso_forward_http(data)
    binary_data = base64.b64decode(response["audio_data"].encode())
        
    audio_array = np.frombuffer(binary_data, dtype=np.float32)
    write("./temp/11.wav", 22050, audio_array)

def test():
    from myapp.ai_classification import AI_Classification
    from myapp.warning_light_detection import warning_light_detection
    from myapp.forgery_detection import forgery_detection
    from myapp.vits_tts_en import vits_tts

    data = {"project_name":"warning_light_detection","image_url":"../6.DeepLearningModelServer/temp/1712561012318.png"}
    global_conf = ServerConfig("./myapp/conf.yaml")
    model = warning_light_detection(global_conf)
    model.dsso_init()
    output = model.dsso_forward_http(data)
    print(data,output)

    data = {"project_name":"forgery_detection","image_url":"../6.DeepLearningModelServer/temp/1712561012318.png"}
    model = forgery_detection(global_conf)
    model.dsso_init()
    output = model.dsso_forward_http(data)
    print(data,output)

    photos_path2 = "../3.forgery_detection/data/VOC2024_2/test/images/9139.png"
    data = {"project_name":"ai_classification","image_url":photos_path2}
    model = AI_Classification(global_conf)
    model.dsso_init()
    output = model.dsso_forward_http(data)
    print(data,output)


async def vits_tts_en():
    data = {"project_name":"vits_tts_en",
            "text":"The forms of printed letters should be beautiful, and that their arrangement on the page should be reasonable and a help to the shapeliness of the letters themselves.",
            "gender":1,  #0 woman, 1 man
            }
    output_audio = "./temp/vits_tts_en.wav"
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        binary_data = base64.b64decode(json.loads(response)["audio_data"].encode())
        
        audio_array = np.frombuffer(binary_data, dtype=np.float32)
        write(output_audio, 22050, audio_array)

async def vits_tts_cn():
    sentence = "遥望星空作文独自坐在乡间的小丘上"

    data = {"project_name":"vits_tts_cn",
            "text":sentence,
            "gender":1,  #0 woman, 1 man
            }
    output_audio = "./temp/vits_tts_cn.wav"
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=99999999999999) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        binary_data = base64.b64decode(json.loads(response)["audio_data"].encode())
        
        audio_array = np.frombuffer(binary_data, dtype=np.float32)
        write(output_audio, 16000, audio_array)
    
async def translation_zh2en():
    article_hi = "With the coming of Category and Product launch, the effectiveness of creative assets and category preference are crucial to our DSSO marketing promotion."
    data = {"project_name":"translation",
            "text":article_hi,
            "task":'zh2en'
            }
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(json.loads(response))

async def translation_en2zh():
    article_ar = "Sitting alone on a hill in the countryside, watching the sun gradually dim, listening to the birdsong gradually weakening, feeling the breeze gradually getting cooler, time always slowly steals our faces, and gradually some people will eventually leave us. The white cherry blossoms are pure and noble, the red cherry blossoms are passionate and unrestrained, and the green cherry blossoms are clear and elegant. The beauty and happiness of the flowers blooming, and the romance and elegance of the flowers falling all contain the life wisdom of cherry blossoms."

    data = {"project_name":"translation",
            "text":article_ar,
            "task":'en2zh'
            }
    encoded_data = json.dumps(data) #.encode("utf-8")

    

    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        print(json.loads(response))

async def tongchuan_en2zh():
    article_ar = "Sitting alone on a hill in the countryside"

    data = {"project_name":"translation",
            "text":article_ar,
            "task":'en2zh'
            }
    tts_data = {"project_name":"vits_tts_cn",
            "text":"",
            "gender":1,  #0 woman, 1 man
            }
    output_audio = "./temp/tongchuan_en2zh.wav"
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        tts_data['text'] = json.loads(response)['result']
    encoded_data = json.dumps(tts_data)
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        print(tts_data)
        await websocket.send(encoded_data)
        response = await websocket.recv()
        binary_data = base64.b64decode(json.loads(response)["audio_data"].encode())
        audio_array = np.frombuffer(binary_data, dtype=np.float32)
        write(output_audio, 16000, audio_array)


async def tongchuan_zh2en():
    article_hi = "独自坐在乡间的小丘上,看着阳光渐渐变暗,听着鸟鸣渐渐变弱,触着清风渐渐变凉"

    data = {"project_name":"translation",
            "text":article_hi,
            "task":'zh2en'
            }
    tts_data = {"project_name":"vits_tts_en",
            "text":"",
            "gender":1,  #0 woman, 1 man
            }
    output_audio = "./temp/tongchuan_zh2en.wav"
    encoded_data = json.dumps(data) #.encode("utf-8")
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        tts_data['text'] = json.loads(response)['result']
    encoded_data = json.dumps(tts_data)
    async with websockets.connect(WS_URL, max_size=3000000) as websocket:
        await websocket.send(encoded_data)
        response = await websocket.recv()
        binary_data = base64.b64decode(json.loads(response)["audio_data"].encode())
        audio_array = np.frombuffer(binary_data, dtype=np.float32)
        write(output_audio, 22050, audio_array)

def chat_with_bot(prompt:str)->dict:
        import urllib3
        data = {"prompt":prompt, "type":'101','stream':False}
        encoded_data = json.dumps(data).encode("utf-8")
        http1 = urllib3.PoolManager()
        r = http1.request('POST',"http://localhost:8501/inference",body=encoded_data)
        response = r.data.decode("utf-8")[5:].strip()
        output = json.loads(response)
        print(output)
        return output


def test_real_asr():
    import sys
    sys.path.append("/home/tione/notebook/lskong2/projects/2.tingjian/")
    import whisper
    import torch,torchaudio
    
    asr_model = whisper.load_model(
                name="small",
                download_root="../2.tingjian/models/"
                )

    file_name = "./temp/b.wav"
    waveform,sp = torchaudio.load(file_name,normalize=True)
    resampler = torchaudio.transforms.Resample(orig_freq=sp, new_freq=16000)
    waveform = resampler(waveform)
    result = asr_model.transcribe(waveform.squeeze(0))#, fp16=torch.cuda.is_available())
    text = result['text'].strip()
    print(text)


    with wave.open(file_name, "rb") as wf:
        framerate = wf.getframerate()
        frame = wf.getnframes()
        audio_data = wf.readframes(frame)
        audio_samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
        audio_tensor = torch.from_numpy(audio_samples).float()
        waveform = resampler(audio_tensor)
        result = asr_model.transcribe(waveform, fp16=torch.cuda.is_available())
        text = result['text'].strip()
        print(text)

def test3():
    a = 5
    if a>1:
        print("1")
    elif a>2:
        print("2")
    else:
        print("else")
    punctuation = set()
    punctuation.add('.')
    punctuation.add('!')
    punctuation.add('?')
    punctuation.add('。')
    punctuation.add('！')
    punctuation.add('？')
    pattern1 = f'[{"".join(re.escape(p) for p in punctuation)}]'

    sentence = "This is a test Is it working.. Greaet's continue"

    result = re.findall(rf'.+?{pattern1}', sentence)
    remaining_text = re.split(rf'{pattern1}', sentence)[-1]
    if remaining_text:
        result.append(remaining_text)

    print(result)





async def realtime_asr_en():
    import time
    start_time = time.time()
    async with websockets.connect(WS_URL,open_timeout=3000,close_timeout=3000) as websocket:

        # wf = wave.open('/home/tione/notebook/lskong2/projects/2.tingjian/test_set/tesla_autopilot.wav', "rb")
        # temp/1cdc7498c6d2b8dde71772e73e75af43.webm
        wf = open('./temp/tesla_autopilot.wav', "rb")

        sample_rate = 16000
        input = {"project_name": "realtime_asr_whisper",
                 "language_code": 'en', #zh 
                 "audio_data": None,  #和之前一样
                 "state": "continue",
                 "sample_rate": sample_rate,  # int
                 "translation_task":"none" # en2zh/zh2en/none  ##翻译任务，英到中/中到英/不翻译随便传
                 }
        
        output = {
            "key":
            list[
                dict[
                "output":str, ###转写结果字符串数组
                "trans":str | None, ###翻译结果字符串数组，可能有None
                "refactoring": bool, ###是否规整，如果True则转写结果深色展示，否则浅色展示
                "timestamp_start": float, ##句子起始时间
                "timestamp_end" :float ##句子结束时间
                ]
            ]
        }

        buffer_size = int(sample_rate * 2 * 1)  # 0.2 seconds of audio
        print("buffer_size: ", buffer_size)
        buffer = 0
        count = 0
        while True:
            """
            count +=1
            if count==120:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Elapsed time: {elapsed_time} seconds")
                # 107.0489194393158 seconds
                # without translation 91.54747581481934 seconds
                # without refactor 108
                exit(0)
            """
            data = wf.read(buffer_size)
            buffer += buffer_size
            print(buffer)
            if len(data) == 0:
                break
            encoded_audio = base64.b64encode(data).decode()
            input['audio_data'] = encoded_audio
            input['state'] = 'continue'

            await websocket.send(json.dumps(input))
            try:
                # Await a response from the WebSocket with a timeout
                response = await asyncio.wait_for(websocket.recv(), 1)
                print("Received response:", response)
            except asyncio.TimeoutError:
                print("Timeout: No response received within the specified time.")
            except websockets.ConnectionClosed:
                print("Connection closed")

        input['state'] = 'finished'
        await websocket.send(json.dumps(input))
        print(await websocket.recv())


def test_cos():
    from models.dsso_util import CosUploader
    uploader = CosUploader(0)

    response = uploader.upload_file("../../softwares1/segment-anything-main/models/sam_onnx_quantized_example.onnx")
    print(response)


async def realtime_asr_en_chatbot():
    import time
    start_time = time.time()
    async with websockets.connect(WS_URL,open_timeout=3000,close_timeout=3000) as websocket:

        # wf = wave.open('/home/tione/notebook/lskong2/projects/2.tingjian/test_set/tesla_autopilot.wav', "rb")
        # temp/1cdc7498c6d2b8dde71772e73e75af43.webm
        wf = open('./temp/tesla_autopilot.wav', "rb")

        sample_rate = 16000
        input = {"project_name": "realtime_asr_whisper_chatbot",
                 "language_code": 'en', #zh 
                 "audio_data": None,  #和之前一样
                 "state": "continue",
                 "sample_rate": sample_rate,  # int
                 "translation_task":"none" # en2zh/zh2en/none  ##翻译任务，英到中/中到英/不翻译随便传
                 }
        
        output = {
            "key":
            list[
                dict[
                "output":str, ###转写结果字符串数组
                "trans":str | None, ###翻译结果字符串数组，可能有None
                "refactoring": bool, ###是否规整，如果True则转写结果深色展示，否则浅色展示
                "timestamp_start": float, ##句子起始时间
                "timestamp_end" :float ##句子结束时间
                ]
            ]
        }

        buffer_size = int(sample_rate * 2 * 1)  # 0.2 seconds of audio
        print("buffer_size: ", buffer_size)
        buffer = 0
        count = 0
        while True:
            """
            count +=1
            if count==120:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Elapsed time: {elapsed_time} seconds")
                # 107.0489194393158 seconds
                # without translation 91.54747581481934 seconds
                # without refactor 108
                exit(0)
            """
            data = wf.read(buffer_size)
            buffer += buffer_size
            print(buffer)
            if len(data) == 0:
                break
            encoded_audio = base64.b64encode(data).decode()
            input['audio_data'] = encoded_audio
            input['state'] = 'continue'

            await websocket.send(json.dumps(input))
            try:
                # Await a response from the WebSocket with a timeout
                response = await asyncio.wait_for(websocket.recv(), 1)
                print("Received response:", response)
            except asyncio.TimeoutError:
                pass
            except websockets.ConnectionClosed:
                print("Connection closed")

        input['state'] = 'finished'
        await websocket.send(json.dumps(input))
        print(await websocket.recv())


async def online_asr_en_microphone():
    async with websockets.connect(WS_URL) as websocket:
        sample_rate = 16000
        input = {"project_name": "realtime_asr_whisper_chatbot",
                 "language_code": 'en', #zh 
                 "audio_data": None,  #和之前一样
                 "state": "continue",
                 "sample_rate": sample_rate,  # int
                 "translation_task":"none" # en2zh/zh2en/none  ##翻译任务，英到中/中到英/不翻译随便传
                 }
        
        output = {"trans_text": "",  #转写结果
                   "response_text": "",  #大模型返回结果
                     "record": False,
                       "if_send": True,
                         "audio_length": 10.0,
                           "speech_timestamps": None,
                             "if_wait": True  #是否需要等待，此时不录音
                    
                             }

        buffer_size = int(sample_rate)  # buffer size for 200ms
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=buffer_size)

        print("Recording...")

        try:
            while True:
                data = stream.read(buffer_size)
                encoded_audio = base64.b64encode(data).decode()
                input['audio_data'] = encoded_audio
                input['state'] = 'continue'

                await websocket.send(json.dumps(input))
                try:
                    # Await a response from the WebSocket with a timeout
                    response = await asyncio.wait_for(websocket.recv(), 1)
                    print("Received response:", response)
                    while response['if_wait']:
                        response = await asyncio.wait_for(websocket.recv(), 1)
                        print("Received response:", response)
                except asyncio.TimeoutError:
                    pass
                except websockets.ConnectionClosed:
                    print("Connection closed")
        except KeyboardInterrupt:
            input['state'] = 'finished'
            await websocket.send(json.dumps(input))
            print(await websocket.recv())
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()





if __name__ =="__main__":


    if len(sys.argv)<2:
        pass
        #asyncio.run(online_asr_en_microphone())
        #ai_meeting_chatbot_offline()
        #vits_conversion()
    elif int(sys.argv[1]) == 1:
        pass
    elif int(sys.argv[1]) == 2:
        pass
    elif int(sys.argv[1]) == 3:
        pass
    elif int(sys.argv[1]) == 4:
        pass
    elif int(sys.argv[1]) == 5:
        pass
    elif int(sys.argv[1]) == 6:
        pass
    elif int(sys.argv[1]) == 7:
        ##asyncio.run(ai_meeting())
        pass
    elif int(sys.argv[1]) == 8:
        pass
    elif int(sys.argv[1]) ==9:   
        asyncio.run(vits_tts_en())  ###英文TTS
    elif int(sys.argv[1]) ==10:  
        asyncio.run(vits_tts_cn()) ###中文TTS
    elif int(sys.argv[1]) ==11: 
        asyncio.run(translation_zh2en())    ###中文-》英文  翻译
    elif int(sys.argv[1]) ==12: 
        asyncio.run(translation_en2zh())    ###英文-》中文  翻译
    elif int(sys.argv[1]) ==13:     
        asyncio.run(tongchuan_en2zh())  ###英文-》中文  同声传译
    elif int(sys.argv[1]) ==14:   
        asyncio.run(tongchuan_zh2en())  ###中文-》英文  同声传译
    elif int(sys.argv[1]) ==15:  
        pass   ###麦克风实时语音识别 【英文实例】
    elif int(sys.argv[1]) ==16:
        chat_with_bot('您好')
    elif int(sys.argv[1]) ==17:
        asyncio.run(video_generation_online())
    elif int(sys.argv[1]) ==18:
        asyncio.run(super_resolution_video())
    elif int(sys.argv[1]) ==19:
        asyncio.run(video_generation_online2())
    elif int(sys.argv[1]) ==20:
        asyncio.run(video_generation_image2video())
    elif int(sys.argv[1]) ==21:
        asyncio.run(video_generation_connect())
    elif int(sys.argv[1]) ==22:
        asyncio.run(sam2())
    elif int(sys.argv[1]) ==23:
        asyncio.run(realtime_asr_en())
    elif int(sys.argv[1])== 24:
        asyncio.run(motion_clone())
    elif int(sys.argv[1]) == 25:
        asyncio.run(jumper_cutter())
    








