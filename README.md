## 환기 예측 시스템
#### <개요>
* 목적: 일산화탄소 농도에 따른 적절한 환기 시점을 예측하기 위해 개발.
* 사용 기술
    * 선형 회귀: 농도 증가 추세를 기반으로 임계치 도달 시간을 예측.
    * Gradio 기반 챗봇: 사용자가 질문하면 예상 환기 시점을 간단히 안내.
* 사용 원리(시각화)
> <img src="https://github.com/user-attachments/assets/4ac5c5f7-ab74-40dc-a204-7cbe63d1e8a7" width="80%" /> 
#### <CO 모니터링 & 예측 챗봇 사용 예시>
> (예시 1)
> <img src="https://github.com/user-attachments/assets/748b8012-20eb-4c25-83a5-3323375beb29" width="80%" />

> (예시 2)
> <img src="https://github.com/user-attachments/assets/96b9d44c-d844-480e-9bc4-e95d7c51c45b" width="80%" />

* 환기 예측 시스템 코드 간단설명
> 1. OpenAI Function을 통한 사용자 질의 처리 : 사용자가 CO 농도 관련 질문을 입력하면, 적절한 답변을 제공하거나 CO 임계값에 도달할 시간을 예측한다.
>    * 'openai_chat(query)' : 필요한 경우 함수를 호출해 임계값 도달 시간을 계산한 후 답변을 생성.
>    * 'gradio_interface(query)' : OpenAI Function의 결과를 Gradio 인터페이스에서 출력하도록 연결.

> 2. Gradio 인터페이스 : 사용자 친화적인 웹 인터페이스를 제공하여 CO 상태 및 예측에 대한 질의응답을 처리
>    * 입력-텍스트 질의 (예: "300ppm에 도달하려면 얼마나 걸리나요?")
>    * 출력-예측된 시간 또는 상태 정보.
>    * 웹서버-Gradio가 제공하는 웹 UI를 통해 실행.

> 3. CO 농도 데이터 처리 및 알림
>   * 데이터 수집
>     * Arduino에서 CO 농도 데이터를 시리얼 통신을 통해 수신.
>     * 1초 간격으로 데이터를 읽어 CSV 파일에 저장하고 버퍼('ppm_buffer')에 추가.
>     * 예외 처리 : 데이터가 잘못된 형식이면 무시하고 로그에 기록.
>   * 상태 확인 및 알림
>     * 'get_co_status(ppm)' : CO 농도 상태(정상, 주의, 위험, 매우 위험)를 판단.
>     * 'send_discord_alert(ppm)' : 주의 이상의 상태에서는 Discord로 알림 전송.

> 4. CSV 파일 초기화 및 데이터 저장
>   * 'init_csv_file' : 지정한 파일 경로에 디렉토리와 CSV 파일을 생성, 헤더를 추가.
>   * 'append_csv_file' : 시간과 ppm 값을 CSV 파일에 추가.

> 5. CO 농도 예측
>   * 'add_ppm_data(ppm)' : 최근 1분간의 데이터를 관리하며 오래된 데이터를 제거.
>   * 'predict_time_to_reach_threshold(threshold)':
>     * 최근 1분간 데이터를 선형 회귀로 분석해 특정 임계값(ppm)에 도달할 시간을 예측.
>     * 농도가 증가하지 않을 경우나 데이터가 부족할 경우 적절한 메시지를 반환.

> 6. 멀티스레드 서버 실행 : Gradio 서버와 실시간 데이터 수집 루프를 동시에 실행하기 위해 Python의 스레드를 활용
>    * Gradio 인터페이스 실행: iface.launch를 별도 스레드에서 실행.
>    * CO 농도 데이터 수집: 메인 스레드에서 시리얼 데이터를 지속적으로 읽어 처리.

> 7. 예외 및 종료 처리
>    * 시리얼 연결 실패 : Arduino 연결이 실패하면 에러 메시지를 출력.
>    * 키보드 인터럽트 : 사용자가 프로그램 종료를 요청하면 안전하게 종료.
>    * 자원 해제 : 시리얼 포트를 닫아 리소스를 정리.

<pre>
<code>
import serial
import time
import requests
from datetime import datetime
import csv
import os
import numpy as np
from openai import OpenAI
import gradio as gr
import json

# ===== 사용자 환경에 맞게 수정해야 하는 부분 =====
PORT = "/dev/ttyUSB0"  # Jetson Nano에서 Arduino 포트 확인 (예: /dev/ttyACM0)
WEBHOOK_URL = "https://discord.com/api/webhooks/1313826821787226132/txS4YAXl6tm_5UWQVzSCX0rQRLGOOELs2a_9PIk3vMNALzxxX2r88bDJcZ6f0K5v_3oe"  # 실제 Discord Webhook URL
os.environ['OPENAI_API_KEY'] = ''
CSV_FILE_PATH = "/home/dli/CO_ver2/co_readings_gradio.csv"  # CSV 파일 저장 경로
# ==============================================
</code>
</pre>

<pre>
<code>
# 상태 기준
# 정상: ppm < 200
# 주의: 200 ≤ ppm < 800
# 위험: 800 ≤ ppm < 3200
# 매우 위험: ppm ≥ 3200

def init_csv_file(filename):
    dir_path = os.path.dirname(filename)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    # 파일이 없을 경우 헤더 기록
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "ppm"])
        print(f"CSV 파일 생성 및 헤더 작성 완료: {filename}")

def append_csv_file(filename, timestamp_str, ppm_value):
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp_str, ppm_value])

def get_co_status(ppm):
    if ppm < 1:
        return "정상", 0x00FF00  # Green
    elif ppm < 5:
        return "주의", 0xFFFF00  # Yellow
    elif ppm < 20:
        return "위험", 0xFFA500  # Orange
    else:
        return "매우 위험", 0xFF0000  # Red
</code>
</pre>

<pre>
<code>
def send_discord_alert(ppm):
    """CO 농도별 단계 알림 전송"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status, color = get_co_status(ppm)
    description = f"현재 CO 농도는 {ppm:.2f} ppm 입니다."
    title = f"⚠️ CO {status} 상태 ⚠️"
    
    data = {
        "content": "@here",
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "fields": [
                {"name": "상태", "value": status, "inline": True},
                {"name": "측정 시간", "value": current_time, "inline": False}
            ],
            "footer": {"text": "CO 모니터링 시스템"}
        }]
    }

    try:
        response = requests.post(WEBHOOK_URL, json=data)
        if response.status_code == 204:
            print(f"[{current_time}] 디스코드 알림 전송 성공: {ppm:.2f} ppm ({status})")
        else:
            print(f"디스코드 알림 전송 실패: {response.status_code}")
    except Exception as e:
        print(f"디스코드 알림 오류: {str(e)}")
</code>
</pre>

<pre>
<code>
# 최근 1분간 데이터 저장용 (1초 간격 -> 60개 데이터)
ppm_buffer = []

def add_ppm_data(ppm):
    ppm_buffer.append((datetime.now(), ppm))
    # 1분보다 오래된 데이터 제거
    cutoff = datetime.now() - timedelta(seconds=60)
    # timedelta import를 위해 위 코드 상단에 from datetime import datetime, timedelta 추가해야 함
    while ppm_buffer and ppm_buffer[0][0] < cutoff:
        ppm_buffer.pop(0)

def predict_time_to_reach_threshold(threshold):
    # 최근 1분간 데이터 사용
    # ppm_buffer: [(time, ppm), ...]
    if len(ppm_buffer) < 2:
        return "데이터가 충분하지 않아 예측 불가합니다."

    # 시간축을 초 단위로 변환
    base_time = ppm_buffer[0][0]
    times = np.array([(t[0]-base_time).total_seconds() for t in ppm_buffer])
    ppms = np.array([t[1] for t in ppm_buffer])

    # 선형 회귀
    # y = m*x + c
    # np.polyfit(x, y, 1) -> (m, c)
    m, c = np.polyfit(times, ppms, 1)

    # 현재 마지막 값 기준으로 앞으로도 m(기울기)로 증가한다고 가정
    # threshold = m*x + c -> x = (threshold - c)/m
    if m <= 0:
        return "CO 농도가 증가하는 추세가 아닙니다. 환기가 급하지 않을 수 있습니다."

    x_threshold = (threshold - c)/m
    current_time_sec = (ppm_buffer[-1][0]-base_time).total_seconds()

    if x_threshold <= current_time_sec:
        return "이미 해당 임계값에 도달한 상태로 보입니다."

    delta = x_threshold - current_time_sec
    hours = delta // 3600  
    minutes = (delta % 3600) // 60  
    seconds = delta % 60  
    return f"{int(hours)}시간 {int(minutes)}분 {int(seconds)}초 후에, 즉 CO 농도가 {threshold}ppm에 도달할 것으로 예상됩니다."
</code>
</pre>

<pre>
<code>
use_functions = [
    {
        "type": "function",
        "function": {
        "name": "predict_time_to_reach_threshold",
        "description": "지정한 임계값(ppm)에 도달하는데 걸리는 예상 시간을 반환",
        "parameters": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "CO 임계값(ppm)"
                }
            },
            "required": ["threshold"]
            }
        }
    },
    {
        "type": "function",
        "function": {
        "name": "add_ppm_data",
        "description": "1분간 CO 데이터를 모아주는 함",
        "parameters": {
            "type": "object",
            "properties": {
                "ppm": {
                    "type": "number",
                    "description": "CO 측정"
                }
            },
            "required": ["ppm"]
            }
        }
    }
]
</code>
</pre>

<pre>
<code>
def ask_openai(llm_model, messages, user_message, functions = ''):
    client = OpenAI()
    proc_messages = messages

    if user_message != '':
        proc_messages.append({"role": "user", "content": user_message})

    if functions == '':
        response = client.chat.completions.create(model=llm_model, messages=proc_messages, temperature = 1.0)
    else:
        response = client.chat.completions.create(model=llm_model, messages=proc_messages, tools=use_functions, tool_choice="auto") # 이전 코드와 바뀐 부분

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors

        available_functions = {
            # "add_ppm_data": add_ppm_data,		
            "predict_time_to_reach_threshold": predict_time_to_reach_threshold
        }

        messages.append(response_message)  # extend conversation with assistant's reply
        print(response_message)
        # Step 4: send the info for each function call and function response to GPT
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            print(function_args)

            if 'user_prompt' in function_args:
                function_response = function_to_call(function_args.get('user_prompt'))
            else:
                function_response = function_to_call(**function_args)

            proc_messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model=llm_model,
            messages=messages,
        )  # get a new response from GPT where it can see the function response

        assistant_message = second_response.choices[0].message.content
    else:
        assistant_message = response_message.content

    text = assistant_message.replace('\n', ' ').replace(' .', '.').strip()


    proc_messages.append({"role": "assistant", "content": assistant_message})

    return text # proc_messages, 
</code>
</pre>

<pre>
<code>
import gradio as gr

def gradio_interface(user_message):
    messages = [
    {"role": "system", "content": "당신은 CO 농도 예측 전문가입니다. 사용자 질문에 대해 필요하면 함수를 호출하여 답해주세요."},
    {"role": "user", "content": "임계값은 10입니다. 환기가 언제 필요할까요?"}
    ]
    answer = ask_openai("gpt-4o-mini", messages, user_message, functions= use_functions)
    return answer



# Gradio 인터페이스
iface = gr.Interface(fn=gradio_interface,
                     inputs="text",
                     outputs="text",
                     title="CO 모니터링 & 예측 챗봇",
                     description="CO 농도 상태 및 환기가 필요한 시간 예측")
</code>
</pre>

<pre>
<code>
from datetime import timedelta
init_csv_file(CSV_FILE_PATH)

try:
    ser = serial.Serial(PORT, 9600, timeout=1)
    print(f"Arduino 연결됨: {PORT}")
    time.sleep(1)  # 시리얼 초기화 대기 시간

    # Gradio 인터페이스 별도 스레드에서 실행
    import threading
    server_thread = threading.Thread(target=iface.launch, kwargs={"debug":True}, daemon=True)
    server_thread.start()

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            try:
                ppm = float(line)
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                append_csv_file(CSV_FILE_PATH, current_time, ppm)
                print(f"CO 농도: {ppm:.2f} ppm")

                add_ppm_data(ppm)

                status, _ = get_co_status(ppm)
                if status in ["주의", "위험", "매우 위험"]:
                    send_discord_alert(ppm)

            except ValueError:
                print(f"잘못된 데이터 수신: {line}")

        # 측정 주기 1초
        time.sleep(1)

except serial.SerialException as e:
    print(f"시리얼 연결 오류: {str(e)}")
except KeyboardInterrupt:
    print("프로그램 종료 요청 받음")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
</code>
</pre>
***
## 마무리
#### <아쉬웠던 점>
> * 실험 환경 및 임계값 설정 한계
>     * 실제 CO(ppm) 기준 데이터를 활용하지 못하고, 실험 여건상 임의의 값을 기준으로 설정하여 진행함.
> * 분석의 단순성
>     * 환기 예측 시스템에 선형 회귀만 적용하여, 다양한 알고리즘을 비교하거나 활용하지 못함.
> * 데이터 활용 범위의 제한
>     * 과거 1분 간의 데이터만을 화룡한 분석으로, 더 방대한 데이터를 반영하지 못함.

#### <개선 방안>
> * 더 많은 데이터를 수집하고 다양한 알고리즘을 적용해 신뢰성 높은 실용적 시스템으로 발전
>    * 1)CO 농도 데이터 : 다양한 환경에서 CO농도 데이터를 장기간 수집하여 모델의 신뢰성을 높임.
>    * 2)다양한 알고리즘 : 다양한 알고리즘을 비교 분석해 최적의 방법론 선정.
>    * 3)실용적 시스템 : 다양한 환경에서도 활용할 수 있는 실용적인 시스템으로 발전.
