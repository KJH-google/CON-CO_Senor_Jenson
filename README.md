# CON-CO_Senor_Jenson

-------------------------

# 센서 측정 및 부저용 아두이노 IDE
const int MQ7_AOUT_PIN = A0;  // MQ-7 센서 아날로그 출력 핀
const int buzzerPin = 8;      // 피에조 부저 연결 핀 (디지털 핀 8)
const float VCC = 5.0;     // Arduino 공급전압  
const float RL = 10.0;   // 로드 저항 값 - 센서 회로에 사용 (보통 10kΩ)
const float R0 = 11.37;    // MQ-7 센서 기준 저항  
const float a = 19.32;    // MQ-7 특정곡선 상수  
const float b = -0.64;    // MQ-7 특정곡선 상수  

void setup() {
  Serial.begin(9600);
  pinMode(buzzerPin, OUTPUT);
  digitalWrite(buzzerPin, LOW); // 초기 상태: 부저 OFF
}

void loop() {
  int sensorValue = analogRead(MQ7_AOUT_PIN);
  float voltage = (sensorValue / 1023.0) * VCC;
  float RS = RL * (VCC - voltage) / voltage;
  float ratio = RS / R0;
  float ppm = a * pow(ratio, b);

  // ppm 값을 시리얼로 전송
  Serial.println(ppm);

  // 상태 판단
  // 정상: ppm < 200
  // 주의: 200 ≤ ppm < 800
  // 위험: 800 ≤ ppm < 3200
  // 매우 위험: ppm ≥ 3200
  if (ppm < 1) {
    // 정상: 부저 꺼짐
    noTone(buzzerPin);
  } else if (ppm < 5) {
    // 주의: 낮은 톤(1000Hz)으로 짧게 울림
    // 1초 루프 내에서 짧게 200ms 울리고 꺼짐
    tone(buzzerPin, 1000, 200);
    delay(200);
    noTone(buzzerPin);
    // 나머지 시간 대기
  } else if (ppm < 20) {
    // 위험: 중간 톤(2000Hz)으로 울림
    // 조금 더 길게 500ms 울리고 500ms 꺼짐 (총 1초 주기)
    tone(buzzerPin, 2000, 500);
    delay(500);
    noTone(buzzerPin);
    delay(500);
  } else {
    // 매우 위험: 높은 톤(3000Hz)으로 계속 울림
    // 여기서는 tone을 계속 주기 어렵기 때문에 다음 루프에서도 같은 상태이면 연속음에 가깝게
    tone(buzzerPin, 3000); 
    // 1초 기다린 후 다시 loop 진입 -> 사실상 계속 울림
    delay(1000);
    return; // 아래 delay(1000)로 안 내려가도록
  }

  delay(1000); // 다음 측정까지 1초 대기
}


--------------

# RO 보정 아두이노 IDE
const int MQ7_PIN = A0;   // MQ-7 센서 아날로그 출력 핀
const float VCC = 5.0;    // Arduino 공급 전압
const float RL = 10;    // 로드 저항값(kΩ) - 사용자가 회로 설계 시 정한 값
const float CLEAN_AIR_RATIO = 1; // 깨끗한 공기에서 RS/R0 비율 (예: MQ-7 데이터시트 참조)
float R0 = 28.73; // 초기값, 실제 보정 후에 업데이트

void setup() {
  Serial.begin(9600);
  // 센서 예열 시간 (예: MQ-7은 10~20분 이상 권장, 여기서는 간단히 2분 예)
  // 실제로는 setup 후 일정 시간 대기하거나, 측정 시작 전 기다린 후 보정할 것.
  Serial.println("Sensor preheating...");
  delay(10000); // 1분 예열 (실제 권장시간 확인)

  Serial.println("Calibrating R0. Please ensure the sensor is in clean air.");
  // 보정 측정값 여러 번 읽기 (예: 50회) 평균을 내어 안정적인 값 사용.
  float avgRS = 0;
  int numReadings = 50;
  for (int i = 0; i < numReadings; i++) {
    float sensorValue = analogRead(MQ7_PIN);
    float voltage = (sensorValue / 1023.0) * VCC;
    float RS = RL * (VCC - voltage) / voltage; // RS 계산식
    avgRS += RS;
    delay(200); // 각 측정 사이 약간 대기
  }
  avgRS = avgRS / numReadings; // 평균 RS

  // R0 계산: R0 = RS / (RS/R0(clean air))
  R0 = avgRS / CLEAN_AIR_RATIO;

  Serial.print("Calibrated R0: ");
  Serial.println(R0, 4);
}

void loop() {
  // 보정 완료 후, ppm 계산 시 R0 사용 예제
  // ppm = a * (RS/R0)^b 형태를 사용
  // 여기서는 단순히 R0가 제대로 계산되었는지 확인하는 코드만 둠.
  
  float sensorValue = analogRead(MQ7_PIN);
  float voltage = (sensorValue / 1023.0) * VCC;
  float RS = RL * (VCC - voltage) / voltage;
  float ratio = RS / R0;  // Ratio = RS/R0

  // 예: MQ-7 데이터시트 곡선 상수 a,b 정의 후 ppm 계산 가능
  // float a = 100.0;
  // float b = -1.5;
  // float ppm = a * pow(ratio, b);

  Serial.print("RS: "); Serial.print(RS);
  Serial.print(" R0: "); Serial.print(R0);
  Serial.print(" Ratio: "); Serial.println(ratio);
  
  delay(1000);
}

------
# Python 코드
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
CSV_FILE_PATH = "/home/dli/CO_ver2/co_readings_gradio.csv"  # CSV 파일 저장 경로
os.environ['OPENAI_API_KEY'] = ''
OpenAI.api_key = os.getenv("OPENAI_API_KEY")
# ==============================================

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


최근 1분간 데이터 저장용 (1초 간격 -> 60개 데이터)
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

    return text # proc_messages


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
