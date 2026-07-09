# 1. OpenAI 라이브러리 설치 (안 깔려있다면 주석 풀고 실행)
#%pip install openai

import base64
from openai import OpenAI

# 2. 이미지 파일을 Base64(텍스트 암호화)로 변환하는 함수
# 비전 모델에게 이미지를 전송하려면 이렇게 문자열로 쪼개서 보내야 합니다.
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

image_path = r"C:\Users\ksb29\OneDrive\바탕 화면\화면 캡처 2026-07-07 133049.png"
base64_image = encode_image(image_path)

# 3. 내 컴퓨터(LM Studio) 서버로 연결 (API 키는 아무거나 적어도 됨)
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

print("🚀 LM Studio의 Qwen-VL 모델에게 이미지를 전송합니다. (추출 중...)")

# 4. 비전 모델에게 OCR 명령 내리기
response = client.chat.completions.create(
    model="qwen2-vl-7b-instruct", # LM Studio는 아무 모델명이나 넣어도 켜져있는 모델로 자동 작동합니다.
    messages=[
        {
            "role": "system",
            "content": "너는 완벽한 한국어 OCR(문자 인식) 도우미야. 이미지에 있는 모든 텍스트를 위에서 아래로 오타 없이 정확하게 추출해. 추출된 텍스트 외에 다른 설명이나 인사말은 절대 하지 마."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "이 이미지에 적힌 텍스트를 타이핑해 줘."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    temperature=0.0 # AI가 상상력을 발휘하지 못하도록(오타 방지) 온도를 0으로 꽁꽁 얼립니다.
)

print("\n=== 🎯 LM Studio (Qwen-VL) 초지능 OCR 추출 결과 ===")
print(response.choices[0].message.content)
