# CSmart-FAQ API 사용 가이드

## API 개요
이 API는 교육 관련 FAQ 질문에 대한 답변을 생성하는 서비스입니다. Gemma 2B 모델을 파인튜닝하여 구축되었습니다.

## API 엔드포인트

### 기본 정보
- **URL**: `https://csmart-ai-faq-finetuning.hf.space/predict`
- **메서드**: `POST`
- **Content-Type**: `application/json`

### 요청 형식
```json
{
    "question": "질문 내용",
    "max_tokens": 80,
    "temperature": 0.3,
    "top_k": 3,
    "top_p": 0.95,
    "repetition_penalty": 1.2
}
```

### 파라미터 설명
- `question` (필수): 질문 내용
- `max_tokens` (선택, 기본값: 80): 생성할 답변의 최대 토큰 수
- `temperature` (선택, 기본값: 0.3): 답변의 다양성 (0.1~1.0)
- `top_k` (선택, 기본값: 3): Top-K 샘플링
- `top_p` (선택, 기본값: 0.95): Top-P (nucleus) 샘플링
- `repetition_penalty` (선택, 기본값: 1.2): 반복 페널티

### 응답 형식
```json
{
    "answer": "생성된 답변 내용"
}
```

## 사용 예제

### Python
```python
import requests
import json

def ask_question(question):
    url = "https://csmart-ai-faq-finetuning.hf.space/predict"
    payload = {
        "question": question,
        "max_tokens": 100,
        "temperature": 0.5
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["answer"]
    except requests.exceptions.RequestException as e:
        print(f"API 호출 오류: {e}")
        return None

# 사용 예제
question = "수학 공부는 어떻게 해야 할까요?"
answer = ask_question(question)
print(f"질문: {question}")
print(f"답변: {answer}")
```

### JavaScript (Node.js)
```javascript
const axios = require('axios');

async function askQuestion(question) {
    const url = "https://csmart-ai-faq-finetuning.hf.space/predict";
    const payload = {
        question: question,
        max_tokens: 100,
        temperature: 0.5
    };
    
    try {
        const response = await axios.post(url, payload, { timeout: 120000 });
        return response.data.answer;
    } catch (error) {
        console.error('API 호출 오류:', error.message);
        return null;
    }
}

// 사용 예제
const question = "오답노트는 어떻게 정리할까요?";
askQuestion(question).then(answer => {
    console.log(`질문: ${question}`);
    console.log(`답변: ${answer}`);
});
```

### cURL
```bash
curl -X POST "https://csmart-ai-faq-finetuning.hf.space/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "영어 단어 암기는 어떻게 해야 할까요?",
       "max_tokens": 80,
       "temperature": 0.3
     }'
```

## 에러 처리

### HTTP 상태 코드
- `200`: 성공
- `400`: 잘못된 요청 (필수 파라미터 누락 등)
- `500`: 서버 내부 오류

### 에러 응답 예시
```json
{
    "detail": "Field required: question"
}
```

### 권장 에러 처리
```python
try:
    response = requests.post(url, json=payload, timeout=120)
    
    if response.status_code == 200:
        result = response.json()
        answer = result.get("answer", "답변을 생성할 수 없습니다.")
    elif response.status_code == 400:
        print("잘못된 요청입니다. 파라미터를 확인해주세요.")
    elif response.status_code == 500:
        print("서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
    else:
        print(f"예상치 못한 오류: {response.status_code}")
        
except requests.exceptions.Timeout:
    print("요청 시간이 초과되었습니다.")
except requests.exceptions.RequestException as e:
    print(f"네트워크 오류: {e}")
```

