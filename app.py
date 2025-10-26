from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional
import uvicorn

# ===============================
# FastAPI 앱 초기화
# ===============================
app = FastAPI(
    title="Gemma 파인튜닝 모델 API",
    description="파인튜닝된 Gemma 모델을 통해 질문에 답변하는 API",
    version="1.0.0"
)

# ===============================
# 전역 변수 (모델 로드)
# ===============================
MODEL_PATH = "./gemma2-finetuned"
tokenizer = None
model = None
device = "cpu"  # Hugging Face Spaces는 기본적으로 CPU

# ===============================
# 요청/응답 모델 정의
# ===============================
class QuestionRequest(BaseModel):
    question: str
    max_tokens: Optional[int] = 120
    temperature: Optional[float] = 0.3
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.8
    repetition_penalty: Optional[float] = 1.8
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "오답노트는 어떻게 정리할까요?",
                "max_tokens": 120,
                "temperature": 0.3
            }
        }

class AnswerResponse(BaseModel):
    question: str
    answer: str
    full_output: str
    model_path: str
    device: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

# ===============================
# 시작 시 모델 로드
# ===============================
@app.on_event("startup")
async def load_model():
    """서버 시작 시 모델을 메모리에 로드"""
    global tokenizer, model
    
    print("🚀 모델 로드 중...")
    try:
        # 필요한 모듈들 import
        from peft import PeftModel
        import os
        
        # Hugging Face 토큰 가져오기 (Space Secret에서)
        hf_token = os.environ.get("HF_TOKEN", None)
        
        if hf_token:
            print("✅ Hugging Face 토큰 확인됨")
        else:
            print("⚠️ HF_TOKEN이 설정되지 않음 (공개 모델만 사용 가능)")
        
        # 베이스 모델과 LoRA adapter를 분리해서 로드
        
        # 캐시 디렉토리 설정 (Docker 환경변수 사용)
        cache_dir = os.environ.get("HF_HOME", "/tmp/hf_cache")
        print(f"   📁 캐시 디렉토리: {cache_dir}")
        
        # 1. 베이스 모델 로드 (google/gemma-2b) - CPU 최적화
        print("   📥 베이스 모델 로드 중...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b",
            torch_dtype=torch.float32,  # CPU에서는 float32 사용
            device_map="cpu",           # CPU로 강제 설정
            token=hf_token,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,     # 메모리 사용량 최적화
            trust_remote_code=True      # Gemma 모델 호환성
        )
        
        # 2. LoRA adapter 로드
        print("   📥 LoRA adapter 로드 중...")
        model = PeftModel.from_pretrained(
            base_model, 
            MODEL_PATH,
            device_map="cpu"  # CPU로 강제 설정
        )
        
        # 3. 토크나이저 로드 (베이스 모델과 동일)
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2b",
            token=hf_token,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        model = model.to(device)
        model.eval()  # 평가 모드로 설정
        print(f"✅ 모델 로드 완료! (Device: {device})")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        raise

# ===============================
# API 엔드포인트
# ===============================

@app.get("/", response_model=dict)
async def root():
    """루트 엔드포인트 - API 정보"""
    return {
        "message": "Gemma 파인튜닝 모델 API 서버",
        "version": "1.0.0",
        "description": "수험생 Q&A 답변 모델",
        "endpoints": {
            "health": "GET /health - 서버 상태 확인",
            "predict": "POST /predict - 질문에 답변",
            "docs": "GET /docs - API 문서"
        },
        "usage": {
            "example": {
                "method": "POST",
                "url": "/predict",
                "body": {
                    "question": "오답노트는 어떻게 정리할까요?"
                }
            }
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인"""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        device=device
    )

@app.post("/predict", response_model=AnswerResponse)
async def predict(request: QuestionRequest):
    """
    질문에 대한 답변 생성
    
    - **question**: 답변을 원하는 질문
    - **max_tokens**: 생성할 최대 토큰 수 (기본값: 120)
    - **temperature**: 생성 다양성 조절 (기본값: 0.3, 낮을수록 보수적)
    - **top_k**: 상위 k개 토큰만 고려 (기본값: 40)
    - **top_p**: nucleus sampling 확률 (기본값: 0.8)
    - **repetition_penalty**: 반복 방지 강도 (기본값: 1.8)
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")
    
    try:
        # 프롬프트 생성
        prompt = f"질문: {request.question}\n답변:"
        
        # 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(device)
        
        # 답변 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=True,
                repetition_penalty=request.repetition_penalty,
                no_repeat_ngram_size=4,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # 결과 디코딩
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 답변 부분만 추출
        answer_only = full_output
        if "답변:" in full_output:
            answer_only = full_output.split("답변:", 1)[1].strip()
            
            # 다음 질문이 나오면 그 전까지만 추출
            if "질문:" in answer_only:
                answer_only = answer_only.split("질문:", 1)[0].strip()
            
            # HTML 태그나 이상한 문자 제거
            if "<" in answer_only:
                answer_only = answer_only.split("<", 1)[0].strip()
        
        return AnswerResponse(
            question=request.question,
            answer=answer_only,
            full_output=full_output,
            model_path=MODEL_PATH,
            device=device
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 생성 중 오류 발생: {str(e)}")

# ===============================
# 서버 실행
# ===============================
if __name__ == "__main__":
    print("="*70)
    print("🚀 Gemma 파인튜닝 모델 API 서버 시작")
    print("="*70)
    print(f"📍 서버 주소: http://0.0.0.0:7860")
    print(f"📚 API 문서: http://0.0.0.0:7860/docs")
    print(f"🔧 Device: {device}")
    print("="*70)
    
    uvicorn.run(
        app, 
        host="0.0.0.0",
        port=7860,  # Hugging Face Spaces는 7860 포트 사용
        log_level="info"
    )

