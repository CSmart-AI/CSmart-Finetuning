#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
로컬 모델 직접 테스트 (서버 없이)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

def load_local_model():
    """로컬 파인튜닝 모델 직접 로드"""
    print("🚀 로컬 파인튜닝 모델 로드 중...")
    
    try:
        # 로컬 파인튜닝 모델 경로 확인
        model_path = "./gemma2-finetuned"
        if not os.path.exists(model_path):
            print(f"❌ 파인튜닝 모델을 찾을 수 없습니다: {model_path}")
            return None, None
        
        print(f"   📁 모델 경로: {model_path}")
        
        # 1. 베이스 모델 로드 (로컬에서)
        print("   📥 베이스 모델 로드 중...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 2. LoRA adapter 로드
        print("   📥 LoRA adapter 로드 중...")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        # 3. 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2b",
            trust_remote_code=True
        )
        
        model.eval()
        print(f"✅ 파인튜닝 모델 로드 완료! (Device: {next(model.parameters()).device})")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return None, None

def ask_question(model, tokenizer, question, max_tokens=120, temperature=0.3):
    """질문에 답변 생성"""
    try:
        # 프롬프트 생성
        prompt = f"질문: {question}\n답변:"
        
        # 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        
        # 답변 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=40,
                top_p=0.8,
                do_sample=True,
                repetition_penalty=1.8,
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
        
        return answer_only, full_output
        
    except Exception as e:
        return f"❌ 답변 생성 실패: {e}", ""

def test_local_model():
    """로컬 모델 직접 테스트"""
    print("="*60)
    print("🧪 로컬 모델 직접 테스트")
    print("="*60)
    
    # 1. 모델 로드
    model, tokenizer = load_local_model()
    if model is None or tokenizer is None:
        print("❌ 모델 로드 실패로 테스트를 중단합니다.")
        return False
    
    # 2. 질문 테스트
    print("\n2️⃣ 질문 테스트...")
    test_questions = [
        "오답노트는 어떻게 정리할까요?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 테스트 {i}/{len(test_questions)} ---")
        print(f"🤔 질문: {question}")
        
        print("⏳ 답변 생성 중...")
        answer, full_output = ask_question(model, tokenizer, question)
        
        if answer.startswith("❌"):
            print(f"❌ {answer}")
        else:
            print(f"✅ 답변: {answer}")
            print(f"📊 Device: {next(model.parameters()).device}")
    
    print("\n" + "="*60)
    print("🎉 로컬 모델 테스트 완료!")
    print("="*60)
    return True

if __name__ == "__main__":
    test_local_model()
