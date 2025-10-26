#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face Space API 테스트 (FAQ-Finetuning용)
"""

import requests
import json
import time

def test_huggingface_api():
    """Hugging Face Space API 테스트"""
    base_url = "https://csmart-ai-faq-finetuning.hf.space/predict"

    print("="*60)
    print("Hugging Face Space API 테스트")
    print("="*60)
    print(f"API 주소: {base_url}")

    print("\n질문 테스트 시작...")
    test_questions = [
        "오답노트는 어떻게 정리할까요?"
    ]

    success_count = 0

    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 테스트 {i}/{len(test_questions)} ---")
        print(f"질문: {question}")

        try:
            payload = {
                "question": question,
                "max_tokens": 80,
                "temperature": 0.3
            }

            print("답변 생성 중... (최대 2분 대기)")
            response = requests.post(
                base_url,
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                data = response.json()
                print(f"답변: {data.get('answer', data)}")
                success_count += 1
            else:
                print(f"API 호출 실패: {response.status_code}")
                print(f"오류: {response.text}")

        except requests.exceptions.Timeout:
            print("타임아웃: 답변 생성이 너무 오래 걸립니다.")
        except requests.exceptions.RequestException as e:
            print(f"요청 실패: {e}")

        if i < len(test_questions):
            time.sleep(2)

    print("\n" + "="*60)
    print(f"테스트 완료! 성공: {success_count}/{len(test_questions)}")
    print("="*60)

    if success_count > 0:
        print("API가 정상적으로 작동합니다!")
    else:
        print("API 테스트에 실패했습니다.")

    return success_count > 0


if __name__ == "__main__":
    test_huggingface_api()
