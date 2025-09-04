
---

# 📦 ZipPick\_ver2 — AI 기반 부동산 데이터 분석 & 투자 인사이트 대시보드

> **Streamlit 기반 Full-Stack + AI 포트폴리오 프로젝트**
> 부동산 데이터를 수집·분석하여 **이상 매물 탐지, 가격 예측, 텍스트 분석, 투자 리포트 자동화**까지 한 번에 제공하는 스마트 대시보드입니다.
> 학습 및 포트폴리오 목적의 개인 프로젝트입니다.

---

## ✨ 주요 기능 (Features)

* **데이터 수집 & 관리**

  * 지역/단지 단위 매물 데이터 조회 및 정리
  * 검색 결과를 **엑셀(.xlsx)** 파일로 저장 및 다운로드 가능

* **AI 기반 데이터 분석**

  * ✅ **이상 매물 탐지**: Isolation Forest로 평당가 기준 이상치 식별
  * ✅ **가격 예측**: 면적 데이터를 활용한 신경망 기반 가격 회귀 모델
  * ✅ **텍스트 분석**

    * BoW 기반 키워드 빈도 분석
    * TF-IDF + 코사인 유사도 기반 **유사 매물 추천**
    * Naive Bayes 분류기로 매물 설명 자동 라벨링(예: 한강뷰, 신축, 역세권 등)
    * Word2Vec 임베딩으로 **비슷한 단어 탐색**
    * Tokenizer + Padding으로 텍스트를 숫자로 변환 및 전처리

* **투자 가치 분석**

  * LSTM 기반 감성 분석: *저렴/고평가/투자 추천/투자 만류* 자동 분류
  * 텍스트·가격·위치 정보를 결합하여 **우선 추천 점수** 산출
  * 상위 매물 기반 **투자 리포트 자동 생성 (GPT 활용)**

---

## 🎥 시연 단계 (Demo Flow)

1. **검색 조건 선택**
2. **부동산 매물 검색 + AI 분석기 실행**
3. **매물 데이터 업로드 & 출력**
4. **이상 매물 리스트 출력 (평단가 기준)**
5. **AI 가격 예측 (면적 + 가격 기반)**
6. **텍스트 AI 분석 전처리**
7. **빈도수 기반 단어 분석 (BoW)**
8. **TF-IDF 기반 문서 유사도 분석 (유사 매물 추천)**
9. **AI 자동 라벨링 (매물 설명 기반 태그 생성)**
10. **Word2Vec 단어 임베딩 (뜻·유사 단어 탐색)**
11. **텍스트 숫자 변환 + 길이 정규화 (Tokenizer + Padding)**
12. **상업적 감성 분석 (LSTM + 우선순위 점수)**
13. **최종 투자 매물 추천 + GPT 리포트 생성**

---

## 🧱 기술 스택 (Tech Stack)

* **Frontend**: [Streamlit](https://streamlit.io)
* **Backend & Data**: Python (`pandas`, `numpy`, `requests`)
* **AI/ML**:

  * `scikit-learn` (Isolation Forest, Naive Bayes, TF-IDF, CountVectorizer)
  * `TensorFlow/Keras` (MLP, LSTM 기반 모델링)
  * `gensim` (Word2Vec 임베딩)
* **LLM Integration**: OpenAI Chat Completions API
* **Infra/Tools**: Docker, `.env` 환경 변수 관리, `st.cache_data` 기반 캐싱

---

## 📂 프로젝트 구조 (Project Structure)

```
ZipPick_ver2/
├─ app_pro.py              # Streamlit 메인 앱
├─ requirements.txt        # 의존성 패키지
├─ Dockerfile              # 컨테이너 환경 설정
├─ readMe.md               # 프로젝트 소개 문서
└─ 매물_*.xlsx              # 검색/분석 결과 (엑셀 다운로드 파일)
```

---

## ⚙️ 실행 방법 (Getting Started)

```bash
# 1) 가상환경 생성 및 활성화
python -m venv venv
.\venv\Scripts\activate    # Windows
# source venv/bin/activate  # macOS/Linux

# 2) 패키지 설치
pip install -r requirements.txt

# 3) OpenAI API 키 등록 (.env 파일 생성)
OPENAI_API_KEY=sk-********************************

# 4) 실행
streamlit run app_pro.py
```

브라우저에서 `http://localhost:8501` 접속 후 사용 가능합니다.

---

## 🧭 나의 역할 (Role & Contribution)

* **기획 (PM)**

  * 프로젝트 목표 정의 및 요구사항 정리
  * 기능 기획(데이터 수집 → 분석 → 리포트) 및 **전체 플로우 설계**

* **개발 (Full-Stack & AI)**

  * **Streamlit 기반 UI/UX** 설계 및 데이터 시각화 구현
  * 데이터 처리 파이프라인 구현 (pandas, numpy)
  * **머신러닝/딥러닝 모델** 설계 (IsolationForest, MLP, LSTM, Word2Vec)
  * OpenAI API 연동 → **투자 리포트 자동 생성 기능 개발**


---

## 📄 라이선스 (License)

이 프로젝트는 **학습 및 포트폴리오 제출용**으로 제작되었습니다.
상업적 사용 시 관련 법규 및 데이터 제공처의 정책을 반드시 준수해야 합니다.
라이선스: **MIT License**


