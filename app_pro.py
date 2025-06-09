import streamlit as st
import requests
import pandas as pd
import json
import datetime
import numpy as np
from sklearn.ensemble import IsolationForest # 이상치 감지
from sklearn.preprocessing import StandardScaler #표준화
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import openai
import re
import os
from dotenv import load_dotenv
load_dotenv()


# ✅ 최신 OpenAI GPT API 키
openai.api_key = os.getenv("OPENAI_API_KEY")

# ✅ 네이버 부동산 API 헤더
headers = {
    "Accept-Encoding": "gzip",
    "Host": "new.land.naver.com",
    "Referer": "https://new.land.naver.com/complexes/102378?ms=37.5018495,127.0438028,16&a=APT&b=A1&e=RETAIL",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IlJFQUxFU1RBVEUiLCJpYXQiOjE3MDExNTE5MzcsImV4cCI6MTcwMTE2MjczN30.dbdp5VH2gNMCxpfj_WtFkYhjM1CkC-t0deZUfgQU-fw"
}

@st.cache_data #데이터를 한 번만 계산하고 저장해두는 기능
def get_regions(cortar_no="0000000000"):
    url = f"https://new.land.naver.com/api/regions/list?cortarNo={cortar_no}"
    r = requests.get(url, headers=headers)
    temp = json.loads(r.text)
    codes = [r['cortarNo'] for r in temp['regionList']]
    names = [r['cortarName'] for r in temp['regionList']]
    return dict(zip(names, codes))

@st.cache_data
def get_complexes(region_code, real_estate_type):
    url = f"https://new.land.naver.com/api/regions/complexes?cortarNo={region_code}&realEstateType={real_estate_type}"
    r = requests.get(url, headers=headers)
    try:
        temp = json.loads(r.text)
        if "complexList" in temp:
            return [(item['complexName'], item['complexNo']) for item in temp['complexList']]
    except:
        return []
    return []

def search_listings(complex_ids, real_estate_type, trade_type):
    results = []
    for complex_id in complex_ids:
        page = 1
        while True:
            url = f"https://new.land.naver.com/api/articles/complex/{complex_id}?realEstateType={real_estate_type}&tradeType={trade_type}&page={page}"
            r = requests.get(url, headers=headers)
            try:
                temp = json.loads(r.text)
                if 'articleList' in temp:
                    results.extend(temp['articleList'])
                if not temp.get('isMoreData'):
                    break
                page += 1
            except:
                break
    return pd.DataFrame(results)

# ✅ UI
st.title("🏢 부동산 매물 검색기 + AI 분석기")
st.markdown("실시간 매물 검색, 이상탐지, AI 가격예측, GPT 요약까지 한 번에!")

with st.sidebar:
    st.header("1️⃣ 검색 조건 선택")
    sido_dict = get_regions()
    sido_name = st.selectbox("시/도 선택", list(sido_dict.keys()))
    sigungu_dict = get_regions(sido_dict[sido_name])
    sigungu_name = st.selectbox("시/군/구 선택", list(sigungu_dict.keys()))
    dong_dict = get_regions(sigungu_dict[sigungu_name])
    dong_name = st.selectbox("읍/면/동 선택", list(dong_dict.keys()))

    real_estate_options = {
        "아파트": "APT", "재건축": "JGC", "오피스텔": "OPST",
        "재개발": "JGB", "분양권": "ABYG", "분양 예정": "PRE"
    }
    selected_types = st.multiselect("주택 구분", list(real_estate_options.keys()), default=["아파트"])
    real_estate_type = ":".join([real_estate_options[i] for i in selected_types]) or "APT"

    trade_type_options = {"매매": "A1", "전세": "B1", "월세": "B2"}
    selected_trade = st.multiselect("거래 유형", list(trade_type_options.keys()), default=["매매"])
    trade_type = ":".join([trade_type_options[i] for i in selected_trade]) or "A1"

    complex_list = get_complexes(dong_dict[dong_name], real_estate_type)
    selected_complex_names = st.multiselect("단지 선택 (선택 안해도 가능)", [x[0] for x in complex_list])
    selected_complex_ids = [x[1] for x in complex_list if x[0] in selected_complex_names]

if st.button("🔍 매물 검색"):
    if not selected_complex_ids:
        if complex_list:
            st.info("⚠️ 단지 미선택 → 전체 단지 매물 검색합니다.")
            selected_complex_ids = [x[1] for x in complex_list]
        else:
            st.warning("❌ 해당 동에 단지가 없습니다.")
    if selected_complex_ids:
        df = search_listings(selected_complex_ids, real_estate_type, trade_type)
        if df.empty:
            st.info("🙁 검색된 매물이 없습니다.")
        else:
            table = df[["articleName", "tradeTypeName", "dealOrWarrantPrc", "area1", "area2", "articleFeatureDesc"]]
            table = table.rename(columns={
                "articleName": "매물명", "tradeTypeName": "거래유형", "dealOrWarrantPrc": "가격(만원)",
                "area1": "면적(㎡)", "area2": "공급면적", "articleFeatureDesc": "매물특징"
            })
            st.subheader("📋 검색 결과")
            st.dataframe(table)

            #엑셀로 저장하는 코드 
            now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"매물_{sido_name}_{dong_name}_{now}.xlsx"
            df.to_excel(file_name, index=False)
            with open(file_name, "rb") as f:
                st.download_button("📥 엑셀 다운로드", data=f, file_name=file_name)

# ✅ 분석기
st.markdown("---")
st.title("📊 AI 분석기")

uploaded = st.file_uploader("📁 매물 엑셀 업로드", type=["xlsx"])
if uploaded:
    df = pd.read_excel(uploaded)
    # st.write("📌 컬럼명 원본:", df.columns.tolist())  # 어떤 컬럼명이 있는지 확인
    df.columns = df.columns.str.strip()
    #st.write("📌 공백 제거 후:", df.columns.tolist())  # 결과 확인

    #누구나 알아볼 수 있게 한글로 변경
    display_columns = {
    "articleNo": "매물번호",
    "articleName": "매물명",
    "articleStatus": "상태",
    "realEstateTypeCode": "부동산유형코드",
    "realEstateTypeName": "부동산유형명",
    "articleRealEstateTypeCode": "매물유형코드",
    "articleRealEstateTypeName": "매물유형명",
    "tradeTypeCode": "거래유형코드",
    "tradeTypeName": "거래유형명",
    "floorInfo": "층정보",
    "dealOrWarrantPrc": "가격(만원)",
    "areaName": "지역명",
    "area1": "면적(㎡)",
    "area2": "공급면적(㎡)",
    "direction": "방향",
    "articleConfirmYmd": "등록일",
    "articleFeatureDesc": "특징",
    "tagList": "태그",
    "buildingName": "건물명",
    "latitude": "위도",
    "longitude": "경도",
    "realtorName": "중개사명",
    "cpName": "중개사무소명",
    "detailAddress": "상세주소",
    "representativeImgUrl": "대표이미지URL"
    # 필요한 컬럼은 추가 가능
}
    
    # 보여줄 때만 rename
    st.dataframe(df[list(display_columns.keys())].rename(columns=display_columns))

    #한줄 내용요약
    st.subheader("📋 업로드된 데이터")
    st.dataframe(
        df[list(display_columns.keys())]
        .head()
        .rename(columns=display_columns)
    )

    # 숫자 정제
    if 'dealOrWarrantPrc' in df.columns:
        df['price_num'] = df['dealOrWarrantPrc'].astype(str).str.replace(",", "").str.extract(r'(\d+)').astype(float)
    if 'area1' in df.columns:
        df['area1'] = df['area1'].astype(str).str.extract(r'([\d\.]+)').astype(float)

    if st.checkbox("✅ 이상 매물 탐지 (Isolation Forest)"):
        if 'price_num' in df.columns and 'area1' in df.columns:
            # ➊ 평당 가격 계산
            df['price_per_m2'] = df['price_num'] / df['area1']

            # ➋ 평당가 평균 계산
            avg_price_per_m2 = df['price_per_m2'].mean()

            # ➌ 평당가 차이 비율 계산
            df['평균 대비 차이(%)'] = ((df['price_per_m2'] - avg_price_per_m2) / avg_price_per_m2 * 100).round(1)

            # ➍ 이상치 탐지
            clf = IsolationForest(contamination=0.1, random_state=42)
            df['anomaly'] = clf.fit_predict(df[['price_per_m2']])

            # ➎ 이상 유형 판별
            df['이상 유형'] = df.apply(lambda row:
                "고가 이상치" if row['anomaly'] == -1 and row['평균 대비 차이(%)'] > 0 else (
                "저가 이상치" if row['anomaly'] == -1 and row['평균 대비 차이(%)'] < 0 else "-"), axis=1)

            # ✅ 전체 결과 구성
            result = df[['articleName', 'dealOrWarrantPrc', 'area1', 'price_per_m2', '평균 대비 차이(%)', '이상 유형']].rename(columns={
                "articleName": "매물명",
                "dealOrWarrantPrc": "가격(만원)",
                "area1": "면적(㎡)",
                "price_per_m2": "평당가(만원)"
            })

            # ✅ 이상치만 필터링
            abnormal_df = result[result['이상 유형'] != "-"]

            if not abnormal_df.empty:
                st.subheader("⚠️ 이상 매물 리스트 (평당가 기준)")
                st.dataframe(abnormal_df)
            else:
                st.info("✅ 이상치로 감지된 매물이 없습니다.")

            # ✅ 전체 결과 출력
            st.subheader("📋 전체 매물 결과")
            st.dataframe(result)


    if st.checkbox("✅ AI 가격 예측 (면적→가격)"):
        if 'area1' in df.columns and 'price_num' in df.columns:
            x = df['area1'].values.reshape(-1, 1)
            y = df['price_num'].values

            # ✅ 간단한 신경망 모델
            model = Sequential()
            model.add(Dense(32, input_dim=1, activation='relu'))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1))
            model.compile(loss='mse', optimizer=Adam(0.01))
            model.fit(x, y, epochs=30, verbose=0)

            # ✅ 예측 및 컬럼 추가
            df['적정 가격(만원)'] = model.predict(df['area1'].values.reshape(-1, 1)).astype(int)

            # ✅ 보기 좋게 정리
            result = df[['articleName', 'area1', 'dealOrWarrantPrc', '적정 가격(만원)']].rename(columns={
                "articleName": "매물명",
                "area1": "면적(㎡)",
                "dealOrWarrantPrc": "실제 가격(만원)"
            })

            st.success("✅ AI가 예측한 적정 가격입니다. 실제 가격과 비교해보세요!")
            st.dataframe(result)

        else:
            st.warning("❌ 면적 또는 가격 데이터가 부족합니다.")


    # 📝 텍스트 AI 분석기
    st.markdown("---")
    st.title("📝 텍스트 AI 분석기")

    # 🔁 역변환용: 한글 → 영문
    reverse_display_columns = {v: k for k, v in display_columns.items()}

    # ✅ 텍스트 컬럼 후보 (object 타입)
    text_column_options = [col for col in df.columns if df[col].dtype == 'object']

    # ✅ 보기용 이름 생성
    text_column_labels = [display_columns.get(col, col) for col in text_column_options]

    # ✅ 사용자에게는 한글 컬럼 보여주고 내부에선 영문 컬럼 처리
    selected_label = st.selectbox(
        "분석할 텍스트 컬럼 선택",
        text_column_labels,
        index=text_column_labels.index("특징") if "특징" in text_column_labels else 0
    )
    text_column = reverse_display_columns.get(selected_label, selected_label)

    # ✅ 전처리
    df['clean_text'] = df[text_column].astype(str) \
        .str.replace(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", "", regex=True) \
        .str.lower()

    # ✅ 샘플 출력
    st.write("✅ 전처리된 텍스트 샘플:")
    st.write(df['clean_text'].head(30).tolist())

    # ✅ ➊ 빈도수 기반 단어 분석
    if st.checkbox("📌 ➊ 빈도수 기반 단어 분석 (BoW)"):
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(max_features=20)
        X = vectorizer.fit_transform(df['clean_text'])
        word_counts = X.toarray().sum(axis=0)
        words = vectorizer.get_feature_names_out()

        freq_df = pd.DataFrame({'단어': words, '빈도수': word_counts})
        freq_df = freq_df.sort_values(by='빈도수', ascending=False)

        st.subheader("📊 가장 많이 나온 단어 Top 20")
        st.dataframe(freq_df)

        # 나눔 고딕 문제로 전체 주석표시
        # fig, ax = plt.subplots()
        # ax.bar(freq_df['단어'], freq_df['빈도수'])
        # plt.xticks(rotation=45)
        # st.pyplot(fig)



    # ✅ ➋ TF-IDF 기반 문서 유사도 분석
    if st.checkbox("📌 ➋ TF-IDF 기반 문서 유사도 분석 (유사한 매물 추천)"):
        
        # 필요한 기본 패키지
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # TF-IDF 벡터 생성
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(df['clean_text'])

        # ✅ 매물명 + 특징 표시용 조합
        df['combo_name'] = df['articleName'] + " | " + df[text_column].fillna("")

        # ✅ 사용자에게 기준 매물 선택하게 하기
        selected_combo = st.selectbox("📌 기준이 될 매물 선택", df['combo_name'].tolist())
        doc_index = df[df['combo_name'] == selected_combo].index[0]

        # ✅ 텍스트 유사도 계산
        text_sim = cosine_similarity(tfidf_matrix[doc_index], tfidf_matrix).flatten()

        # ✅ 가격 차이 기반 점수 계산 (0~1 사이로 변환)
        price_target = df.loc[doc_index, 'price_num']
        price_scores = 1 - abs(df['price_num'] - price_target) / price_target
        price_scores = price_scores.clip(lower=0)

        # ✅ 텍스트 + 가격 유사도 평균 계산
        final_scores = (text_sim + price_scores) / 2

        # ✅ 자기 자신 제외 후 상위 5개 매물 선택
        sim_scores = list(enumerate(final_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != doc_index][:5]

        # ✅ 결과 출력
        st.subheader("📌 기준 매물")
        st.write(df.iloc[doc_index][['articleName', 'dealOrWarrantPrc', text_column]])

        st.subheader("🧩 유사한 매물 Top 5 (텍스트 + 가격)")
        for idx, score in sim_scores:
            price_diff_percent = round((df.loc[idx, 'price_num'] - price_target) / price_target * 100, 1)
            price_trend = "📉 저렴함" if price_diff_percent < 0 else "📈 비쌈"
            
            st.markdown(f"**🔹 유사도 점수: {score:.2f}** ({price_trend}, {abs(price_diff_percent)}%)")
            st.write(df.iloc[idx][['articleName', 'dealOrWarrantPrc', text_column]])


    
    # ✅ ➌ 머신러닝 기반 텍스트 분류
    # ✅ ➌ AI가 매물 설명을 읽고 자동으로 태그를 붙여요!
    if st.checkbox("📌 ➌ AI가 매물 설명을 보고 자동 분류해요"):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.metrics import classification_report
        import pandas as pd
        import re

        st.info("✍️ 매물 설명을 읽고, AI가 자동으로 태그를 붙이는 실습이에요!\n"
                "예: '한강뷰', '신축', '역세권', '로얄층' 등")

        # ✅ 층수에서 숫자만 뽑아서 'floor' 만들기
        df['floor'] = df['floorInfo'].apply(
            lambda x: int(re.search(r'\d+', str(x)).group()) if re.search(r'\d+', str(x)) else -1
        )

        # ✅ 설명을 보고 태그(label)를 자동으로 붙이는 함수
        def label_property(row):
            desc = str(row['articleFeatureDesc'])
            name = str(row['articleName'])
            floor = row['floor']

            if '한강' in desc:
                return '한강뷰'
            elif 15 <= floor <= 20:
                return '로얄층'
            elif '역' in desc or '역' in name:
                return '역세권'
            elif '신축' in desc or '신축급' in desc:
                return '신축'
            elif '풀옵션' in desc or '가전' in desc or '가구' in desc:
                return '풀옵션'
            elif '반려' in desc or '펫' in desc or '동물' in desc:
                return '반려동물'
            elif '즉시입주' in desc or '바로입주' in desc:
                return '즉시입주'
            elif '확장' in desc:
                return '확장형'
            elif '남향' in desc:
                return '남향'
            else:
                return '기타'

        # ✅ 태그 붙이기
        df['label'] = df.apply(label_property, axis=1)

        # ✅ 문장을 숫자로 바꾸기
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df['clean_text'])
        y = df['label']

        # ✅ 데이터 나누기
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ✅ AI 분류기 훈련
        clf = MultinomialNB()
        clf.fit(X_train, y_train)

        # ✅ 예측 실행
        y_pred = clf.predict(X_test)

        # ✅ 분류 결과 요약 (표 형태로 보기 쉽게 변경)
        st.subheader("📊 AI 예측 결과 요약표 (태그별 정확도 분석)")

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose().reset_index()
        report_df = report_df.rename(columns={
            'index': '태그',
            'precision': '정확도 (Precision)',
            'recall': '재현율 (Recall)',
            'f1-score': 'F1 점수',
            'support': '샘플 수'
        })

        report_df[['정확도 (Precision)', '재현율 (Recall)', 'F1 점수']] = report_df[
            ['정확도 (Precision)', '재현율 (Recall)', 'F1 점수']
        ].round(2)

        visible_df = report_df[~report_df['태그'].isin(['accuracy', 'macro avg', 'weighted avg'])]

        st.dataframe(visible_df)

        st.info("""
    📘 용어 설명:
    - **정확도 (Precision)**: AI가 예측한 것 중에 진짜 맞은 비율
    - **재현율 (Recall)**: 실제 정답 중에서 AI가 잘 맞춘 비율
    - **F1 점수**: 위 둘을 종합한 점수 (AI의 종합 실력 평가)
    """)

        # ✅ 예측 라벨 분포 보기
        st.subheader("📈 예측된 태그별 건수")
        label_counts_df = pd.DataFrame(pd.Series(y_pred).value_counts()).reset_index()
        label_counts_df.columns = ['예측된 태그', '건수']
        st.dataframe(label_counts_df)

        # ✅ 실제 vs 예측 결과 샘플
        st.subheader("🔍 실제 설명 vs AI가 붙인 태그 보기")
        sample_df = pd.DataFrame({
            "🏷️ 매물명": df.loc[y_test.index, 'articleName'].values,
            "📝 설명": df.loc[y_test.index, 'clean_text'].values,
            "✅ 실제 태그": y_test.values,
            "🤖 AI 예측": y_pred
        }).reset_index(drop=True)

        st.dataframe(sample_df.head(10))



    
    # ✅ ➍ Word2Vec 단어 임베딩 직관적 설명형 표로 보기
    if st.checkbox("📌 ➍ Word2Vec 단어 임베딩 (뜻과 유사 단어로 보기)"):
        from gensim.models import Word2Vec
        import pandas as pd

        st.info("🧠 Word2Vec으로 학습된 단어와 그 뜻, 비슷한 단어를 직관적으로 보여줍니다.")

        # 문장 나누기
        sentences = df['clean_text'].apply(lambda x: x.split()).tolist()

        # Word2Vec 훈련
        model = Word2Vec(sentences, vector_size=50, window=5, min_count=2, workers=4, epochs=100)

        # 상위 20개 단어만 사용
        vocab = list(model.wv.index_to_key)[:20]

        # 사용자에게 보여줄 단어 설명 사전 (직접 채우는 부분, 실전이면 GPT로 자동 생성 가능)
        word_explanations = {
            "한강": "서울 중심을 흐르는 강. 조망이 좋은 집에서 보이는 경우가 많음",
            "아파트": "주거용 건물. 한국에서 가장 일반적인 집 형태",
            "역세권": "지하철역 가까운 곳. 교통이 편리함",
            "저층": "1~5층 정도 낮은 층",
            "고층": "15층 이상 높은 층",
            "신축": "지은 지 얼마 안 된 새 아파트",
            "매매": "부동산을 사고파는 거래 방식",
            "전세": "보증금만 내고 일정 기간 집에 사는 방식",
            "테라스": "야외 공간이 딸린 구조",
            "리모델링": "집 안을 새로 고친 상태",
            # 그 외는 자동 생성
        }

        # 결과 정리
        rows = []
        for word in vocab:
            try:
                similar = model.wv.most_similar(word, topn=3)
                similar_words = ', '.join([w for w, _ in similar])
            except:
                similar_words = '-'
            rows.append({
                "단어": word,
                "뜻 (설명)": word_explanations.get(word, "❓이해가 어려운 단어 또는 설명 없음"),
                "비슷한 단어 TOP3": similar_words
            })

        # 표 출력
        result_df = pd.DataFrame(rows)
        st.dataframe(result_df)



    # ✅ 텍스트를 숫자로 바꾸고, 길이를 딱 20칸으로 맞춰요
    if st.checkbox("📌 ➎ 텍스트를 숫자로 변환하고 길이를 맞춰요 (Tokenizer + Padding)"):
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        st.info("✍️ AI는 글자를 바로 이해하지 못해요.\n"
                "그래서 단어를 숫자로 바꾸고, 길이도 똑같이 맞춰줘야 해요!")

        # 1️⃣ 단어를 숫자로 바꾸는 사전 만들기
        tokenizer = Tokenizer(num_words=1000, oov_token="<기타>")
        tokenizer.fit_on_texts(df['clean_text'])

        # 2️⃣ 문장을 숫자 목록으로 바꾸기
        sequences = tokenizer.texts_to_sequences(df['clean_text'])

        # 3️⃣ 숫자 목록의 길이를 딱 20칸으로 맞춰주기
        padded = pad_sequences(sequences, maxlen=20, padding='post', truncating='post')

        # ✅ 단어 사전 보기 (상위 10개)
        st.subheader("🔤 자주 나오는 단어들이 어떤 숫자로 바뀌었는지 볼까요?")
        word_index = tokenizer.word_index
        st.write(dict(list(word_index.items())[:10]))

        # ✅ 예시 문장 고르기
        st.subheader("📘 아래에서 문장을 하나 골라보세요!")
        sample_texts = df['clean_text'].dropna().unique().tolist()
        selected_text = st.selectbox("👇 문장을 고르면 숫자로 바꾸는 과정을 보여드려요", sample_texts)

        # ✅ 선택한 문장 숫자로 바꾸기
        seq = tokenizer.texts_to_sequences([selected_text])[0]
        padded_seq = pad_sequences([seq], maxlen=20, padding='post', truncating='post')[0]

        st.write("🔍 선택한 문장:", selected_text)
        
        # ✅ 보기 쉽게 단어-숫자 매핑 표로 보여주기
        tokens = selected_text.split()
        st.subheader("🔢 단어가 어떤 숫자로 바뀌었는지 볼까요?")
        display_rows = []

        for i in range(20):
            if i < len(seq):
                display_rows.append({
                    "위치": i + 1,
                    "단어": tokens[i] if i < len(tokens) else "(단어 없음)",
                    "숫자 ID": seq[i],
                    "설명": "단어를 숫자로 바꾼 값"
                })
            else:
                display_rows.append({
                    "위치": i + 1,
                    "단어": "(빈칸)",
                    "숫자 ID": 0,
                    "설명": "0으로 채워진 빈칸"
                })

        st.dataframe(pd.DataFrame(display_rows))

        # ✅ 전체 결과 요약
        st.success(f"""🧾 전체 정리:
    총 {padded.shape[0]}개의 문장이 있고,  
    모든 문장을 숫자 20개짜리로 통일했어요!  
    짧은 문장은 뒤에 0으로 채우고,  
    긴 문장은 뒤에서 잘랐어요.  
    → 이제 AI가 읽고 계산하기 편해졌어요 😊""")


    # ✅ ➏ 상업적 감성 분석 (기준 완화 다중선택 + 우선순위 점수)
    if st.checkbox("📌 ➏ 상업적 감성 분석 (LSTM + 기준 완화 + 우선순위)"):

        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.utils import to_categorical
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        import re

        st.info("🏠 전체 매물 설명을 AI가 읽고, 감정을 예측하고, 실거래 기준으로 우선 추천해줘요!")

        # ✅ 감정 라벨 분류
        def label_rule(text):
            text = str(text)
            if any(x in text for x in ["급매", "시세 이하", "가격 착함", "즉시 입주", "급하게 팔아요", "할인", "빠른 매도"]):
                return "저렴한 매물"
            elif any(x in text for x in ["비쌈", "시세보다 높음", "고평가", "매우 비쌈", "호가 높음"]):
                return "고평가 매물"
            elif any(x in text for x in ["역세권", "신축", "전망 좋음", "재건축 기대", "학군", "강남", "전세가율 높음"]):
                return "투자 추천"
            elif any(x in text for x in ["소음", "오래됨", "하자", "유찰", "벌레", "안좋음"]):
                return "투자 만류"
            else:
                return "매력적이지 않음"

        # ✅ 추천사유 설명
        def recommend_reason(text):
            text = str(text)
            if "급매" in text:
                return "💸 급매 키워드 포함"
            elif "역세권" in text:
                return "🚉 역세권 입지"
            elif "신축" in text:
                return "🏗️ 신축 건물"
            elif "전망 좋음" in text:
                return "🌅 전망 우수"
            elif "재건축 기대" in text:
                return "🏢 재건축 기대"
            elif "가격 착함" in text or "시세 이하" in text:
                return "💰 저렴한 가격"
            else:
                return "🔍 일반 매물"

        # ✅ 평당가격 계산
        def calculate_price_per_area(row):
            try:
                price = int(re.sub(r'\D', '', str(row.get('dealOrWarrantPrc', '0'))))
                area = float(row.get('area1', 0))
                return price / area if area > 0 else None
            except:
                return None

        df['sentiment_label'] = df['clean_text'].apply(label_rule)
        df['평당가격'] = df.apply(calculate_price_per_area, axis=1)
        평균평당가 = df['평당가격'].dropna().mean()

        # ✅ 우선순위 점수 계산
        def smart_priority_score(row):
            score = 0
            text = str(row['clean_text'])

            if row['AI_예측감정'] in ['저렴한 매물', '투자 추천']:
                score += 2
            if '아파트' in row.get('articleName', ''):
                score += 2
            try:
                price_per_area = row['평당가격']
                if price_per_area and price_per_area < 평균평당가 * 0.9:
                    score += 3
            except:
                pass
            if "급매" in text or "할인" in text:
                score += 2
            if "역세권" in text:
                score += 1
            if "신축" in text:
                score += 1
            if "전망 좋음" in text or "재건축 기대" in text:
                score += 1
            if "학군" in text or "전세가율" in text:
                score += 1
            if "비쌈" in text:
                score -= 1
            if "소음" in text or "하자" in text:
                score -= 1

            return score

        # ✅ 텍스트 전처리
        tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
        tokenizer.fit_on_texts(df['clean_text'])
        sequences = tokenizer.texts_to_sequences(df['clean_text'])
        padded = pad_sequences(sequences, maxlen=20, padding='post')

        # ✅ 라벨 인코딩
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(df['sentiment_label'])
        labels_categorical = to_categorical(labels_encoded, num_classes=5)

        # ✅ 학습/검증 분리
        X_train, X_test, y_train, y_test = train_test_split(padded, labels_categorical, test_size=0.2, random_state=42)

        # ✅ 모델 학습
        model = Sequential()
        model.add(Embedding(input_dim=1000, output_dim=32, input_length=20))
        model.add(LSTM(64))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=0)

        # ✅ 예측 결과 저장
        acc = model.evaluate(X_test, y_test, verbose=0)[1]
        st.success(f"🎯 감성 분석 모델 정확도: {acc * 100:.2f}%")

        all_pred_probs = model.predict(padded)
        all_pred_labels = label_encoder.inverse_transform(np.argmax(all_pred_probs, axis=1))
        df['AI_예측감정'] = all_pred_labels
        df['추천사유'] = df.apply(lambda row: recommend_reason(row['clean_text']) if row['AI_예측감정'] in ['투자 추천', '저렴한 매물'] else '', axis=1)
        df['우선추천점수'] = df.apply(smart_priority_score, axis=1)

        # ✅ 사용자 선택 필터
        selected_labels = st.multiselect(
            "🔧 추천 매물에 포함할 감정 선택",
            options=['투자 추천', '저렴한 매물', '고평가 매물', '투자 만류', '매력적이지 않음'],
            default=['투자 추천']
        )

        # ✅ 필터링 후 highlight_df 생성
        highlight_df = df[df['AI_예측감정'].isin(selected_labels)]
        highlight_df = highlight_df.sort_values(by='우선추천점수', ascending=False)

        # ✅ 결과 출력
        st.subheader("📋 전체 매물 감정 분석 결과 (점수 포함)")
        st.dataframe(highlight_df[['articleName', 'clean_text', 'AI_예측감정', '추천사유', '우선추천점수']].head(5))

        if not highlight_df.empty:
            st.subheader("💡 AI가 추천한 상위 매물 (우선순위 높은 순)")

            # ✅ 표시하고 싶은 컬럼 목록
            desired_cols = [
                'articleName', 'clean_text', 'dealOrWarrantPrc', 'realtorName',
                '평당가격', 'AI_예측감정', '추천사유', '우선추천점수'
            ]

            # ✅ 실제 highlight_df에 존재하는 컬럼만 필터링
            existing_cols = [col for col in desired_cols if col in highlight_df.columns]

            # ✅ 있는 컬럼만 출력
            st.dataframe(highlight_df[existing_cols].head(5))

        else:
            st.warning("📭 조건에 맞는 추천 매물이 없습니다. 감정 필터를 더 완화해보세요.")


    # ✅ GPT 투자 리포트 (AI 추천 상위 매물 기반)
    st.markdown("---")
    st.title("📈 투자 리포트 (GPT 생성)")

    if st.button("📊 AI 추천 상위 매물 기반 투자 리포트 생성"):
        try:
            # 👉 상위 추천 매물만 텍스트로 구성 (highlight_df 기준)
            top_recommend = highlight_df.head(5)
            if top_recommend.empty:
                st.warning("추천된 상위 매물이 없습니다. 먼저 감성 분석을 실행해주세요.")
            else:
                text_summary = top_recommend[['articleName', 'dealOrWarrantPrc', 'area1']].to_string(index=False)
                sentiment_summary = top_recommend[['articleName', 'AI_예측감정', '추천사유']].to_string(index=False)

                # 📌 프롬프트 (초보 투자자 관점, 상위 추천 매물 기준)
                prompt = f"""
                당신은 부동산 투자 리포트를 작성하는 AI 전문가입니다.
                아래 AI가 추천한 부동산 상위 매물 5건을 보고, 초보 투자자도 이해할 수 있도록 분석해 주세요.

                다음 내용을 꼭 포함해 주세요:
                1. 각 매물의 투자 매력도(입지, 가격 메리트, 재건축 가능성 등)
                2. 유의해야 할 요소(소음, 하자, 고평가 우려 등)
                3. 전체 매물 중 가장 추천할 만한 순위 및 이유
                4. 초보자가 이해할 수 있도록 용어를 최대한 쉽게 설명
                5. 마지막에 5줄 이내 발표용 요약 문단 포함

                🔸 추천 매물 요약 (이름 / 가격 / 면적):
                {text_summary}

                🔸 감정 예측 및 추천 사유:
                {sentiment_summary}

                이제 위 내용을 바탕으로 투자 리포트를 작성해주세요.
                """

                # ✅ GPT-3.5 Turbo 호출
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=4000
                )

                gpt_report = response.choices[0].message.content.strip()

                st.subheader("📑 GPT 투자 리포트")
                st.markdown(gpt_report)

        except Exception as e:
            st.warning("❌ GPT 리포트 생성 중 오류 발생")
            st.error(e)


else:
    st.info("⬆️ 엑셀 파일을 업로드하면 분석이 시작됩니다.")


