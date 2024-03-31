import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


# 받아온 데이터 프레임의 시간 형식을 맞춰 주고, 최신 순으로 정렬
def to_datetime(df):
    # 날짜 변환
    for i, time in enumerate(df['Time']):
        if ord('a') <= ord(time[-1]) <= ord('z'):
            # 현재 연도 가져오기
            current_year = datetime.now().year

            # 날짜 형식 문자열을 datetime으로 변환
            date_format = "%H:%M %d %b"
            date_str_with_year = f"{current_year} {time}"
            datetime_obj = datetime.strptime(date_str_with_year, f"%Y {date_format}")

            # datetime 객체를 pandas Timestamp로 변환
            timestamp = pd.Timestamp(datetime_obj)
            df.at[i, 'Time'] = timestamp

        else:
            # 현재 연도, 월, 일 가져오기
            current_year = datetime.now().year
            current_month = datetime.now().strftime('%b')
            current_day = datetime.now().day

            # 날짜 형식 문자열을 datetime으로 변환
            date_format = "%Y %H:%M %d %b"
            date_str_with_year_month_day = f"{current_year} {time} {current_day} {current_month}"
            datetime_obj = datetime.strptime(date_str_with_year_month_day, date_format)

            # datetime 객체를 pandas Timestamp로 변환
            timestamp = pd.Timestamp(datetime_obj)
            df.at[i, 'Time'] = timestamp
    return df


# 데이터 읽어오기(함수로)
### @st.cache_data  -->  데이터 저장해두기.  불필요한 실행 반복하지 않도록 하기
@st.cache_data
def read_data_merged():
    raw = pd.DataFrame()

    for filename in os.listdir('./data/'):
        if filename.endswith('.xlsx'):
            # 확장자를 제거한 파일명 가져오기
            domain_name = os.path.splitext(filename)[0]

            temp = pd.read_excel(f'./data/{filename}')
            temp['domain'] = domain_name
            raw = pd.concat([raw, temp], ignore_index=True)

    raw = to_datetime(raw)
    raw = raw.sort_values(by='Time', ascending=False)
    raw = raw.reset_index(drop=True)
    return raw


#################################################

st.title('BBC News Analysis')
columns = st.columns(2)

with columns[0]:
    # 상위 15개 인기(최신 기사가 많은) 토픽 시각화
    st.subheader('Hot Topics')

    # 데이터 읽어오기
    df = read_data_merged()

    # related_dict에는 'Related' 컬럼의 값들을 쪼개어 topic들을 key로 갖고, 해당 topic의 기사 수와 토픽이 속하는 도메인을 value로 저장
    related_dict = {}
    for _, row in df.iterrows():
        related_topics = row['Related'].split(', ')
        domain_value = row['domain']

        for idx, topic in enumerate(related_topics):
            if topic not in related_dict:
                related_dict[topic] = [1, []]
            else:
                related_dict[topic][0] += 1

            if domain_value not in related_dict[topic][1]:
                related_dict[topic][1].append(domain_value)

    # 도메인 선택 상자
    domain_options = ['All'] + list(df['domain'].unique())
    selected_domain = st.selectbox("Select Domain:", domain_options, key=0)

    # 필터링된 related_dict 생성
    if selected_domain == 'All':
        filtered_related_dict = related_dict
    else:
        filtered_related_dict = {topic: values for topic, values in related_dict.items() if selected_domain in values[1]}

    # related_df 생성
    related_df = pd.DataFrame(list(filtered_related_dict.items()), columns=['Related_Topic', 'Frequency'])
    related_df['Frequency'] = related_df['Related_Topic'].apply(lambda x: filtered_related_dict[x][0])
    related_df['Domains'] = related_df['Related_Topic'].apply(lambda x: filtered_related_dict[x][1])

    # related_df를 빈도수 기준으로 내림차순으로 정렬한 후 상위 15개 행만 선택
    top_related_df = related_df.sort_values(by='Frequency', ascending=False).head(15)

    # 그래프 그리기
    fig = plt.figure(figsize=(15, 10))
    sns.barplot(x='Related_Topic', y='Frequency', data=top_related_df, palette='viridis')
    plt.title(f'Popular 10 Topics for Selected Domain: {selected_domain}')
    plt.xlabel('Related Topic', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.xticks(rotation=30, fontsize=15)
    plt.show()
    st.pyplot(fig)


#################################################

with columns[1]:
    # 각 도메인 별 기사 수의 시간에 따른 추이 시각화
    st.subheader('News Trends by Domain')

    # 'Time' 컬럼을 datetime 형식으로 변환
    df['Time'] = pd.to_datetime(df['Time'])

    # 'Time' 컬럼에서 월 추출
    df['Month'] = df['Time'].dt.to_period("M").dt.month

    # 모든 도메인에 대한 뉴스 기사 수 계산
    all_domain_counts = df.groupby(['Month', 'domain']).size().unstack().fillna(0)

    # 그래프 플로팅
    st.line_chart(all_domain_counts)


#################################################

# 전체 기사의 제목, 날짜, 내용, 토픽, 도메인을 볼 수 있음. 도메인, 토픽에 따라 필터링도 가능
st.title('Filtering Your NEWS')

# 'domain' 및 'Related' 컬럼 값으로 데이터 필터링
selected_domain = st.selectbox("Filter by Domain:", ['All'] + list(df['domain'].unique()), key=2)

# 'Related' 옵션을 related_dict의 키 값으로 대체하고 'All' 옵션 추가
related_options = ['All'] + sorted(list(related_dict.keys()))
selected_related = st.selectbox('Filter by Related:', related_options)

# 검색어 입력
search_query = st.text_input("Search by Title:", "")

if selected_domain != 'All' and selected_related != 'All':
    filtered_df = df[(df['domain'] == selected_domain) & (df['Related'].str.lower().str.contains(selected_related.lower()))]
elif selected_domain != 'All':
    filtered_df = df[(df['domain'] == selected_domain)]
elif selected_related != 'All':
    filtered_df = df[df['Related'].str.lower().str.contains(selected_related.lower())]
else:
    # No filtering selected, show the entire dataframe
    filtered_df = df

# 검색어로 'Title' 필터링
if search_query:
    filtered_df = filtered_df[filtered_df['Title'].str.lower().str.contains(search_query.lower())]


# 기사 테이블 표시
selected_column_list = st.multiselect("Show Columns:", df.columns,
                                      default=['Title', 'Time', 'Content', 'Related', 'domain'])
st.dataframe(filtered_df[selected_column_list])



st.markdown(' ')
st.markdown(' ')
st.markdown(' ')
st.markdown(' ')
st.markdown(' ')
st.markdown(' ')

##########################################

# 설명
st.markdown('#### 타겟 투자사/회사 : NH 투자증권 ')

# 펼침 옵션을 이용한 서비스 설명
with st.expander("서비스 소개"):
    st.write("""
    NH투자증권을 비롯한 금융 기관에서 투자 전략 수립 및 뉴스 트렌드 분석에 활용될 수 있는 플랫폼. 
    다양한 도메인의 뉴스 기사를 수집하고, 토픽별로 분석하여 사용자에게 가치 있는 정보를 제공 가능.

    **주요 Selling Point:**
    - **정보의 집약성 및 편의성:** BBC 뉴스 기사를 도메인 및 토픽별로 집약하여 제공, 특정 정보 쉽게 찾기 가능.
    - **도메인 별 핫한 토픽 탐색:** 각 도메인에서의 핫한 토픽 분포 확인으로 중요 이슈 파악.
    - **시계열 데이터로써의 가치:** 시간에 따른 뉴스 기사 수의 추이 분석으로 동향 예측 가능.
    - **토픽 및 도메인별 필터링:** 상세 정보 열람을 위한 토픽 및 도메인별 필터링 기능 제공.
    - **투자 전략 수립 및 의사 결정에 활용:** 분석 결과를 기반으로 투자 전략 수립에 활용 가능.

    해당 서비스는 사용자에게 효과적인 정보 수집과 분석을 제공함으로써 투자에 도움을 주고, 효율적인 의사 결정을 가능케 할 수 있음.
    """)


# 데이터 수집 페이지
with st.expander("데이터 수집 페이지"):
    st.write('https://www.bbc.com/news/technology')
    st.write('https://www.bbc.com/news/business')
    st.write('https://www.bbc.com/news/world/europe')
    st.write('https://www.bbc.com/news/world/us_and_canada')
