# Portfolio
---
**목차(Context)**
> <h6>(※ 목차의 프로젝트를 클릭하시면 상세내용으로 이동합니다.)</h6>

* [1.은행 고객 이탈 예측](#1-은행-고객-이탈-예측)  
```은행 고객 이탈을 막기 위해 ML 분류모델(CatBoost)을 활용한 Classification과 이탈 방지를 위한 전략 도출을 수행하였습니다.```
* [2.얼굴 이미지로 나이, 성별, 감정 예측](#2-얼굴-이미지로-나이-성별-감정-예측)  
```프로젝트는 다양한 비즈니스 분야에서 고객의 니즈 파악을 위한 목표를 가지고 진행한 DL 프로젝트입니다.```  
* [3.이커머스 기업 비즈니스 전략 제안](#3-이커머스-기업-비즈니스-전략-제안)  
```이커머스 Domain에 지역 및 고객별 특징을 고려한 경영 전략 도출을 위한 데이터 분석을 수행하였습니다.```
* [4.이디야 & 스타벅스 매장 위치 상관관계 분석](#4-이디야-\&-스타벅스-매장-위치-상관관계-분석)  
```동일 업종 안에서의 기업이 미투 마케팅 전략을 사용하는가에 대한 데이터 분석을 수행하였습니다.```

## 1. 은행 고객 이탈 예측  
<!--[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DAjihwanPark/portfolio/blob/main/%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8A/02_notebook/S_PJT01_CODE.ipynb) -->
* 문제정의  
```
· 디지털 전환으로 경쟁이 심화되어 마케팅 비용 증가했지만, 고객의 '조용한 이탈' 증가
· 고객의 이탈로 매출 감소와 고객 충성도가 하락하며 이는 기업 경쟁력 약화로 연결
· 신규 고객을 유치하는 것보다 기존 고객을 유지하는 것이 더 어려움
```  
* 수행역할  
```
· Pandas, matplotlib, seaborn, statsmodel을 활용하여 데이터 전처리 및 시각화
· 다양한 머신러닝 기법을 활용한 분류 모델 구축
```
* 기대효과 및 Lesson and Learned  
```
<기대효과>
· 고객의 이탈을 예측하여 효과적인 마케팅 전략 도출 및 고객 이탈 방지 기대
<Lesson adn Learned>
· 불균형 데이터를 처리하는 방법에 있어서 Recall/Precision의 Trade Off 관계를 이해
· 가공 데이터의 한계로 새로운 컬럼을 추가하는 feature engineering 부분의 아쉬움이 존재
```
<br>
※ 프로젝트 상세 > [바로가기](https://github.com/lsh010228/ML_Team_Project)  
 

## 2. 얼굴 이미지로 나이, 성별, 감정 예측
<!--[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DAjihwanPark/portfolio/blob/main/%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8B/02_notebook/exmaple01.ipynb)-->
* 문제정의  
```
· 언택트 트렌드와 맞물려 온오프라인 통합 경험의 매개체인 무인 매장이 늘어남에 따라 AI 시스템의 필요성 및 수요 증가
· 비즈니스 분야에서 고객의 정보를 얻고 고객 만족도를 파악하기에 어려움이 존재
```  
* 수행역할  
```
· 다양한 얼굴 검출 모델을 사용해보며 최적의 검출 모델 선정
· 딥러닝의 CNN 모델을 활용하여 성별 예측 모델 구축
· 대용량 파일의 전처리 및 모델 테스트
```  
* 기대효과 및 Lesson and Learned  
```
<기대효과>
· 모델을 활용하여 고객 서비스업에서 서비스의 질을 개선하고 만족도를 높일 수 있음
· 지속적인 모니터링이 필요한 병원/복지 시설에서 환자의 상태 파악에 도움
<Lesson adn Learned>
· 대용량 데이터 처리 경험 - 대용량 이미지 데이터의 다양한 처리 시도를 통한 작업 효율 증대
· Tensorflow + Pytorch - 딥러닝 프레임워크 숙련도 향상
· 모델 성능 향상 필요 - 데이터 확보, 모델 구조 개선, 최적화 기법 활용 필요
```
<br>

※ 프로젝트 상세 > [바로가기](https://github.com/lsh010228/DL_FACE_REC)  

## 3. 이커머스 기업 비즈니스 전략 제안
* 문제정의  
```
· 중국의  ‘알리익스프레스, 테무, 쉬인’  이커머스 플랫폼이 초저가 상품과 막대한 광고전략을 펼쳐 국내 사용자 수가 급증
· 국내 이커머스 고객의 불가피한 이탈이 예상돼 자구책 마련을 위한 비즈니스 전략 필요
```  
* 수행역할  
```
· Pandas, matplotlib, seaborn을 활용하여 데이터 전처리 및 시각화 분석
· RFM 기법을 통한 고객 세분화 및 비즈니스 전략 도출
```  
* 기대효과 및 Lesson and Learned  
```
<기대효과>
· 지역 및 고객별 특징을 고려한 경영전략을 통해 고객 활성화 및 매출 증진 효과 기대
<Lesson adn Learned>
· 고객별 특성에 따라 고객 세분화를 통해 효과적인 마케팅 방법에 대한 이해와 경험
· 가공 데이터의 한계로 RFM 분석 이외에 코호트 분석, 재구매율에 대한 분석에 대한 해석의 어려움
```
<br>

※ 프로젝트 상세 및 Code - [바로가기](https://github.com/lsh010228/Final_project)

## 4. 이디야 & 스타벅스 매장 위치 상관관계 분석
* 문제정의  
```
· 이디야 커피는 스타벅스 커피 매장이 위치하는 곳에 매장을 위치시키는 미투 마케팅 전략을 사용하는가
```  
* 수행역할  
```
· BeautifulSoup, Selenium을 이용해 웹 크롤링을 통한 스타벅스와 이디야의 매장 위치 정보 데이터 구축
· Folium을 이용한 매장별 지도 시각화 및 Tableau를 통한 대시보드 제작 
```  
* Lesson and Learned  
```
· BeautifulSoup, Selenium을 이용한 웹 데이터 수집 및 정제 과정 경험
· EDA 과정을 통한 가설 수립 및 검증
```
<br>
<!--※ 프로젝트 상세 및 Code - [바로가기](https://github.com/DAjihwanPark/portfolio/tree/main/프로젝트A)-->


