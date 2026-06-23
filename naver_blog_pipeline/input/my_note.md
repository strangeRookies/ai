lstm 프레임별로 나눈거 결과 확인 및 학습 추가진행
       현재 mjpeg으로 그려내는 방식에서 webrtc에 오버레이 형식으로 전환예정
       mqtt로 보내는 json 형식에 맞춰서 db에 쌓고 프론트는 이걸 보고 오버레이 그리는걸로.
나는 보내기만하면됨

기존 baseline에서는 Faint recall이 0으로 나타났으며, 주요 원인은 캐시 누락이 아니라 Normal 3010개, Faint 144개 수준의 클래스 불균형으로 확인되었다. 이에 Weighted CrossEntropy와 Oversampling 방식을 적용하여 Faint 예측을 유도하였다. 그 결과 Weighted CE에서는 Faint recall이 약 0.057, Oversampling에서는 약 0.1까지 상승하여, 클래스 불균형 대응이 효과가 있음을 확인하였다.

다만 아직 Faint recall과 F1-score가 낮아 실전 적용 수준은 아니며, 현재 51차원 keypoint feature만으로는 Faint와 Normal을 충분히 구분하기 어렵다는 한계가 남아 있다. 다음 단계에서는 전체 데이터 기반 keypoint cache를 확장하고, 이후 center_drop, velocity, torso_angle 등 motion feature를 추가하여 Faint 탐지 성능을 개선할 예정이다.

발표 방식은 단순 기술 나열식이 아니라,
“왜 필요한가 → 어떤 문제를 해결하려 했는가 → MVP에서 무엇을 구현했는가 → 어떤 기술로 구현했는가 → 어떻게 확장할 수 있는가” 흐름으로 구성
