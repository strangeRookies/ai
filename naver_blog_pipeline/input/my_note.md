# GPU PC RTSP 채널과 AI 분석 대상 동기화

## 배경

GPU PC에서는 cam1, cam2, cam3, cam4만 실제 RTSP로 송출 중인데, 백엔드 DB나 프론트 화면에는 더 많은 카메라가 남아 있을 수 있다.

이 상태에서는 프론트에서 영상이 보이지 않는 카메라에서도 AI 알림이 발생할 수 있다.

## 해결 방향

- 백엔드는 active 상태인 카메라만 내려준다.
- 프론트는 백엔드 API에서 받은 활성 카메라만 렌더링한다.
- AI 분석 서버는 실제 RTSP 연결이 가능한 카메라만 분석한다.

## 실행 예시

```bash
python scripts/run_registered_cameras.py --backend-base-url "http://127.0.0.1:18080" --rtsp-base-url "rtsp://127.0.0.1:8554"
```

## 이미지 메모

![카메라 흐름도](./images/camera-flow.png)

