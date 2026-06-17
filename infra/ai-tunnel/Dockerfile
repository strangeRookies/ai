FROM alpine:3.18

# OpenSSH 클라이언트 및 bash 설치
RUN apk add --no-cache openssh-client bash

WORKDIR /app

# 진입점 스크립트 복사 및 실행 권한 부여
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
