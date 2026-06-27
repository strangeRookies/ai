@echo off
setlocal
chcp 65001 >nul
title View-only SSH Tunnel to GPU PC
echo ========================================================
echo View-only tunnel - does NOT restart or touch any process.
echo Use this if someone else already started AI_DEV_실행_딸깍.bat
echo or AI_STABLE_실행_딸깍.bat and you just want to watch the stream.
echo ========================================================
echo Local Ports -> GPU PC:
echo   - 8888 : HLS Stream
echo   - 8889 : WebRTC WHEP
echo   - 8189 : WebRTC ICE
echo (no reverse port - viewers don't need to expose their backend)
echo ========================================================
echo Connecting... Keep this window open to maintain tunnel.
echo.
ssh -N -L 8888:127.0.0.1:8888 -L 8889:127.0.0.1:8889 -L 8189:127.0.0.1:8189 welabs@58.127.241.84
