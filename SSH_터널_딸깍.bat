@echo off
title SSH Tunnel to GPU PC
echo ========================================================
echo Starting SSH Port Forwarding Tunnel to GPU PC
echo ========================================================
echo Local Ports -> GPU PC:
echo   - 8888 : HLS Stream
echo   - 8889 : WebRTC WHEP
echo   - 8189 : WebRTC ICE
echo   - 8010 ~ 8013 : AI Overlays
echo Reverse Port (Reverse Tunnel):
echo   - 8080 : GPU PC -> Windows Backend (8080)
echo ========================================================
echo Connecting... Keep this window open to maintain tunnel.
echo.
ssh -N -L 8888:127.0.0.1:8888 -L 8889:127.0.0.1:8889 -L 8189:127.0.0.1:8189 -L 8010:127.0.0.1:8010 -L 8011:127.0.0.1:8011 -L 8012:127.0.0.1:8012 -L 8013:127.0.0.1:8013 -R 8080:127.0.0.1:8080 welabs@58.127.241.84
