@echo off
rem 過夜跑對局：板子要插著、App 要開著。結果存在 logs\games.jsonl。
rem 規則 AI 100 局（幾分鐘就打完），Gemini 4 Neon 和 Claude Opus 5.5 各 6 局（額度用完會自動等重置再接著打）。
cd /d %~dp0
set ADB=%LOCALAPPDATA%\Arduino15\packages\arduino\tools\adb\32.0.0\adb.exe
"%ADB%" forward tcp:7000 tcp:7000
rem --resume：讀 logs\games.jsonl，已經打完的開局不再重打，中斷後重開也能接著打。
python -u pc\llm_bridge.py --run rule:100 --run gemini:6 --run opus:6 --resume --log logs\games.jsonl
pause
