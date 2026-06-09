import json
import os
import sys
import time
import requests
from requests.auth import HTTPBasicAuth

# Windows 콘솔 인코딩 에러 방지
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

CONFIG_PATH = r"C:\Users\user\.gemini\antigravity\mcp_config.json"
if not os.path.exists(CONFIG_PATH):
    print(f"[Error] Config file not found at {CONFIG_PATH}")
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

jira_conf = config["mcpServers"]["jira-community"]["env"]
JIRA_URL = jira_conf["JIRA_URL"].rstrip("/")
JIRA_EMAIL = jira_conf["JIRA_EMAIL"]
JIRA_API_TOKEN = jira_conf["JIRA_API_TOKEN"]
auth = HTTPBasicAuth(JIRA_EMAIL, JIRA_API_TOKEN)
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

PROJECT_KEY = "SSD"
PROJECT_NAME = "최종 쉴더스 관제 시스템"
LEAD_ACCOUNT_ID = "712020:0f90ff3d-f355-4955-a787-64d1e2f55b84"


# 1. 기존 비즈니스 프로젝트 삭제
print(f"[1/3] Deleting old project {PROJECT_KEY} to prevent conflict...")
delete_url = f"{JIRA_URL}/rest/api/3/project/{PROJECT_KEY}"
res_del = requests.delete(delete_url, auth=auth, headers=headers)
if res_del.status_code in [204, 200]:
    print(f" -> [Success] Deleted old project {PROJECT_KEY}.")
    time.sleep(2) # 안정성을 위한 대기
elif res_del.status_code == 404:
    print(f" -> [Info] No existing {PROJECT_KEY} project found.")
else:
    print(f" -> [Warn] Deleting project failed: {res_del.status_code} - {res_del.text}")

# 2. 소프트웨어(Kanban) 템플릿 프로젝트 생성
print(f"[2/3] Creating fresh Jira SOFTWARE (Kanban) project {PROJECT_KEY}...")
project_url = f"{JIRA_URL}/rest/api/3/project"
project_payload = {
    "key": PROJECT_KEY,
    "name": PROJECT_NAME,
    "projectTypeKey": "software",
    "projectTemplateKey": "com.pyxis.greenhopper.jira:gh-kanban-template",
    "leadAccountId": LEAD_ACCOUNT_ID
}

res_proj = requests.post(project_url, auth=auth, headers=headers, json=project_payload)
if res_proj.status_code == 201:
    print(f" -> [Success] Fresh Jira Software project {PROJECT_KEY} created successfully!")
else:
    print(f" -> [Error] Software project creation failed: {res_proj.status_code} - {res_proj.text}")
    sys.exit(1)

# 3. 쇄신된 WBS 스크립트 수정 및 구동준비
print(f"[3/3] Optimizing sequential {PROJECT_KEY} sync script for Kanban hierarchy...")
# 기존의 sync_premium_wbs.py를 원본으로 사용
source_sync_path = r"C:\Users\user\Documents\최종 쉴더스\sync_premium_wbs.py"
with open(source_sync_path, "r", encoding="utf-8") as fs:
    code = fs.read()

# PROJECT_KEY와 Epic 이슈타입에 대한 하드코딩 변경
code = code.replace('PROJECT_KEY = "KAN"', f'PROJECT_KEY = "{PROJECT_KEY}"')
code = code.replace('"issuetype": {"name": "Epic" if JIRA_URL.endswith(".net") else "Task"}', '"issuetype": {"name": "Epic"}')

# M1~M4의 range(69,91) 청소부를 SSD 번호(1~150) 청소부로 변경
clean_old_shd_code = f"""# 2. 기존 백로그 정리 (삭제)
print("[1/4] Cleaning up previous project keys...")
for i in range(1, 150):
    issue_key = f"{{PROJECT_KEY}}-{{i}}"
    url = f"{{JIRA_URL}}/rest/api/3/issue/{{issue_key}}"
    try:
        res = requests.delete(url, auth=auth, headers=headers)
    except Exception:
        pass"""

start_idx = code.find('# 2. 기존 KAN-69 ~ KAN-90')
end_idx = code.find('# 3. 4대 에픽')
if start_idx != -1 and end_idx != -1:
    code = code[:start_idx] + clean_old_shd_code + "\n\n" + code[end_idx:]

temp_sync_path = os.path.join(r"C:\Users\user\Documents\최종 쉴더스", f"sync_premium_wbs_{PROJECT_KEY.lower()}.py")
with open(temp_sync_path, "w", encoding="utf-8") as fw:
    fw.write(code)

print(f"[OK] Re-creation and pipeline patch complete! Ready to sync perfectly from {PROJECT_KEY}-1.")
sys.exit(0)

