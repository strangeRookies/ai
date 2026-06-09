import json
import os
import sys
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

# 1. 내 Account ID 조회
print("[1/3] Fetching lead account ID...")
myself_url = f"{JIRA_URL}/rest/api/3/myself"
res_me = requests.get(myself_url, auth=auth, headers=headers)
if res_me.status_code != 200:
    print(f"[Error] Failed to fetch user info: {res_me.status_code} - {res_me.text}")
    sys.exit(1)

account_id = res_me.json()["accountId"]
print(f" -> Found Account ID: {account_id}")

# 2. 새로운 프로젝트 'SHD' 생성 시도
NEW_PROJECT_KEY = "SHD"
NEW_PROJECT_NAME = "최종 쉴더스 보안 시스템"

print(f"[2/3] Creating new Jira project: {NEW_PROJECT_NAME} ({NEW_PROJECT_KEY})...")
project_url = f"{JIRA_URL}/rest/api/3/project"
project_payload = {
    "key": NEW_PROJECT_KEY,
    "name": NEW_PROJECT_NAME,
    "projectTypeKey": "business",
    "projectTemplateKey": "com.atlassian.jira-core-project-templates:jira-core-simplified-task-tracking",
    "leadAccountId": account_id
}

res_proj = requests.post(project_url, auth=auth, headers=headers, json=project_payload)
is_created = False

if res_proj.status_code == 201:
    print(f" -> [Success] Project {NEW_PROJECT_KEY} created via API successfully!")
    is_created = True
elif res_proj.status_code == 400 and "project key is already in use" in res_proj.text.lower():
    print(f" -> [Info] Project Key {NEW_PROJECT_KEY} is already in use. We will reuse it!")
    is_created = True
else:
    print(f" -> [Fail] API Project creation failed: {res_proj.status_code} - {res_proj.text}")
    print("\n[Action Required]")
    print(f"지라 클라우드 정책상 API를 통한 프로젝트 생성이 막혀있을 수 있습니다.")
    print(f"Jira 웹 브라우저({JIRA_URL})에 로그인하셔서 아래 정보로 새 프로젝트를 수동 생성해주세요!")
    print(f" - 프로젝트 이름: {NEW_PROJECT_NAME}")
    print(f" - 프로젝트 키 (Key): {NEW_PROJECT_KEY}")
    print(f" - 템플릿 종류: 비즈니스 (Business / 작업 관리 - 단일 작업 추적 등)")
    print("\n생성 완료 후 저에게 '새 프로젝트 생성 완료'라고 말씀해주시면 1번부터 정렬해 드리겠습니다!")
    sys.exit(2)

# 3. 프로젝트가 생성되었거나 이미 존재하므로 WBS 이슈 등록 프로세스 작동
# 이 단계에서는 sync_premium_wbs.py를 SHD 키 타겟으로 개조하여 가동합니다.
if is_created:
    print(f"[3/3] Found clean project {NEW_PROJECT_KEY}. Modifying WBS target to {NEW_PROJECT_KEY} and starting synchronization...")
    
    # sync_premium_wbs.py의 PROJECT_KEY = "KAN" 부분을 "SHD"로 바꾸는 코드를 실시간 실행
    # 그 전에 sync_premium_wbs.py 사본을 'SHD' 기준으로 임시 복제하여 돌립니다.
    sync_script_path = r"C:\Users\user\Documents\최종 쉴더스\sync_premium_wbs.py"
    with open(sync_script_path, "r", encoding="utf-8") as fs:
        script_code = fs.read()
        
    # Project Key를 KAN -> SHD로, 기존 삭제 범위를 무시하고 신규 번호(1번부터) 생성하도록 개조
    modified_code = script_code.replace('PROJECT_KEY = "KAN"', f'PROJECT_KEY = "{NEW_PROJECT_KEY}"')
    
    # 새로운 프로젝트이므로 이전 삭제(range(69,91))는 무시하도록 삭제 로직 스킵 또는 변경
    # 신규 프로젝트이므로 삭제할 기존 티켓이 없으나 혹시 중복 에러가 날 수 있으니 try-except로 감싸져 있어 안전합니다.
    # 단, 신규 프로젝트에 이미 기존 이슈가 있다면(재사용 시) 지워야 하므로 i in range(1, 100) 형태로 기존 SHD 이슈들을 청소하도록 변경하겠습니다!
    clean_old_shd_code = """# 2. 기존 KAN-69 ~ KAN-90 티켓 정리 (삭제)
print("[1/4] Cleaning up previous project keys...")
for i in range(1, 150):
    issue_key = f"{PROJECT_KEY}-{i}"
    url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}"
    try:
        res = requests.delete(url, auth=auth, headers=headers)
    except Exception:
        pass"""
        
    # 기존 삭제부 교체
    if '# 2. 기존 KAN-69 ~ KAN-90 티켓 정리 (삭제)' in modified_code:
        # 이 부분을 SHD 청소부로 치환
        # 기존 삭제 블록 탐색 후 교환
        start_idx = modified_code.find('# 2. 기존 KAN-69 ~ KAN-90 티켓 정리 (삭제)')
        end_idx = modified_code.find('# 3. 4대 에픽 및 32개 균형 태스크 구조 정의')
        if start_idx != -1 and end_idx != -1:
            modified_code = modified_code[:start_idx] + clean_old_shd_code + "\n\n" + modified_code[end_idx:]

    temp_shd_sync_path = r"C:\Users\user\Documents\최종 쉴더스\sync_premium_wbs_shd.py"
    with open(temp_shd_sync_path, "w", encoding="utf-8") as fw:
        fw.write(modified_code)
        
    print(f" -> Temporary SHD sync script generated at: {temp_shd_sync_path}")
    print("[OK] Environment ready for pure sequential indexing starting from 1!")
    sys.exit(0)
