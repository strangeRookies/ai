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

PROJECT_KEY = "KAN"

print("Cleaning up intermediate duplicate keys (KAN-93 to KAN-128)...")
for i in range(93, 129):
    issue_key = f"{PROJECT_KEY}-{i}"
    url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}"
    try:
        res = requests.delete(url, auth=auth, headers=headers)
        if res.status_code == 204:
            print(f" -> Deleted intermediate duplicate issue: {issue_key}")
        elif res.status_code == 404:
            pass
        else:
            print(f" -> Skip/Error deleting {issue_key}: {res.status_code}")
    except Exception as e:
        print(f" -> Failed to delete {issue_key}: {str(e)}")

print("Cleanup V2 finished successfully!")
sys.exit(0)
