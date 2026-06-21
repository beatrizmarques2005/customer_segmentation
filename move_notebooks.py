import os
import base64
import requests

token = os.environ.get("GITHUB_TOKEN")
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}

owner = "beatrizmarques2005"
repo = "customer_segmentation"
branch = "main"

files = [
    {
        "old_path": "notebook_testing/analysis.ipynb",
        "new_path": "notebooks/analysis.ipynb",
        "create_msg": "Move analysis.ipynb to notebooks/",
        "delete_msg": "Remove old notebook_testing/analysis.ipynb"
    },
    {
        "old_path": "notebook_testing/clustering.ipynb",
        "new_path": "notebooks/clustering.ipynb",
        "create_msg": "Move clustering.ipynb to notebooks/",
        "delete_msg": "Remove old notebook_testing/clustering.ipynb"
    },
    {
        "old_path": "notebook_testing/preprocessing.ipynb",
        "new_path": "notebooks/preprocessing.ipynb",
        "create_msg": "Move preprocessing.ipynb to notebooks/",
        "delete_msg": "Remove old notebook_testing/preprocessing.ipynb"
    }
]

for f in files:
    print(f"Processing {f['old_path']} -> {f['new_path']}")
    
    # 1. Read the current file content from the old path (using local checked out file)
    with open(f["old_path"], "rb") as file_handle:
        file_bytes = file_handle.read()
    
    # Base64 encode it
    encoded_content = base64.b64encode(file_bytes).decode("utf-8")
    
    # 2. Create the file at the new path with that exact same content
    create_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{f['new_path']}"
    create_data = {
        "message": f["create_msg"],
        "content": encoded_content,
        "branch": branch
    }
    
    print(f"Creating new file at {f['new_path']}...")
    r_create = requests.put(create_url, json=create_data, headers=headers)
    if r_create.status_code not in [200, 201]:
        print(f"Error creating file: {r_create.status_code} {r_create.text}")
        exit(1)
    
    create_res = r_create.json()
    new_sha = create_res["content"]["sha"]
    commit_sha_create = create_res["commit"]["sha"]
    print(f"Created successfully. New SHA: {new_sha}, Commit SHA: {commit_sha_create}")
    
    # 3. Get the SHA of the old file from API
    get_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{f['old_path']}?ref={branch}"
    r_get = requests.get(get_url, headers=headers)
    if r_get.status_code != 200:
        print(f"Error getting old file metadata: {r_get.status_code} {r_get.text}")
        exit(1)
    
    old_sha = r_get.json()["sha"]
    print(f"Old file SHA: {old_sha}")
    
    # 4. Delete the file at the old path
    delete_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{f['old_path']}"
    delete_data = {
        "message": f["delete_msg"],
        "sha": old_sha,
        "branch": branch
    }
    
    print(f"Deleting old file at {f['old_path']}...")
    r_delete = requests.delete(delete_url, json=delete_data, headers=headers)
    if r_delete.status_code != 200:
        print(f"Error deleting old file: {r_delete.status_code} {r_delete.text}")
        exit(1)
        
    delete_res = r_delete.json()
    commit_sha_delete = delete_res["commit"]["sha"]
    print(f"Deleted successfully. Commit SHA: {commit_sha_delete}")

print("All file moves completed successfully!")