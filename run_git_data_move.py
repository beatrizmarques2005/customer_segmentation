import os
import requests
import json

token = os.environ.get("GITHUB_TOKEN")
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
}

owner = "beatrizmarques2005"
repo = "customer_segmentation"

# 1. Get latest commit on main branch
r_ref = requests.get(f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/main", headers=headers)
if r_ref.status_code != 200:
    print(f"Error getting ref: {r_ref.status_code} {r_ref.text}")
    exit(1)
ref_data = r_ref.json()
current_commit_sha = ref_data["object"]["sha"]
print(f"Current commit SHA of refs/heads/main: {current_commit_sha}")

# Get tree SHA of that commit
r_commit = requests.get(f"https://api.github.com/repos/{owner}/{repo}/git/commits/{current_commit_sha}", headers=headers)
if r_commit.status_code != 200:
    print(f"Error getting commit: {r_commit.status_code} {r_commit.text}")
    exit(1)
commit_data = r_commit.json()
base_tree_sha = commit_data["tree"]["sha"]
print(f"Base tree SHA: {base_tree_sha}")

# 2. Create new tree
tree_payload = {
    "base_tree": base_tree_sha,
    "tree": [
        {
            "path": "notebooks/analysis.ipynb",
            "mode": "100644",
            "type": "blob",
            "sha": "ae8303690d800de4e116d76d20575e876e9a7ec3"
        },
        {
            "path": "notebooks/clustering.ipynb",
            "mode": "100644",
            "type": "blob",
            "sha": "06d5d310e50a232bc855a089d308555aec229e6d"
        },
        {
            "path": "notebooks/preprocessing.ipynb",
            "mode": "100644",
            "type": "blob",
            "sha": "29d8fc5ecd98e3f8f7e8502a533b9f356ddaed08"
        },
        {
            "path": "notebook_testing/analysis.ipynb",
            "mode": "100644",
            "type": "blob",
            "sha": None
        },
        {
            "path": "notebook_testing/clustering.ipynb",
            "mode": "100644",
            "type": "blob",
            "sha": None
        },
        {
            "path": "notebook_testing/preprocessing.ipynb",
            "mode": "100644",
            "type": "blob",
            "sha": None
        }
    ]
}

r_tree = requests.post(f"https://api.github.com/repos/{owner}/{repo}/git/trees", json=tree_payload, headers=headers)
if r_tree.status_code != 201:
    print(f"Error creating tree: {r_tree.status_code} {r_tree.text}")
    exit(1)
new_tree_sha = r_tree.json()["sha"]
print(f"New tree SHA: {new_tree_sha}")

# 3. Create commit
commit_payload = {
    "message": "Rename notebook_testing -> notebooks",
    "tree": new_tree_sha,
    "parents": [current_commit_sha]
}
r_new_commit = requests.post(f"https://api.github.com/repos/{owner}/{repo}/git/commits", json=commit_payload, headers=headers)
if r_new_commit.status_code != 201:
    print(f"Error creating commit: {r_new_commit.status_code} {r_new_commit.text}")
    exit(1)
new_commit_sha = r_new_commit.json()["sha"]
print(f"New commit SHA: {new_commit_sha}")

# 4. Update ref refs/heads/main
ref_payload = {
    "sha": new_commit_sha,
    "force": False
}
r_update_ref = requests.patch(f"https://api.github.com/repos/{owner}/{repo}/git/refs/heads/main", json=ref_payload, headers=headers)
if r_update_ref.status_code != 200:
    print(f"Error updating ref: {r_update_ref.status_code} {r_update_ref.text}")
    exit(1)
print(f"Successfully updated refs/heads/main to {new_commit_sha}")

# Verification: Fetch recursive tree of the new commit
r_verify = requests.get(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{new_commit_sha}?recursive=1", headers=headers)
if r_verify.status_code != 200:
    print(f"Error during verification: {r_verify.status_code} {r_verify.text}")
    exit(1)
verify_tree = r_verify.json().get("tree", [])
paths = [item["path"] for item in verify_tree]

notebooks_exist = all(p in paths for p in [
    "notebooks/analysis.ipynb",
    "notebooks/clustering.ipynb",
    "notebooks/preprocessing.ipynb"
])
testing_removed = not any(p.startswith("notebook_testing/") for p in paths)

print("VERIFICATION RESULTS:")
print(f"Notebooks moved successfully: {notebooks_exist}")
print(f"notebook_testing/ folder removed: {testing_removed}")
print(f"Commit URL: https://github.com/{owner}/{repo}/commit/{new_commit_sha}")