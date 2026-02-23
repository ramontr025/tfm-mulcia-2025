from analyze_ancestors_api import get_ancestors, USE_LOCAL, BASE_URL, EDITION

print(f"Testing Local API: {USE_LOCAL}")
print(f"URL: {BASE_URL}")

# Root concept: 138875005
# Clinical finding: 404684003
concept_id = "404684003" 
cache = {}
version = None

print(f"Fetching ancestors for {concept_id}...")
ancestors = get_ancestors(concept_id, version, cache)

print(f"Found {len(ancestors)} ancestors.")
for anc in ancestors:
    print(f" - {anc['conceptId']}: {anc['term']}")

if len(ancestors) > 0:
    print("SUCCESS")
else:
    print("FAILURE or Root Concept")
