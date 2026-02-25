import urllib.request
import urllib.parse
import json
import uuid

BASE_URL = "http://localhost:8000/api/v1/evaluations"

def print_result(name, res_text, status):
    print(f"--- {name} ---")
    print(f"Status: {status}")
    try:
        print(f"Response: {json.dumps(json.loads(res_text), indent=2)}")
    except:
        print(f"Response (text): {res_text}")
    print("="*40)

def make_request(name, url, method="GET", json_data=None, files=None):
    try:
        if json_data:
            data = json.dumps(json_data).encode('utf-8')
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method=method)
        elif files:
            # Simple multipart/form-data for file upload
            boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
            body = []
            for fieldname, (filename, content, mimetype) in files.items():
                body.extend([
                    f"--{boundary}".encode('utf-8'),
                    f'Content-Disposition: form-data; name="{fieldname}"; filename="{filename}"'.encode('utf-8'),
                    f'Content-Type: {mimetype}'.encode('utf-8'),
                    b'',
                    content.encode('utf-8') if isinstance(content, str) else content,
                ])
            body.append(f"--{boundary}--".encode('utf-8'))
            body.append(b'')
            data = b'\r\n'.join(body)
            headers = {'Content-Type': f'multipart/form-data; boundary={boundary}'}
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
        else:
            req = urllib.request.Request(url, method=method)

        with urllib.request.urlopen(req) as response:
            res_text = response.read().decode('utf-8')
            print_result(name, res_text, response.status)
    except urllib.error.HTTPError as e:
        print_result(name, e.read().decode('utf-8'), e.code)
    except Exception as e:
        print(f"--- {name} ---")
        print(f"Error: {e}")
        print("="*40)

print("Starting endpoint tests...")

# 1. Health
make_request("HEALTH", f"{BASE_URL}/health")

# 2. Config
make_request("CONFIG", f"{BASE_URL}/config")

# 3. Metrics
make_request("METRICS", f"{BASE_URL}/metrics")

# 4. Upload
files = {'file': ('test.md', '# Knowledge\n\nThe Earth orbits the Sun.', 'text/markdown')}
make_request("UPLOAD", f"{BASE_URL}/upload", method="POST", files=files)

# 5. Evaluate Hallucination
req_halluc = {
    "query": "What is the capital of France?",
    "agent_output": "The capital of France is Paris.",
    "reference_context": "Paris is the capital and most populous city of France."
}
make_request("HALLUCINATION", f"{BASE_URL}/hallucination", method="POST", json_data=req_halluc)

# 6. Evaluate QA
req_qa = {
    "question": "What is 2+2?",
    "agent_answer": "4",
    "reference_answer": "4"
}
make_request("QA", f"{BASE_URL}/qa", method="POST", json_data=req_qa)

# 7. Evaluate Toxicity
req_tox = {
    "text": "You are a very kind person."
}
make_request("TOXICITY", f"{BASE_URL}/toxicity", method="POST", json_data=req_tox)

# 8. Evaluate (General)
req_eval = {
    "agent_id": str(uuid.uuid4()),
    "query": "Where is the Eiffel Tower?",
    "agent_output": "The Eiffel Tower is in Paris.",
    "reference_data": {"reference": "Eiffel Tower is located in Paris, France."},
    "eval_metrics": ["hallucination", "qa"]
}
make_request("EVALUATE", f"{BASE_URL}/evaluate", method="POST", json_data=req_eval)

# 9. Batch Evaluate
req_batch = {
    "eval_type": "hallucination",
    "data": [
        {
            "query": "What color is the sky?",
            "agent_output": "Blue",
            "reference_context": "The sky is blue."
        },
        {
            "query": "How many legs does a dog have?",
            "agent_output": "Four",
            "reference_context": "Dogs have four legs."
        }
    ],
    "provide_explanations": True
}
make_request("BATCH", f"{BASE_URL}/batch", method="POST", json_data=req_batch)

# 10. Compare
params_comp = {
    "agent1_id": "a1",
    "agent2_id": "a2",
    "query": "Compare apples and oranges.",
    "output1": "Apples are red, oranges are orange.",
    "output2": "Both are fruits but they are different colors.",
    "reference": "Apples and oranges are both fruits."
}
query_string = urllib.parse.urlencode(params_comp)
make_request("COMPARE", f"{BASE_URL}/compare?{query_string}", method="POST")

# 11. Evaluate and Correct
req_eval_corr = {
    "agent_id": str(uuid.uuid4()),
    "query": "What orbits the Earth?",
    "agent_output": "The Earth orbits the Moon.",
    "reference_data": {},
    "eval_metrics": ["kg_verification"]
}
make_request("EVALUATE AND CORRECT", f"{BASE_URL}/evaluate-and-correct", method="POST", json_data=req_eval_corr)

