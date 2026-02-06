
import requests
import json

#chat_model_url = "https://chat-route.impactsummit.nxtgen.cloud"
#chat_model_name = "ibm-granite/granite-3.3-8b-instruct"

#chat_model_url = "https://model-serve-scale-route-vllm-scale.apps.cluster-9lksb.9lksb.sandbox1114.opentlc.com/"
#chat_model_name = "ibm-granite/granite-3.3-2b-instruct" 

#chat_model_url = "https://model-serve-qwen3-32b.impactsummit.nxtgen.cloud"
#chat_model_name = "Qwen/Qwen3-32B" #ibm-granite/granite-3.3-8b-instruct"

chat_model_url = "https://model-serve-chat.impactsummit.nxtgen.cloud"
chat_model_name = "ibm-granite/granite-3.3-8b-instruct" #ibm-granite/granite-3.3-8b-instruct"
url = chat_model_url+"/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Dummy" # use a real API key if server requires it
}
data = {
    "model": chat_model_name,
    "messages": [
        {"role": "user", "content": "how many indic languages are there?"}
    ]
}

try:
    response = requests.post(url, headers=headers, data=json.dumps(data), 
                             #verify="chain.pem")
                             verify=False)
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
