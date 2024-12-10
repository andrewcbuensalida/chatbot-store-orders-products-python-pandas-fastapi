Design diagram: https://lucid.app/lucidchart/17251476-0916-4e73-a293-efc65f495a13/edit?viewport_loc=-236%2C25%2C1879%2C773%2C0_0&invitationId=inv_77fa4519-6144-44a1-b5a6-303999def8ff

## Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment (mac/linux):

```bash
source .venv/bin/activate
```

For Windows
```bash
.venv\Scripts\activate.bat
```

In the future, to deactivate venv
```bash
.venv\Scripts\deactivate.bat 
```

## Install dependencies. This is not needed if doing a pip install in the notebook
pip install -r requirements.txt

## To save dependencies into requirements.txt
pip freeze > requirements.txt

## Paired repos
https://github.com/andrewcbuensalida/chatbot-store-openai-pandas.git
https://github.com/andrewcbuensalida/chatbot-store-orders-products-python-pandas-fastapi.git
https://github.com/andrewcbuensalida/react-chatbot-store.git

## To start app
Create a .env file like example.env, then fill it in with your own API keys

After activating virtual environment and installing packages 
`cd Mock_Api`

To start the orders/products server
`uvicorn mock_api:app --reload --port 8001`

To start the OpenAI server
`uvicorn openai_api:app --reload`

Open Postman and send a POST request to 
`http://127.0.0.1:8000`

With a JSON body
```json 
{
    "message":{
        "content": [
            {
                "type": "text",
                "text": "What are the top 5 highly-rated guitar products?"
            }
        ]
    }
}
```

To stop server
`tasklist | findstr uvicorn`
`taskkill /PID <PID> /F`