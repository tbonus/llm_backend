from typing import Any
from fastapi import FastAPI, Request, Response
from groq import AsyncGroq
import instructor
from pydantic import BaseModel

app = FastAPI()


class ParsedRequest(BaseModel):
    endpoint_path: str
    method: str
    headers: Any
    cookies: dict[str, str] | None
    query_params: dict[str, Any] | None
    body: bytes | None


class BackendResponse(BaseModel):
    status_code: int
    headers: dict[str, str]
    content: str


async def call_llm(request: ParsedRequest):
    client = instructor.from_groq(AsyncGroq(), mode=instructor.Mode.JSON)
    response = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a backend server that needs to respond to users queries. You will get a request and then you need to respond to it.",
            },
            {
                "role": "user",
                "content": request.model_dump_json()
            }
        ],
        model="qwen-2.5-32b",
        response_model=BackendResponse,
    )

    return response

@app.route(
    "/{request_path:path}",
    methods=["GET", "HEAD", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
)
async def only_endpoint(request: Request):
    parsed_request = ParsedRequest(
        endpoint_path=str(request.url),
        method=request.method,
        headers={k: v for k, v in request.headers.items()},
        cookies=request.cookies,
        query_params={k: v for k, v in request.query_params.items()},
        body=(
            await request.body() if request.method in ["POST", "PATCH", "PUT"] else None
        ),
    )

    print(parsed_request)

    response = await call_llm(parsed_request)
    print(response)

    return Response(content=response.content, status_code=response.status_code, headers=response.headers)
