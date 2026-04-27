from openai import OpenAI
from dotenv import load_dotenv
import httpx
import json
import os

load_dotenv()

API_BASE = "http://localhost:8000"
client   = OpenAI()  


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            "description": "Busca recomendações de filmes personalizadas para um usuário pelo ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "ID do usuário (1 a 943)"
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_movies_by_title",
            "description": "Busca filmes pelo título ou parte do título.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Título ou parte do título do filme"
                    }
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_movie_details",
            "description": "Retorna detalhes de um filme pelo ID, incluindo média de ratings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {
                        "type": "integer",
                        "description": "ID do filme"
                    }
                },
                "required": ["movie_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_movies",
            "description": "Retorna filmes similares a um filme pelo ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "movie_id": {
                        "type": "integer",
                        "description": "ID do filme"
                    }
                },
                "required": ["movie_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stats",
            "description": "Retorna estatísticas gerais do sistema.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

def execute_tool(name: str, arguments: str) -> str:
    inputs = json.loads(arguments)
    try:
        if name == "get_recommendations":
            r = httpx.get(f"{API_BASE}/recommendations/{inputs['user_id']}")
            return json.dumps(r.json(), ensure_ascii=False)

        elif name == "search_movies_by_title":
            r = httpx.get(f"{API_BASE}/movies/search", params={"q": inputs["title"]})
            return json.dumps(r.json(), ensure_ascii=False)

        elif name == "get_movie_details":
            r = httpx.get(f"{API_BASE}/movies/{inputs['movie_id']}")
            return json.dumps(r.json(), ensure_ascii=False)

        elif name == "get_similar_movies":
            r = httpx.get(f"{API_BASE}/movies/{inputs['movie_id']}/similar")
            return json.dumps(r.json(), ensure_ascii=False)

        elif name == "get_stats":
            r = httpx.get(f"{API_BASE}/stats")
            return json.dumps(r.json(), ensure_ascii=False)

        else:
            return f"Ferramenta '{name}' não encontrada."

    except Exception as e:
        return f"Erro: {str(e)}"

def run_agent(user_message: str, history: list) -> tuple[str, list]:
    system_prompt = """Você é um assistente especialista em recomendação de filmes.
Você tem acesso a um sistema de ML real com dados do MovieLens (943 usuários, 1682 filmes).
O sistema usa SVD (Matrix Factorization) com RMSE de 0.7253.
Responda sempre em português, de forma natural e conversacional.
Apresente listas de filmes de forma amigável com título e informações relevantes."""

    history = history + [{"role": "user", "content": user_message}]
    messages = [{"role": "system", "content": system_prompt}] + history

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # barato e eficiente
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                print(f"\n[agente usando ferramenta: {tool_call.function.name}]")
                result = execute_tool(
                    tool_call.function.name,
                    tool_call.function.arguments
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        elif response.choices[0].finish_reason == "stop":
            final_text = message.content
            history = history + [{"role": "assistant", "content": final_text}]
            return final_text, history

        else:
            break

    return "Erro no agente.", history

if __name__ == "__main__":
    print("🎬 MovieLens Agent (OpenAI) — digite 'sair' para encerrar\n")
    history = []

    while True:
        user_input = input("Você: ").strip()
        if user_input.lower() in ["sair", "exit", "quit"]:
            print("Até logo!")
            break
        if not user_input:
            continue

        print("Agente: ", end="", flush=True)
        response, history = run_agent(user_input, history)
        print(response)
        print()