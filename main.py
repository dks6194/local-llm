import json
import uuid
import os
import calendar
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Generator, Any, Dict

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ollama

# Load environment variables
load_dotenv()
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY", "")

app = FastAPI(title="Ollama Chat")

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SAVED_CHATS_DIR = Path("./saved_chats")
SAVED_CHATS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "general": "qwen3:1.7b",
    "code": "qwen2.5-coder:3b"
}

# ============================================================================
# TOOLS DEFINITION
# ============================================================================

def get_current_time(timezone: str = "local") -> str:
    """Get the current time."""
    now = datetime.now()
    return now.strftime("%H:%M:%S")

def get_current_date(format: str = "full") -> str:
    """Get the current date."""
    now = datetime.now()
    if format == "short":
        return now.strftime("%Y-%m-%d")
    elif format == "long":
        return now.strftime("%B %d, %Y")
    else:  # full
        return now.strftime("%A, %B %d, %Y")

def get_day_of_week() -> str:
    """Get the current day of the week."""
    return datetime.now().strftime("%A")

def get_calendar_month(year: int = None, month: int = None) -> str:
    """Get a text calendar for a specific month."""
    now = datetime.now()
    year = year or now.year
    month = month or now.month
    return calendar.month(year, month)

def get_timestamp() -> str:
    """Get the current Unix timestamp."""
    return str(int(datetime.now().timestamp()))

def get_latest_news(category: str = None, country: str = "in", max_results: int = 5, language: str = "en") -> str:
    """Fetch latest news headlines from newsdata.io API."""
    if not NEWSDATA_API_KEY:
        return "Error: NEWSDATA_API_KEY not configured in .env file"
    
    try:
        url = "https://newsdata.io/api/1/latest"
        params = {
            "apikey": NEWSDATA_API_KEY,
            "country": country,
            "language": language
        }
        
        if category:
            params["category"] = category
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get("status") != "success":
            error_msg = data.get("results", {})
            if isinstance(error_msg, dict):
                error_msg = error_msg.get("message", "Unknown error")
            return f"Error from news API: {error_msg}"
        
        articles = data.get("results", [])
        
        # Filter out duplicates
        articles = [a for a in articles if not a.get("duplicate", False)][:max_results]
        
        if not articles:
            return "No news articles found."
        
        # Format news for readability
        news_output = []
        for i, article in enumerate(articles, 1):
            title = article.get("title", "No title")
            source = article.get("source_name", article.get("source_id", "Unknown"))
            pub_date = article.get("pubDate", "")
            link = article.get("link", "")
            categories = article.get("category", [])
            category_str = ", ".join(categories) if categories else ""
            
            # Clean up description
            description = article.get("description", "")
            if description:
                # Truncate if too long
                if len(description) > 200:
                    description = description[:200] + "..."
            
            news_item = f"{i}. {title}\n   📰 Source: {source}"
            if pub_date:
                news_item += f" | 📅 {pub_date}"
            if category_str:
                news_item += f"\n   🏷️ Categories: {category_str}"
            if description:
                news_item += f"\n   {description}"
            if link:
                news_item += f"\n   🔗 {link}"
            
            news_output.append(news_item)
        
        total = data.get("totalResults", len(articles))
        header = f"📰 Found {total:,} total articles. Showing top {len(articles)}:\n\n"
        
        return header + "\n\n".join(news_output)
    except Exception as e:
        return f"Error: {str(e)}"

# Tool registry - maps tool names to functions
TOOL_FUNCTIONS: Dict[str, callable] = {
    "get_current_time": get_current_time,
    "get_current_date": get_current_date,
    "get_day_of_week": get_day_of_week,
    "get_calendar_month": get_calendar_month,
    "get_timestamp": get_timestamp,
    "get_latest_news": get_latest_news,
}

# Tool definitions for Ollama
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current time. Use this when the user asks what time it is.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (currently only 'local' is supported)",
                        "default": "local"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get the current date. Use this when the user asks what date or day it is today.",
            "parameters": {
                "type": "object",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["short", "long", "full"],
                        "description": "Date format: 'short' (2024-01-15), 'long' (January 15, 2024), 'full' (Monday, January 15, 2024)",
                        "default": "full"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_day_of_week",
            "description": "Get what day of the week it is (Monday, Tuesday, etc.)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_calendar_month",
            "description": "Get a text calendar for a specific month. Useful when user asks to see a calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "The year (defaults to current year)"
                    },
                    "month": {
                        "type": "integer",
                        "description": "The month (1-12, defaults to current month)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_timestamp",
            "description": "Get the current Unix timestamp (seconds since epoch).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_news",
            "description": "Fetch the latest news headlines. Use this when user asks for news, current events, or what's happening in the world.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["business", "entertainment", "environment", "food", "health", "politics", "science", "sports", "technology", "top", "world"],
                        "description": "News category to filter by (optional)"
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code (e.g., 'in' for India, 'us' for USA). Defaults to 'in'",
                        "default": "in"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of news articles to return (1-10)",
                        "default": 5
                    },
                    "language": {
                        "type": "string",
                        "description": "Language code for news (e.g., 'en' for English, 'hi' for Hindi). Defaults to 'en'",
                        "default": "en"
                    }
                },
                "required": []
            }
        }
    }
]

def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool and return its result."""
    if tool_name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool '{tool_name}'"
    
    try:
        func = TOOL_FUNCTIONS[tool_name]
        result = func(**arguments)
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"

# Pydantic Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model_key: str  # 'general' or 'code'
    messages: List[Message]

class SaveChatRequest(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    model: str
    messages: List[Message]

# Utilities
def get_model_name(key: str) -> str:
    return MODELS.get(key, MODELS["general"])

# Endpoints

@app.get("/api/models")
async def list_models():
    """Return available models configuration."""
    return {
        "models": [
            {"key": "general", "name": MODELS["general"], "label": "💬 General"},
            {"key": "code", "name": MODELS["code"], "label": "💻 Code"}
        ]
    }

@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    """Stream chat response from Ollama with tool support."""
    model_name = get_model_name(request.model_key)
    
    # Convert Pydantic messages to dicts
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    
    # Keep last 20 messages context
    if len(messages) > 20:
        messages = messages[-20:]

    async def generate():
        try:
            # First, make a non-streaming call to check for tool usage
            initial_response = ollama.chat(
                model=model_name,
                messages=messages,
                tools=TOOLS,
                stream=False
            )
            
            # Check if the model wants to use tools
            if initial_response.get("message", {}).get("tool_calls"):
                tool_calls = initial_response["message"]["tool_calls"]
                
                # Build conversation with tool results
                updated_messages = messages.copy()
                updated_messages.append(initial_response["message"])
                
                # Process each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"].get("arguments", {})
                    
                    # Execute the tool
                    tool_result = execute_tool(tool_name, tool_args)
                    
                    # Add tool result to conversation
                    updated_messages.append({
                        "role": "tool",
                        "content": tool_result
                    })
                
                # Get final response with tool results (streaming)
                final_stream = ollama.chat(
                    model=model_name,
                    messages=updated_messages,
                    stream=True
                )
                
                for chunk in final_stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
            else:
                # No tool calls - make a streaming call for real-time output
                stream = ollama.chat(
                    model=model_name,
                    messages=messages,
                    stream=True
                )
                for chunk in stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
                    
        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/api/chats")
async def get_chats():
    """List all saved chats."""
    chats = []
    for filepath in SAVED_CHATS_DIR.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                chats.append({
                    "id": data.get("id", filepath.stem),
                    "title": data.get("title", "Untitled"),
                    "timestamp": data.get("timestamp", ""),
                    "model": data.get("model", "general"),
                })
        except (json.JSONDecodeError, KeyError):
            continue
    
    # Sort by timestamp desc
    chats.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return chats

@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Load a specific chat."""
    filepath = SAVED_CHATS_DIR / f"{chat_id}.json"
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Chat not found")
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chats")
async def save_chat(chat: SaveChatRequest):
    """Save a chat session."""
    chat_id = chat.id if chat.id else str(uuid.uuid4())
    
    # Generate title if missing
    title = chat.title
    if not title and chat.messages:
        # User messages only
        user_msgs = [m for m in chat.messages if m.role == "user"]
        if user_msgs:
            first_msg = user_msgs[0].content
            title = first_msg[:50] + "..." if len(first_msg) > 50 else first_msg
        else:
            title = "Untitled Chat"
    
    data = {
        "id": chat_id,
        "title": title,
        "model": chat.model,
        "timestamp": datetime.now().isoformat(),
        "messages": [m.dict() for m in chat.messages] # Convert model to dict
    }
    
    filepath = SAVED_CHATS_DIR / f"{chat_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    return {"id": chat_id, "title": title}

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete a chat."""
    filepath = SAVED_CHATS_DIR / f"{chat_id}.json"
    if filepath.exists():
        filepath.unlink()
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Chat not found")

# Serve Frontend - index.html
@app.get("/")
async def serve_index():
    return FileResponse("index.html")

# Run functionality
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
