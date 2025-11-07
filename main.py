# main.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse
import uvicorn
from app.voice_agent.views import router as voice_router
from app.comparison.views import router as comparison_router

app = FastAPI(title="Voice Agent API", description="Speech-to-Speech API with Streaming Comparison")

# CORS for browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates for HTML
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(voice_router)
app.include_router(comparison_router)

# Root redirect to /voice_agent
@app.get("/")
async def redirect_to_voice_agent():
    return RedirectResponse(url="/voice_agent")

# Comparison UI
@app.get("/compare", response_class=HTMLResponse)
async def comparison_page(request: Request):
    return templates.TemplateResponse("compare.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)