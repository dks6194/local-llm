# 🤖 Local LLM Chat & Model Manager

A self-hosted, web-based chat application and model management system powered by **Ollama** and **FastAPI**. This project simplifies the process of setting up and managing local Large Language Models (LLMs) with a premium web interface.

## ✨ Features

- **🚀 Instant Setup Wizard**: No more CLI-based setup. Start the container and choose your models through a beautiful web interface.
- **💬 Premium Chat UI**: A clean, dark-mode chat interface with sidebar history, Markdown support, and code highlighting.
- **⚙️ Dynamic Model Management**:
  - **In-App Model settings**: Add or remove models directly from the chat UI.
  - **Category-based Sorting**: Distinct "General" and "Coder" model categories.
  - **Usage Variants**: Choose between "Low System Usage" (smaller quantized models) and "High System Usage" (larger models) with RAM estimates.
- **💾 Disk Usage Tracking**: Real-time monitoring of disk space used by each model and total system usage.
- **🛡️ Persistence**: Your model choices and chat history are saved in the `./config` and `./saved_chats` directories, surviving container restarts.
- **🐳 Dockerized**: Single-command deployment with everything bundled.

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, Uvicorn, Subprocess (Ollama interaction)
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (no framework overhead)
- **Engine**: Ollama (Local LLM runner)
- **Deployment**: Docker, Docker Compose

## 🚀 Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Installation & Run

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd local_llm
   ```

2. **Configure (Optional)**:
   Create a `.env` file from the example if you want to use specific features like the news tool:
   ```bash
   cp .env.example .env
   ```

3. **Start the application**:
   ```bash
   docker compose up --build
   ```

4. **Access the App**:
   Open your browser and navigate to `http://localhost:8000`.
   - If it's your first run, you will be redirected to the **Setup Wizard** to choose your models.
   - Once models are downloaded, you can start chatting!

## 📂 Project Structure

- `main.py`: The FastAPI backend handling chat logic and model management APIs.
- `index.html`: The main chat interface + settings panel.
- `setup.html`: The initial configuration wizard.
- `start.sh`: Container entrypoint script that initializes Ollama and the web server.
- `config/`: (Auto-generated) Stores your selected model configuration.
- `saved_chats/`: (Auto-generated) Stores your chat history.
- `Dockerfile` & `docker-compose.yml`: Container orchestration setup.

## ⚙️ Timezone Settings
By default, the application is set to `Asia/Kolkata` (IST). To change this, update the `TZ` environment variable in `docker-compose.yml`:
```yaml
environment:
  - TZ=Your/Timezone
```

## 📜 License
This project is open-source. Feel free to use and modify!
