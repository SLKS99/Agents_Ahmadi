"""
FastAPI server that watches the filesystem and triggers agents via HTTP endpoints.

This uses `watchdog`, which internally selects the best backend:
- Windows: FileSystemWatcher
- macOS: FSEvents
- Linux/other: inotify/polling

The server exposes REST endpoints that the WatcherAgent can call to trigger routing.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from tools.memory import MemoryManager
from agents.router import AgentRouter
from agents.hypothesis_agent import HypothesisAgent
from agents.experiment_agent import ExperimentAgent
from agents.curve_fitting_agent import CurveFittingAgent
from agents.watcher_agent import WatcherAgent
from agents.fallback_agent import FallbackAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="POLARIS Watcher Server", version="1.0.0")

# Global state
memory = MemoryManager()
observer: Optional[Observer] = None
watch_dir: Optional[Path] = None

# Request/Response models
class FileEventRequest(BaseModel):
    file_path: str
    event_type: str = "created"
    metadata: Optional[Dict[str, Any]] = None

class RouteRequest(BaseModel):
    payload: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    watch_dir: Optional[str] = None
    observer_running: bool = False


class Watcher(FileSystemEventHandler):
    """
    Handles filesystem events. When certain files appear (e.g.
    `output_from_agent_A.json`), this notifies the router so the next agent
    in the workflow can run.
    """

    def __init__(self, memory: MemoryManager, watch_dir: Path):
        super().__init__()
        self.memory = memory
        self.watch_dir = watch_dir

    def _build_router(self) -> AgentRouter:
        """
        Construct a router instance.  In this headless context we only need
        minimal agent objects – they can ignore Streamlit UI.
        """
        # Dummy descriptions; router only needs names + confidence/run_agent
        hyp = HypothesisAgent(name="Hypothesis Agent", desc="Background", question="")
        exp = ExperimentAgent(name="Experiment Agent", desc="Background", params_const={})
        cf = CurveFittingAgent(name="Curve Fitting", desc="Background")
        fb = FallbackAgent(name="Fallback Agent", desc="Handles failures")

        return AgentRouter(
            agents=[WatcherAgent(), hyp, exp, cf],
            fallback_agent=fb,
        )

    def _route_next(self, trigger_file: Path) -> None:
        payload: Dict[str, Any] = {
            "source": "filesystem",
            "trigger_file": str(trigger_file),
            "event": "created",
        }
        router = self._build_router()
        try:
            router.route(payload, self.memory)
        except Exception as e:
            logger.error(f"Error routing: {e}")

    def on_created(self, event):
        if event.is_directory:
            return

        path = Path(event.src_path)

        # Example conventions:
        # - output_from_hypothesis.json → trigger Experiment Agent next
        # - output_from_experiment.json → trigger Curve Fitting Agent next
        # You can customize these patterns as needed.
        filename = path.name

        if filename.startswith("output_from_") and filename.endswith(".json"):
            # Log event and route
            self.memory.log_event(
                "watcher",
                {"file": str(path), "event": "created"},
                mode="watcher",
            )
            self._route_next(path)


# Global watcher instance
watcher_handler: Optional[Watcher] = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        watch_dir=str(watch_dir) if watch_dir else None,
        observer_running=observer is not None and observer.is_alive()
    )


@app.post("/watch/start")
async def start_watching(watch_directory: Optional[str] = None):
    """Start watching a directory for filesystem events"""
    global observer, watch_dir, watcher_handler
    
    if observer is not None and observer.is_alive():
        return JSONResponse(
            status_code=400,
            content={"error": "Observer is already running"}
        )
    
    # Determine watch directory
    if watch_directory:
        watch_dir = Path(watch_directory)
    else:
        watch_dir = Path(os.environ.get("WATCH_DIR", os.getcwd()))
    
    if not watch_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Watch directory does not exist: {watch_dir}"
        )
    
    watcher_handler = Watcher(memory=memory, watch_dir=watch_dir)
    observer = Observer()
    observer.schedule(watcher_handler, path=str(watch_dir), recursive=False)
    observer.start()
    
    logger.info(f"Started watching directory: {watch_dir}")
    
    return {
        "status": "started",
        "watch_dir": str(watch_dir),
        "message": f"Watching directory: {watch_dir}"
    }


@app.post("/watch/stop")
async def stop_watching():
    """Stop watching for filesystem events"""
    global observer
    
    if observer is None or not observer.is_alive():
        return JSONResponse(
            status_code=400,
            content={"error": "Observer is not running"}
        )
    
    observer.stop()
    observer.join()
    observer = None
    
    logger.info("Stopped watching")
    
    return {"status": "stopped", "message": "Observer stopped"}


@app.post("/route")
async def route_event(request: RouteRequest, background_tasks: BackgroundTasks):
    """Route a filesystem event to the appropriate agent"""
    try:
        # Route in background
        background_tasks.add_task(_route_event_task, request.payload)
        
        return {
            "status": "routed",
            "payload": request.payload,
            "message": "Event routed to appropriate agent"
        }
    except Exception as e:
        logger.error(f"Error routing event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _route_event_task(payload: Dict[str, Any]):
    """Background task to route an event"""
    try:
        router = watcher_handler._build_router() if watcher_handler else None
        if not router:
            # Create a temporary router if watcher not initialized
            hyp = HypothesisAgent(name="Hypothesis Agent", desc="Background", question="")
            exp = ExperimentAgent(name="Experiment Agent", desc="Background", params_const={})
            cf = CurveFittingAgent(name="Curve Fitting", desc="Background")
            fb = FallbackAgent(name="Fallback Agent", desc="Handles failures")
            router = AgentRouter(
                agents=[WatcherAgent(), hyp, exp, cf],
                fallback_agent=fb,
            )
        router.route(payload, memory)
    except Exception as e:
        logger.error(f"Error in routing task: {e}")

@app.post("/file-event")
async def handle_file_event(event: FileEventRequest, background_tasks: BackgroundTasks):
    """Handle a filesystem event (file created/modified)"""
    try:
        file_path = Path(event.file_path)
        
        # Log the event
        memory.log_event(
            "watcher",
            {
                "file": str(file_path),
                "event": event.event_type,
                "metadata": event.metadata or {}
            },
            mode="watcher",
        )
        
        # Build payload for routing
        payload: Dict[str, Any] = {
            "source": "filesystem",
            "trigger_file": str(file_path),
            "event": event.event_type,
            "metadata": event.metadata or {}
        }
        
        # Route in background
        background_tasks.add_task(_route_event_task, payload)
        
        return {
            "status": "processed",
            "file": str(file_path),
            "event": event.event_type,
            "message": "File event processed and routed"
        }
    except Exception as e:
        logger.error(f"Error handling file event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    global observer
    if observer is not None and observer.is_alive():
        observer.stop()
        observer.join()
        logger.info("Observer stopped on shutdown")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("WATCHER_PORT", 8000))
    host = os.environ.get("WATCHER_HOST", "0.0.0.0")
    
    logger.info(f"Starting POLARIS Watcher Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
