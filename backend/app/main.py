import asyncio
from dataclasses import asdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel, Field
from .simulation import Config, World

app = FastAPI(title="Synapse Lab")
config = Config(); world = World(config); running = False; lock = asyncio.Lock()

class Settings(BaseModel):
    width: int = Field(ge=100, le=5000, default=800)
    height: int = Field(ge=100, le=5000, default=600)
    population: int = Field(ge=0, le=500, default=80)
    food_count: int = Field(ge=0, le=1000, default=180)
    predators: int = Field(ge=0, le=100, default=5)
    max_population: int = Field(ge=1, le=1000, default=500)
    seed: int = 42
    mutation_rate: float = Field(ge=0, le=1, default=.05)
    food_spawn_rate: int = Field(ge=0, le=50, default=2)

@app.get('/')
def dashboard():
    return FileResponse(Path(__file__).resolve().parents[2] / 'frontend' / 'index.html')

@app.get('/health')
def health(): return {"status":"ok"}
@app.get('/simulation/state')
def state(): return world.snapshot()
@app.get('/simulation/metrics')
def metrics(): return {"tick":world.tick,"metrics":world.snapshot()["metrics"]}
@app.post('/simulation/start')
async def start():
    global running
    running = True
    return {"running":running}
@app.post('/simulation/pause')
async def pause():
    global running
    running = False
    return {"running":running}
@app.post('/simulation/step')
async def step():
    async with lock: world.step()
    return world.snapshot()
@app.post('/simulation/reset')
async def reset():
    global world, running
    running = False
    async with lock: world = World(config)
    return world.snapshot()
@app.put('/simulation/config')
async def set_config(value: Settings):
    global config, world, running
    config = Config(**value.model_dump()); running = False
    async with lock: world = World(config)
    return world.snapshot()

@app.websocket('/simulation/stream')
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(world.snapshot())
            await asyncio.sleep(.1)
    except WebSocketDisconnect: pass

@app.on_event('startup')
async def simulation_loop():
    async def run():
        global running
        while True:
            if running:
                async with lock: world.step()
            await asyncio.sleep(.02)
    asyncio.create_task(run())
