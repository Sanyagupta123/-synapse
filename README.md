# Synapse Lab

A small artificial-life playground built around an array-backed NumPy simulation and a Canvas dashboard.

## Run locally

Install the backend dependencies and start the API:

```powershell
cd backend
python -m pip install -r requirements.txt
python -m app
```

In another terminal, install and start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

The API listens at `http://127.0.0.1:8000`; Vite proxies requests and WebSocket traffic from its development server.

## Performance choices

Organism positions, energy, species, and traits live in NumPy arrays. Movement, energy updates, and nearest-target searches use vector operations; the UI batches the world into one Canvas draw. WebSocket frames are sent at 10 Hz independently from the simulation's 50 Hz update loop. This is a prototype, not a validated ecological model. Nearest-neighbor searches currently compare arrays directly and may become costly near the population and food caps; a spatial grid is the next optimization if profiling warrants it.

The engine is deterministic for a fixed seed and configuration. The dashboard shows live population, food, birth, and energy metrics. Configuration changes reset the world. Experimental comparison, neural network behavior, persistence/export, trait history charts, and a full test suite are not implemented yet.
