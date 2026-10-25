"""Compact vectorized artificial-life simulation. One tick is deterministic per seed."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    width: int = 800
    height: int = 600
    population: int = 80
    food_count: int = 180
    predators: int = 5
    max_population: int = 500
    seed: int = 42
    mutation_rate: float = .05
    food_spawn_rate: int = 2

class World:
    """Array-backed state; vectorized metabolism and movement avoid per-entity objects."""
    def __init__(self, config: Config):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.tick = 0
        n = min(config.population + config.predators, config.max_population)
        self.pos = self.rng.random((n, 2)) * [config.width, config.height]
        self.energy = np.full(n, 20.0)
        self.species = np.r_[np.zeros(min(config.population, n), dtype=np.uint8), np.ones(max(0, n-config.population), dtype=np.uint8)]
        self.speed = self.rng.uniform(1.0, 2.5, n)
        self.vision = self.rng.uniform(45, 100, n)
        self.ids = np.arange(1, n+1, dtype=np.int64)
        self.next_id = n+1
        self.food = self.rng.random((config.food_count, 2)) * [config.width, config.height]
        self.births = 0
        self.deaths = 0

    def step(self):
        c = self.config
        alive = self.energy > 0
        self.deaths += int((~alive).sum())
        self.pos, self.energy, self.species, self.speed, self.vision, self.ids = (x[alive] for x in (self.pos, self.energy, self.species, self.speed, self.vision, self.ids))
        n = len(self.energy)
        if n:
            # Vectorized nearest food/prey queries. Capped food count keeps this bounded.
            herb = self.species == 0
            pred = ~herb
            targets = np.zeros_like(self.pos)
            if len(self.food) and herb.any():
                delta = self.food[None, :, :] - self.pos[herb, None, :]
                d2 = np.einsum('ijk,ijk->ij', delta, delta)
                nearest = d2.argmin(axis=1)
                targets[herb] = self.food[nearest]
            if pred.any() and herb.any():
                delta = self.pos[herb][None, :, :] - self.pos[pred, None, :]
                d2 = np.einsum('ijk,ijk->ij', delta, delta)
                nearest = d2.argmin(axis=1)
                targets[pred] = self.pos[herb][nearest]
            direction = targets - self.pos
            norm = np.linalg.norm(direction, axis=1)
            wander = self.rng.normal(0, 1, (n, 2))
            direction = np.divide(direction, norm[:, None], out=wander, where=norm[:, None] > 1e-6)
            direction[~herb] *= 1.2
            self.pos += direction * self.speed[:, None]
            self.pos[:, 0] = np.clip(self.pos[:, 0], 0, c.width)
            self.pos[:, 1] = np.clip(self.pos[:, 1], 0, c.height)
            self.energy -= np.where(herb, .06, .09) + self.speed * .015
            if len(self.food) and herb.any():
                delta = self.pos[herb, None, :] - self.food[None, :, :]
                d2 = np.einsum('ijk,ijk->ij', delta, delta)
                hit = d2.min(axis=0) < 100
                eaten = np.flatnonzero(d2.argmin(axis=1)[d2.min(axis=1) < 100])
                self.energy[np.flatnonzero(herb)[eaten]] += 8
                self.food = self.food[~hit]
            if pred.any() and herb.any():
                hi = np.flatnonzero(herb); pi = np.flatnonzero(pred)
                d = self.pos[pi, None, :] - self.pos[hi][None, :, :]
                d2 = np.einsum('ijk,ijk->ij', d, d)
                caught = d2.min(axis=0) < 36
                if caught.any():
                    prey = hi[caught]; self.energy[pi[np.argmin(d2[:, caught], axis=0)]] += 12
                    self.energy[prey] = 0
        # Regeneration and reproduction are bounded; offspring act starting next tick.
        if c.food_spawn_rate:
            add = self.rng.random((c.food_spawn_rate, 2)) * [c.width, c.height]
            self.food = np.concatenate((self.food, add))[-1000:]
        room = max(0, c.max_population - len(self.energy))
        parents = np.flatnonzero((self.energy > 34) & (self.species == 0))[:room]
        if len(parents):
            kids = self.pos[parents] + self.rng.normal(0, 8, (len(parents), 2))
            kids[:, 0] = np.clip(kids[:, 0], 0, c.width); kids[:, 1] = np.clip(kids[:, 1], 0, c.height)
            self.energy[parents] -= 15
            k = len(parents)
            self.pos = np.concatenate((self.pos, kids)); self.energy = np.r_[self.energy, np.full(k, 16.)]
            self.species = np.r_[self.species, np.zeros(k, dtype=np.uint8)]
            self.speed = np.r_[self.speed, np.clip(self.speed[parents] + self.rng.normal(0, .1, k), .5, 4)]
            self.vision = np.r_[self.vision, np.clip(self.vision[parents] + self.rng.normal(0, 4, k), 10, 200)]
            self.ids = np.r_[self.ids, np.arange(self.next_id, self.next_id+k)]; self.next_id += k
            self.births += k
        self.tick += 1

    def snapshot(self):
        return {"tick": self.tick, "organisms": [{"id": int(self.ids[i]), "x": round(float(p[0]), 2), "y": round(float(p[1]), 2), "energy": round(float(self.energy[i]), 2), "species": "predator" if self.species[i] else "herbivore", "speed": round(float(self.speed[i]), 2)} for i, p in enumerate(self.pos)], "food": self.food.round(1).tolist(), "metrics": {"population": len(self.energy), "herbivores": int((self.species == 0).sum()), "predators": int((self.species == 1).sum()), "food": len(self.food), "births": self.births, "deaths": self.deaths, "mean_energy": round(float(self.energy.mean()), 2) if len(self.energy) else 0}}
