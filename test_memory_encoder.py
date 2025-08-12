"""Unit tests for MemoryEncoder and Agent memory integration."""
import pytest
import torch
import numpy as np

import agent1
from agent1 import MemoryEncoder, update_edge_memory, update_gate_memory, update_poi_memory


class DummyEnvSpace:
    def __init__(self):
        class Space:
            nvec = [14, 5]
        self.single_action_space = Space()


def test_encoder_empty_memory():
    encoder = MemoryEncoder()
    memory = {"edges": {}, "gates": {}, "poi": {}}
    vec = encoder(memory)
    assert vec.shape == (64,)
    assert not torch.isnan(vec).any()
    assert not torch.isinf(vec).any()


def test_encoder_populated_edges():
    encoder = MemoryEncoder()
    memory = {"edges": {}, "gates": {}, "poi": {}}
    update_edge_memory(memory, 1, 2, "right", died=False)
    update_edge_memory(memory, 2, 3, "left", died=True)
    update_edge_memory(memory, 3, 4, None, died=False)  # direction=None should be skipped
    vec = encoder(memory)
    assert vec.shape == (64,)
    assert not torch.isnan(vec).any()


def test_encoder_populated_poi():
    encoder = MemoryEncoder()
    memory = {"edges": {}, "gates": {}, "poi": {}}
    update_poi_memory(memory, 1, 2, 0, "sword")
    update_poi_memory(memory, 1, 5, 1, "potion_big")
    update_poi_memory(memory, 2, 3, 2, "potion_small")
    update_poi_memory(memory, 4, 1, 1, "unknown_kind")
    vec = encoder(memory)
    assert vec.shape == (64,)
    assert not torch.isnan(vec).any()


def test_encoder_populated_gates():
    encoder = MemoryEncoder()
    memory = {"edges": {}, "gates": {}, "poi": {}}
    switch1 = (5, 3, 1, "opener", "press")
    gates1 = [(2, 4, 0, True), (2, 4, 1, False)]
    update_gate_memory(memory, switch1, gates1)

    switch2 = (6, 1, 2, "closer", "release")
    gates2 = [(3, 1, 0, False)]
    update_gate_memory(memory, switch2, gates2)

    vec = encoder(memory)
    assert vec.shape == (64,)
    assert not torch.isnan(vec).any()


def test_encoder_gradient_flow():
    encoder = MemoryEncoder()
    memory = {"edges": {}, "gates": {}, "poi": {}}
    update_edge_memory(memory, 1, 2, "right", died=False)
    update_poi_memory(memory, 1, 2, 0, "sword")
    update_gate_memory(memory, (5, 3, 1, "opener", "press"), [(2, 4, 0, True)])

    vec = encoder(memory)
    loss = vec.pow(2).sum()
    loss.backward()

    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN grad for {name}"


def test_agent_integration_with_mem_vec():
    envs = DummyEnvSpace()
    agent = agent1.Agent(envs)
    memory = {"edges": {}, "gates": {}, "poi": {}}
    update_edge_memory(memory, 1, 2, "right", died=False)

    mem_vec = agent.mem_encoder(memory)

    obs = {
        "grid": torch.zeros((2, 60, 5, 12), dtype=torch.float32),
        "state": torch.zeros((2, 30), dtype=torch.float32),
        "room": torch.zeros((2, 1), dtype=torch.int32),
        "action_history": torch.zeros((2, 5), dtype=torch.int32),
        "repeat_history": torch.zeros((2, 5), dtype=torch.int32),
    }

    action, logprob, entropy, value = agent.get_action_and_value(obs, mem_vec)
    assert action.shape == (2, 2)
    assert logprob.shape == (2,)
    assert value.shape == (2, 1)
    assert not torch.isnan(value).any()


def test_level_transition_memory_clear():
    memory = {"edges": {}, "gates": {}, "poi": {}}
    update_edge_memory(memory, 1, 2, "right", died=False)
    update_poi_memory(memory, 1, 2, 0, "sword")
    update_gate_memory(memory, (5, 3, 1, "opener", "press"), [(2, 4, 0, True)])

    assert len(memory["edges"]) > 0
    assert len(memory["poi"]) > 0
    assert len(memory["gates"]) > 0

    # simulate level clear
    memory["edges"].clear()
    memory["poi"].clear()
    memory["gates"].clear()

    assert len(memory["edges"]) == 0
    assert len(memory["poi"]) == 0
    assert len(memory["gates"]) == 0
