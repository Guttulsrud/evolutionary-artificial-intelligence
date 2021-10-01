from src.cellular_automata.CAC import CellularAutomataController

config = {
    "time_steps": 10,
    "high": 1,
    "high_threshold": 0.1,
    "low": 1,
    "low_threshold": -0.1,
    "output": 1,
    "n_neighbours": 2,
    "width": 12,
    "rule_number": 191,
    "boundary_condition": "periodic"
}

CAC = CellularAutomataController(config=config)

CAC.run(None)
