scenario_dirs = [
    "agent_SAC", "memory"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
agent_specs[scenario_dirs[0]] = 1
agent_specs[scenario_dirs[1]] = 2
print(agent_specs)