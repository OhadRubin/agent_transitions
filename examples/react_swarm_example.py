from react_swarm_agent import ReactSwarmAgent

def example_usage():
    # Initialize the React Swarm agent
    agent = ReactSwarmAgent(
        task_description="Find a red book on the shelf",
        available_actions=["search[entity]", "lookup[string]", "finish[answer]"],
        observation_formatter=lambda action, obs: f"Observed: {obs}"
    )
    
    # Run a conversation
    response = agent.run([
        {"role": "user", "content": "Help me find the red book"}
    ])
    
    print("Final Agent:", response.agent.name)
    print("Final Messages:", response.messages[-1]["content"])
    print("Context Variables:", response.context_variables) 