from swarm import Agent, Result
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class ReactState:
    thought: Optional[str] = None
    action: Optional[str] = None
    observation: Optional[str] = None
    context: Dict[str, Any] = None

class ReactSwarmAgent:
    def __init__(
        self,
        task_description: str,
        available_actions: List[str],
        observation_formatter: Callable[[str, Any], str],
        model: str = "gpt-4"
    ):
        """Initialize the React Swarm agent system.
        
        Args:
            task_description: Description of the task to be performed
            available_actions: List of action names that can be performed
            observation_formatter: Function to format observations
            model: Model to use for the agents
        """
        self.task_description = task_description
        self.available_actions = available_actions
        self.observation_formatter = observation_formatter

        # Create agents for thinking and acting
        self.thinking_agent = self._create_thinking_agent(model)
        self.acting_agent = self._create_acting_agent(model)
        
        # Initialize state
        self.current_state = ReactState(context={})

    def _create_thinking_agent(self, model: str) -> Agent:
        """Creates the thinking agent that generates reasoning traces."""

        def thinking_instructions(context_variables: Dict[str, Any]) -> str:
            prompt = f"Task: {self.task_description}\n\n"
            prompt += "Available actions:\n"
            for action in self.available_actions:
                prompt += f"- {action}\n"

            prompt += "\nHistory:\n"
            history = context_variables.get("history", [])
            for state in history:
                if state.get("context"):
                    prompt += f"Context: {state['context']}\n"

            prompt += "\nWhat should I do next? Express your reasoning."
            return prompt

        def transition_to_acting():
            return self.acting_agent

        return Agent(
            name="Thinking Agent",
            model=model,
            instructions=thinking_instructions,
            functions=[transition_to_acting]
        )

    def _create_acting_agent(self, model: str) -> Agent:
        """Creates the acting agent that executes actions."""

        def acting_instructions(context_variables: Dict[str, Any]) -> str:
            thought = context_variables.get("current_thought", "")
            return f"""Based on this thought, what action should be taken?
Available actions: {', '.join(self.available_actions)}

Thought: {thought}

Extract the most appropriate action from the thought. Return just the action name."""

        def execute_action(action_name: str) -> Result:
            if action_name not in self.available_actions:
                return Result(
                    value="Invalid action",
                    agent=self.thinking_agent
                )

            # Update state with action
            self.current_state.action = action_name
            
            # Get and format observation
            observation = "..." # Get observation from environment
            formatted_obs = self.observation_formatter(action_name, observation)
            
            # Update state and history
            self.current_state.observation = formatted_obs
            history = self.current_state.context.get("history", [])
            history.append({
                "thought": self.current_state.thought,
                "action": action_name,
                "observation": formatted_obs,
                "context": self.current_state.context
            })

            # Transition back to thinking
            return Result(
                value=formatted_obs,
                agent=self.thinking_agent,
                context_variables={
                    "history": history,
                    "current_observation": formatted_obs
                }
            )

        return Agent(
            name="Acting Agent",
            model=model,
            instructions=acting_instructions,
            functions=[execute_action]
        )

    def run(self, messages: List[Dict[str, str]], context_variables: Dict[str, Any] = None):
        """Run the React agent system.
        
        Args:
            messages: List of message objects
            context_variables: Optional context variables
        """
        from swarm import Swarm

        client = Swarm()

        # Initialize with thinking agent
        response = client.run(
            agent=self.thinking_agent,
            messages=messages,
            context_variables=context_variables or {}
        )

        return response 


def example_usage():
    # Initialize the React Swarm agent
    agent = ReactSwarmAgent(
        task_description="Find a red book on the shelf",
        available_actions=["search[entity]", "lookup[string]", "finish[answer]"],
        observation_formatter=lambda action, obs: f"Observed: {obs}",
    )

    # Run a conversation
    response = agent.run([{"role": "user", "content": "Help me find the red book"}])

    print("Final Agent:", response.agent.name)
    print("Final Messages:", response.messages[-1]["content"])
    print("Context Variables:", response.context_variables)

if __name__ == "__main__":
    example_usage()