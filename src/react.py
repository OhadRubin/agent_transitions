from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import gymnasium as gym
from transitions import Machine
import numpy as np
from pydantic import BaseModel, ValidationError, BeforeValidator
from typing_extensions import Annotated
import openai
from openai import OpenAI
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, replace

class StateMachineEnv(gym.Env):
    """
    A base class for creating Gymnasium environments with state machines.
    Supports flexible callbacks for transitions.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        states: List[str],
        actions: Dict[int, Union[str, Dict[str, Any]]],
        initial_state: str,
        transitions: List[Dict[str, Any]],
        observation_space: gym.Space,
        callbacks: Dict[str, Dict[str, Union[Callable, List[Callable]]]] = None,
        render_mode: Optional[str] = None,
        max_steps: int = 1000
    ):
        """
        Initialize the environment.
        
        Args:
            states: List of possible states
            actions: Dictionary mapping action integers to transition triggers
            initial_state: Starting state
            transitions: List of transition dictionaries with keys:
                - trigger: name of the transition
                - source: source state(s)
                - dest: destination state
            callbacks: Dictionary of callbacks for each transition:
                {
                    'trigger_name': {
                        'conditions': single callable or list of callables,
                        'before': single callable or list of callables,
                        'after': single callable or list of callables,
                        'on_enter': single callable or list of callables for destination state,
                        'on_exit': single callable or list of callables for source state
                    }
                }
            observation_space: Gymnasium space for observations
            render_mode: How to render the environment
            max_steps: Maximum steps before truncation
        """
        super().__init__()

        self.states = states
        self.actions = actions
        self.initial_state = initial_state
        self.max_steps = max_steps
        self.steps = 0
        self.callbacks = callbacks or {}

        # Set up action space based on number of defined actions
        self.action_space = gym.spaces.Discrete(len(actions))

        # Set up observation space as provided
        self.observation_space = observation_space

        # Initialize state machine with default callbacks for each state
        self.machine = Machine(
            model=self,
            states=states,
            initial=initial_state,
            after_state_change=['_update_observation']
        )

        # Add transitions with their callbacks
        self._add_transitions(transitions)

        # Set up rendering
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize observation
        self._observation = None
        self._update_observation()

    def _add_transitions(self, transitions: List[Dict[str, Any]]):
        """Add transitions with their associated callbacks."""
        for transition in transitions:
            trigger = transition['trigger']

            # Get callbacks for this trigger
            trigger_callbacks = self.callbacks.get(trigger, {})

            # Add callbacks to transition definition
            transition_with_callbacks = transition.copy()

            # Add condition callbacks
            conditions = trigger_callbacks.get('conditions', [])
            if callable(conditions):
                conditions = [conditions]
            if conditions:
                transition_with_callbacks['conditions'] = conditions

            # Add before callbacks
            before = trigger_callbacks.get('before', [])
            if callable(before):
                before = [before]
            if before:
                transition_with_callbacks['before'] = before

            # Add after callbacks
            after = trigger_callbacks.get('after', [])
            if callable(after):
                after = [after]
            if after:
                transition_with_callbacks['after'] = after

            # Add the transition with all callbacks
            self.machine.add_transition(**transition_with_callbacks)

    def _update_observation(self):
        """Update the observation based on current state."""
        pass

    def get_observation(self) -> Any:
        """Get the current observation."""
        return self._observation

    def compute_reward(self, action: int) -> float:
        """Compute reward for current state and action."""
        return 0.0

    def is_terminated(self) -> bool:
        """Check if episode should terminate."""
        return False

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[Any, dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        self.steps = 0
        self.machine.set_state(self.initial_state)
        self._update_observation()

        return self.get_observation(), {}

    def step(self, action: Union[int, Tuple[int, Dict[str, Any]]]) -> Tuple[Any, float, bool, bool, dict]:
        """Execute environment step with optional parameters."""
        self.steps += 1

        # Handle both simple actions and parameterized actions
        action_params = {}
        if isinstance(action, tuple):
            action_idx, action_params = action
        else:
            action_idx = action

        # Get action definition
        action_def = self.actions.get(action_idx)

        # Handle string (simple trigger) or dict (parameterized trigger)
        if isinstance(action_def, dict):
            trigger = action_def['trigger']
            # Merge default params with provided params
            action_params = {**action_def.get('default', {}), **action_params}
        else:
            trigger = action_def

        # Try to execute the transition
        if trigger is not None:
            try:
                trigger_method = getattr(self, trigger)
                trigger_method(**action_params)
            except Exception as e:
                # Handle invalid transitions silently
                pass

        # Get reward and check termination
        reward = self.compute_reward(action_idx)
        terminated = self.is_terminated()
        truncated = self.steps >= self.max_steps

        return (
            self.get_observation(),
            reward,
            terminated,
            truncated,
            {'state': self.state}
        )


from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List
# from src.state_machine_env import StateMachineEnv


@dataclass(frozen=True)
class StateContext:
    """Immutable context object to hold state data"""

    state: str
    data: Dict[str, Any]
    observation: Any


class FunctionalStateEnvironment:
    def __init__(
        self,
        states: List[str],
        actions: Dict[int, str],
        initial_state: str,
        transitions: List[Dict[str, Any]],
        state_handlers: Dict[str, Callable[[StateContext, Any], StateContext]],
    ):
        self._state_machine = StateMachineEnv(
            states=states,
            actions=actions,
            initial_state=initial_state,
            transitions=transitions,
            observation_space=None,  # Define based on your needs
        )
        self._state_handlers = state_handlers
        self._context = StateContext(state=initial_state, data={}, observation=None)

    def apply_action(self, action: int) -> StateContext:
        """Pure function to compute next state"""
        observation, reward, terminated, truncated, info = self._state_machine.step(
            action
        )

        # Get handler for current state
        handler = self._state_handlers.get(info["state"])

        if handler:
            # Create new context immutably
            new_context = handler(self._context, action)
        else:
            new_context = StateContext(
                state=info["state"], data=self._context.data, observation=observation
            )

        self._context = new_context
        return new_context


# Initialize OpenAI client
client = OpenAI()


@dataclass(frozen=True)
class ReactState:
    thought: Optional[str] = None
    action: Optional[str] = None
    observation: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReactContext(StateContext):
    history: List[ReactState] = field(default_factory=list)
    current: ReactState = field(default_factory=lambda: ReactState())


class ReactEnvironment(FunctionalStateEnvironment):
    def __init__(
        self,
        task_description: str,
        available_actions: List[str],
        observation_formatter: Callable[[str, Any], str],
    ):
        # Initialize state machine with React states
        super().__init__(
            states=["THINKING", "ACTING", "OBSERVING", "FINISHED"],
            actions={
                0: "think",  # Generate thought
                1: "act",  # Execute action
                2: "observe",  # Process observation
            },
            initial_state="THINKING",
            transitions=[
                {"trigger": "think", "source": "*", "dest": "THINKING"},
                {"trigger": "act", "source": "THINKING", "dest": "ACTING"},
                {"trigger": "observe", "source": "ACTING", "dest": "OBSERVING"},
                {"trigger": "finish", "source": "*", "dest": "FINISHED"},
            ],
            state_handlers={
                "THINKING": self._handle_thinking,
                "ACTING": self._handle_acting,
                "OBSERVING": self._handle_observing,
                "FINISHED": self._handle_finished,
            },
        )
        self.task_description = task_description
        self.available_actions = available_actions
        self.observation_formatter = observation_formatter
        
        # Initialize ReactContext instead of StateContext
        self._context = ReactContext(
            state="THINKING",
            data={},
            observation=None,
            history=[],
            current=ReactState()
        )

    def _create_thinking_prompt(self, context: ReactContext) -> str:
        """Create prompt for thinking state"""
        prompt = f"Task: {self.task_description}\n\n"
        prompt += "Available actions:\n"
        for action in self.available_actions:
            prompt += f"- {action}\n"

        prompt += "\nHistory:\n"
        for state in context.history:
            if state.context:
                prompt += f"Context: {state.context}\n"

        prompt += "\nWhat should I do next? Express your reasoning."
        return prompt

    def _handle_thinking(self, context: ReactContext, action: int) -> ReactContext:
        """Generate next thought using LLM"""
        prompt = self._create_thinking_prompt(context)

        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        thought = response.choices[0].message.content

        new_state = ReactState(
            thought=thought,  # Set the thought explicitly
            action=None,
            observation=None,
            context=context.current.context
        )

        return ReactContext(
            state="THINKING",
            data=context.data,
            observation=context.observation,
            history=context.history + [context.current],
            current=new_state
        )

    def _create_acting_prompt(self, thought: str) -> str:
        """Create prompt for action parsing"""
        prompt = f"""Based on this thought, what action should be taken?
Available actions: {', '.join(self.available_actions)}

Thought: {thought}

Extract the most appropriate action from the thought. Return just the action name."""
        return prompt

    def _handle_acting(self, context: ReactContext, action: int) -> ReactContext:
        """Execute action based on thought"""
        prompt = self._create_acting_prompt(context.current.thought)  # Use thought instead of context

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            # Return to thinking state on error
            return self._handle_thinking(context, action)

        action_name = response.choices[0].message.content.strip()

        if action_name not in self.available_actions:
            return self._handle_thinking(context, action)

        new_state = ReactState(
            thought=context.current.thought,
            action=action_name,  # Set the action explicitly
            observation=None,
            context=context.current.context
        )

        return ReactContext(
            state="ACTING",
            data=context.data,
            observation=context.observation,
            history=context.history + [context.current],
            current=new_state
        )

    def _handle_observing(self, context: ReactContext, action: int) -> ReactContext:
        """Process observation from environment"""
        # Get observation from environment
        observation = self.observation_formatter(
            context.current.action,
            context.observation if context.observation else "no observation"
        )

        new_state = ReactState(
            thought=context.current.thought,
            action=context.current.action,
            observation=observation,  # Set the observation explicitly
            context=context.current.context
        )

        return ReactContext(
            state="OBSERVING",
            data=context.data,
            observation=observation,
            history=context.history + [context.current],
            current=new_state
        )

    def _handle_finished(self, context: ReactContext, action: int) -> ReactContext:
        """Handle completion"""
        return context

    def apply_action(self, action: int) -> ReactContext:
        """Pure function to compute next state"""
        # Execute the state machine transition first
        observation, reward, terminated, truncated, info = self._state_machine.step(action)
        
        # Update the context state based on the state machine's current state
        updated_context = ReactContext(
            state=info["state"],  # Use state from info
            data=self._context.data,
            observation=observation,  # Use the observation from state machine
            history=self._context.history,
            current=self._context.current
        )

        # Get handler for the new state
        handler = self._state_handlers.get(updated_context.state)

        if handler:
            new_context = handler(updated_context, action)
        else:
            new_context = updated_context

        # Handle termination if needed
        if terminated or truncated:
            new_context = replace(new_context, state="FINISHED")

        self._context = new_context
        return new_context

    @property
    def action_space(self) -> gym.spaces.Discrete:
        """Return the action space"""
        return gym.spaces.Discrete(3)  # think, act, observe


def example_usage():
    # Initialize environment
    env = ReactEnvironment(
        task_description="Find a red book on the shelf",
        available_actions=["search[entity]", "lookup[string]", "finish[answer]"],
        observation_formatter=lambda action, obs: f"Observed: {obs}",
    )

    # Run ReAct loop
    context = env._context
    while context.state != "FINISHED":
        context = env.apply_action(0)  # Think
        context = env.apply_action(1)  # Act
        context = env.apply_action(2)  # Observe
