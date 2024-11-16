import unittest
from unittest.mock import Mock, patch
import gymnasium as gym
import numpy as np
from typing import List, Dict, Any

from src.react import ReactEnvironment, ReactContext, ReactState

# Mock OpenAI API responses
MOCK_THOUGHT_RESPONSE = {
    "choices": [{
        "message": {
            "content": "I should search for the red book on the shelf first."
        }
    }]
}

MOCK_ACTION_RESPONSE = {
    "choices": [{
        "message": {
            "content": "search[entity]"
        }
    }]
}

class TestReactEnvironment(unittest.TestCase):
    def setUp(self):
        """Setup that runs before each test"""
        self.patcher = patch('src.react.client')
        self.mock_client = self.patcher.start()
        
        # Setup default response
        self.mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="I should search for the red book on the shelf first."))]
        )
        
        def mock_observation_formatter(action: str, obs: Any) -> str:
            return f"Found {obs} while performing {action}"

        self.env = ReactEnvironment(
            task_description="Find a red book on the shelf",
            available_lm_actions=["search[entity]", "lookup[string]", "finish[answer]"],
            observation_formatter=mock_observation_formatter
        )

    def tearDown(self):
        """Cleanup that runs after each test"""
        self.patcher.stop()

    def test_environment_initialization(self):
        """Test that environment is properly initialized"""
        self.assertEqual(self.env.task_description, "Find a red book on the shelf")
        self.assertIn("search[entity]", self.env.available_lm_actions)
        self.assertEqual(self.env._context.state, "THINKING")
        self.assertIsInstance(self.env._context, ReactContext)

    def test_thinking_state(self):
        """Test the thinking state handler"""
        self.mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content="I should search for the red book on the shelf first."))]
        )
        
        context = self.env._handle_thinking(self.env._context, 0)
        
        self.mock_client.chat.completions.create.assert_called_once()
        call_args = self.mock_client.chat.completions.create.call_args[1]
        self.assertIn("Task: Find a red book on the shelf", call_args["messages"][0]["content"])
        
        self.assertEqual(context.state, "THINKING")
        self.assertEqual(context.current.thought, "I should search for the red book on the shelf first.")

    def test_acting_state(self):
        """Test the acting state handler"""
        responses = [
            Mock(choices=[Mock(message=Mock(content="I should search for the red book on the shelf first."))]),
            Mock(choices=[Mock(message=Mock(content="search[entity]"))])
        ]
        self.mock_client.chat.completions.create.side_effect = responses
        
        context = self.env._handle_thinking(self.env._context, 0)
        context = self.env._handle_acting(context, 1)
        
        self.assertEqual(context.state, "ACTING")
        self.assertEqual(context.current.lm_action, "search[entity]")

    def test_observing_state(self):
        """Test the observing state handler"""
        initial_state = ReactState(
            thought="I should search for the red book",
            lm_action="search[entity]",
            context={"location": "shelf"}
        )
        context = ReactContext(
            state="ACTING",
            data={},
            observation=None,
            history=[],
            current=initial_state
        )
        
        new_context = self.env._handle_observing(context, 2)
        
        self.assertEqual(new_context.state, "OBSERVING")
        self.assertIsInstance(new_context.current.observation, str)
        self.assertIn("search[entity]", new_context.current.observation)

    def test_full_react_cycle(self):
        """Test a complete think-act-observe cycle"""
        responses = [
            Mock(choices=[Mock(message=Mock(content="I should search for the red book on the shelf first."))]),
            Mock(choices=[Mock(message=Mock(content="search[entity]"))])
        ]
        self.mock_client.chat.completions.create.side_effect = responses
        
        context = self.env._context
        
        context = self.env.apply_sm_action(0)
        self.assertEqual(context.state, "THINKING")
        self.assertIsNotNone(context.current.thought)
        
        context = self.env.apply_sm_action(1)
        self.assertEqual(context.state, "ACTING")
        self.assertIsNotNone(context.current.lm_action)
        
        context = self.env.apply_sm_action(2)
        self.assertEqual(context.state, "OBSERVING")
        self.assertIsNotNone(context.current.observation)

    def test_invalid_action_handling(self):
        """Test handling of invalid LM actions"""
        responses = [
            Mock(choices=[Mock(message=Mock(content="I should search"))]),
            Mock(choices=[Mock(message=Mock(content="invalid_action"))]),
            Mock(choices=[Mock(message=Mock(content="I should try something else"))])
        ]
        self.mock_client.chat.completions.create.side_effect = responses
        
        context = self.env._handle_thinking(self.env._context, 0)
        context = self.env._handle_acting(context, 1)
        
        self.assertEqual(context.state, "THINKING")

    def test_observation_formatting(self):
        """Test the observation formatter"""
        observation = "red book"
        formatted_obs = self.env.observation_formatter("search[entity]", observation)
        self.assertIsInstance(formatted_obs, str)
        self.assertIn("search[entity]", formatted_obs)
        self.assertIn("red book", formatted_obs)

    def test_action_space(self):
        """Test that all SM actions in action space are valid"""
        for action in [0, 1, 2]:
            self.assertTrue(self.env.action_space.contains(action))

    def test_context_immutability(self):
        """Test that ReactContext maintains immutability"""
        initial_state = ReactState(
            thought="initial thought",
            lm_action=None,
            observation=None
        )
        context = ReactContext(
            state="THINKING",
            data={},
            observation=None,
            history=[],
            current=initial_state
        )
        
        with self.assertRaises(Exception):
            context.state = "ACTING"

    def test_history_management(self):
        """Test that history is properly maintained through state transitions"""
        context = self.env._context
        for _ in range(3):
            context = self.env.apply_sm_action(_)
        
        self.assertGreater(len(context.history), 0)
        self.assertTrue(all(isinstance(state, ReactState) for state in context.history))
        
        history_states = [state.thought for state in context.history if state.thought]
        self.assertGreater(len(history_states), 0)

if __name__ == '__main__':
    unittest.main() 