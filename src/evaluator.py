"""Evaluator for running LLM game interactions on VideoGameBench."""
from typing import Dict, Any, Optional, List, Callable
import time
import random
import asyncio
from abc import ABC, abstractmethod
from src.llm.utils import convert_from_dict
from src.llm.realtime_agent import GameBoyAgent, WebBrowsingAgent
from src.consts import NO_OP_SKIP_FRAMES
from src.emulators.gba.interface import GBAInterface
from src.emulators.dos.website_server import DOSGameServer
from src.utils import is_same_image

class BaseVGBenchEvaluator(ABC):
    """Abstract base class for evaluators that coordinate between game emulators and LLMs."""
    
    def __init__(
        self, 
        max_steps: int = 1000,
        step_delay: float = 0.1,
        metrics: Optional[List[Callable]] = None,
        checkpoints: Optional[List[str]] = None
    ):
        self.max_steps = max_steps
        self.step_delay = step_delay
        self.metrics = metrics or []
        self.checkpoints = checkpoints

    @abstractmethod
    async def run_episode(self, agent) -> Dict[str, Any]:
        """Run an episode of the game."""
        pass

class GBEvaluator(BaseVGBenchEvaluator):
    """Evaluator class that coordinates between GB
     emulators (e.g. Game Boy) and LLMs."""
    
    def __init__(
        self, 
        game_interface: GBAInterface,
        max_steps: int = 1000,
        step_delay: float = 0.1,
        skip_frames: int = 1,
        metrics: Optional[List[Callable]] = None,
        fake_actions: bool = False,
        action_frames: int = 30,
        checkpoints: Optional[List[str]] = None
    ):
        super().__init__(max_steps, step_delay, metrics, checkpoints)
        self.game = game_interface
        self.fake_actions = fake_actions
        self.skip_frames = skip_frames
        self.action_frames = action_frames

    async def run_episode(self, gba_agent: GameBoyAgent) -> Dict[str, Any]:
        """
        Run an episode of a game using the Gym-based emulator and interacting with it
        at the frame level. The game pauses when the LLM is thinking.

        Part of VG-Bench Lite.
        """
        metrics = {
            'total_steps': 0,
            'total_reward': 0.0,
            'completed': False,
            'history': []
        }
        
        try:
            # Get initial observation
            obs = self.game.get_observation()
            if obs is None:
                return metrics
                
            # Initialize action list
            current_actions = []
            printable_action = None
            # Main loop
            for step in range(self.max_steps):
                # Get new actions if needed
                if not current_actions:
                    if self.fake_actions:
                        buttons = self.game.get_available_buttons()
                        random_button = buttons[int(time.time() * 1000) % len(buttons)]
                        action = {button: (button == random_button) for button in buttons}
                        current_actions = [action]
                    else:
                        # actions = self.llm.get_action(obs)
                        actions = await gba_agent.get_action(obs, prev_action=printable_action)
                        current_actions = actions if isinstance(actions, list) else [actions]
                        print("Parsed actions: ", current_actions)

                # Get next action from list
                if len(current_actions) > 0:
                    action = current_actions.pop(0)
                else:
                    action = None
                
                if action is None:
                    print(f"Current action: None.\nRemaining actions: {current_actions}")
                    # If no buttons pressed, call no_op
                    next_obs, reward, done, info = self.game.no_op(NO_OP_SKIP_FRAMES)
                else: 
                    printable_action = convert_from_dict(action)
                    # Take step in environment
                    print(f"Current action: {printable_action} for {self.action_frames} frames.\nRemaining actions: {current_actions}")
                    next_obs, reward, done, info = self.game.step(action, skip_frames = self.action_frames)

                # Store all but last action in sequence, that gets stored above
                if len(current_actions) > 0:
                    gba_agent.store_observation(next_obs, prev_action=printable_action)

                # Record step
                step_data = {
                    'observation': obs,
                    'action': action,
                    'reward': reward,
                    'next_observation': next_obs,
                    'done': done,
                    'info': info
                }
                metrics['history'].append(step_data)
                metrics['total_steps'] += 1
                metrics['total_reward'] += reward
                
                # Update observation
                obs = next_obs
                
                # Optional delay between steps
                if self.step_delay > 0:
                    time.sleep(self.step_delay)
                    
                if self.checkpoints and is_same_image(obs, self.checkpoints[-1]):
                    metrics['completed'] = True
                    break
                else:
                    metrics['completed'] = False
                    
            # Calculate additional metrics if provided
            if self.metrics:
                metrics['custom_metrics'] = {}
                for metric_fn in self.metrics:
                    metrics['custom_metrics'][metric_fn.__name__] = metric_fn(metrics['history'])
                    
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user")
        except Exception as e:
            print(f"Error during evaluation: {e}")
        finally:
            self.game.close()
            
        return metrics

    async def run_episode_realtime(self, gba_agent: GameBoyAgent) -> Dict[str, Any]:
        """
        Run an episode of a game using the Gym-based emulator with LLM interacting
        in realtime, e.g. async calls to get actions and observations.

        Part of VG-Bench
        """
        metrics = {
            'total_steps': 0,
            'total_reward': 0.0,
            'completed': False,
            'history': [],
            'response_times': []
        }
        
        try:
            # Get initial observation
            obs = self.game.get_observation()
            if obs is None:
                return metrics
                
            # Initialize action list and frame counter
            current_actions = []
            current_action = None
            frames_left = 0
            
            # Main loop
            for step in range(self.max_steps):
                # If we've used all actions, request new ones from LLM
                if not current_actions and frames_left == 0:
                    start_time = time.time()
                    action_task = asyncio.create_task(gba_agent.get_action(obs))
                    
                    # Call no_op() while waiting for the LLM response
                    while not action_task.done():
                        no_op_obs, _, _, _ = self.game.no_op(self.skip_frames)
                        await asyncio.sleep(0.01)
                    
                    # Get the new list of actions
                    current_actions = await action_task
                    if not isinstance(current_actions, list):
                        current_actions = [current_actions]  # Convert single action to list
                    
                    response_time = time.time() - start_time
                    metrics['response_times'].append(response_time)
                    print(f"Got new actions: {current_actions} (response time: {response_time:.2f}s)")
                
                if len(current_actions) > 0 or frames_left > 0:
                    # Get next action if needed
                    if frames_left == 0:
                        current_action = current_actions.pop(0)
                        frames_left = 1 # self.action_frames
                        
                        print(f"Current action: {current_action}, Remaining actions: {current_actions}")
                    
                    # Take step in environment
                    next_obs, reward, done, info = self.game.step(current_action, self.action_frames)

                    frames_left -= 1
                    if frames_left == 0:
                        gba_agent.store_observation(next_obs)
                    
                    # Record step
                    step_data = {
                        'observation': obs,
                        'action': current_action,
                        'reward': reward,
                        'next_observation': next_obs,
                        'done': done,
                        'info': info
                    }
                    metrics['history'] = [step_data]  # Only keep most recent step
                    
                    # Update metrics
                    metrics['total_steps'] += 1
                    metrics['total_reward'] += reward
                    
                    # Update observation
                    obs = next_obs
                
                # Optional delay between steps
                if self.step_delay > 0:
                    # Call no_op() while waiting for the LLM response
                    for _ in range(int(self.step_delay * 100)):  # Convert seconds to 10ms intervals
                        obs, _, _, _ = self.game.no_op(self.skip_frames)
                        await asyncio.sleep(0.01)
                    
                if self.checkpoints and is_same_image(obs, self.checkpoints[-1]):
                    metrics['completed'] = True
                    break
                else:
                    metrics['completed'] = False
                    
            # Calculate additional metrics if provided
            if self.metrics:
                metrics['custom_metrics'] = {}
                for metric_fn in self.metrics:
                    metrics['custom_metrics'][metric_fn.__name__] = metric_fn(metrics['history'])
                
            # Calculate average response time
            if metrics['response_times']:
                metrics['avg_response_time'] = sum(metrics['response_times']) / len(metrics['response_times'])
                    
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user")
        except Exception as e:
            print(f"Error during realtime evaluation: {e}")
        return metrics
        
    

class DOSEvaluator(BaseVGBenchEvaluator):
    """Evaluator class that coordinates between DOS emulators and LLMs."""
    
    def __init__(self, 
            max_steps: int = 1000, 
            step_delay: float = 0.1,
            metrics: Optional[List[Callable]] = None,
            checkpoints: Optional[List[str]] = None):
        super().__init__(max_steps, step_delay, metrics, checkpoints)
        
    async def run_episode(self, dos_agent: WebBrowsingAgent, url: str, task: str, server: DOSGameServer) -> Dict[str, Any]:
        """
        Run an episode of a game using JS-DOS with LLM interacting
        in realtime or paused based on if lite mode is on (controlled in agent).

        """
        try:
            # Start the agent
            await dos_agent.start(initial_url=url)
            
            # Execute the task
            await dos_agent.run_episode(task=task, checkpoints=self.checkpoints)
        finally:
            # Stop the agent
            await dos_agent.stop()
            
            # Stop the server if it was started
            if server:
                server.stop()