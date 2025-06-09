import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.env_logger import EnvLogger

MAX_ACCELERATION = 1.0  # Max acceleration in x or y
MAX_VELOCITY = 2.0  # Max velocity in x or y
ARENA_WIDTH = 20.0
ARENA_HEIGHT = 30.0
TIME_STEP = 0.1  # Time step duration
PROXIMITY_THRESHOLD = 0.5  # Collision / too close threshold
EPISODE_LENGTH = 300  # Max frames per episode

# --- Route Definition (Slant Route) ---
# Offensive agent starts near one sideline, runs a slant towards midfield
SLANT_ROUTE_START = np.array([2.0, 5.0])
# Target end zone is y=ARENA_HEIGHT, slant aims for midfield
SLANT_ROUTE_WAYPOINT_1 = np.array([ARENA_WIDTH / 2, ARENA_HEIGHT / 2])
SLANT_ROUTE_END = np.array(
    [ARENA_WIDTH - 2.0, ARENA_HEIGHT - 5.0]
)  # A general target area


class ManCoverageEnv(AECEnv):
    metadata = {
        "render_modes": ["human"],
        "name": "ManCoverage-v0",
        "is_parallelizable": False,  # Standard for AECEnv
        "render_fps": 30,
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.agents = ["offensive_agent_0", "defensive_agent_0"]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)

        # Agent states: [x, y, vx, vy]
        self.agent_states = {
            agent: np.zeros(4, dtype=np.float32) for agent in self.agents
        }
        self.offensive_agent_target_point = SLANT_ROUTE_WAYPOINT_1.astype(
            np.float32
        ).copy()  # Current target on route

        # Action space: [ax, ay] for each agent
        self.action_spaces = {
            agent: spaces.Box(
                low=-MAX_ACCELERATION,
                high=MAX_ACCELERATION,
                shape=(2,),
                dtype=np.float32,
            )
            for agent in self.agents
        }

        # Observation space:
        # For offensive agent: [my_x, my_y, my_vx, my_vy, opponent_x, opponent_y, target_x, target_y]
        # For defensive agent: [my_x, my_y, my_vx, my_vy, opponent_x, opponent_y]
        offensive_low = np.array(
            [
                0,
                0,
                -MAX_VELOCITY,
                -MAX_VELOCITY,
                0,
                0,
                0,
                0,
            ],
            dtype=np.float32,
        )
        offensive_high = np.array(
            [
                ARENA_WIDTH,
                ARENA_HEIGHT,
                MAX_VELOCITY,
                MAX_VELOCITY,
                ARENA_WIDTH,
                ARENA_HEIGHT,
                ARENA_WIDTH,
                ARENA_HEIGHT,
            ],
            dtype=np.float32,
        )

        defensive_low = np.array(
            [0, 0, -MAX_VELOCITY, -MAX_VELOCITY, 0, 0], dtype=np.float32
        )
        defensive_high = np.array(
            [
                ARENA_WIDTH,
                ARENA_HEIGHT,
                MAX_VELOCITY,
                MAX_VELOCITY,
                ARENA_WIDTH,
                ARENA_HEIGHT,
            ],
            dtype=np.float32,
        )

        self.observation_spaces = {
            "offensive_agent_0": spaces.Box(
                low=offensive_low, high=offensive_high, dtype=np.float32
            ),
            "defensive_agent_0": spaces.Box(
                low=defensive_low, high=defensive_high, dtype=np.float32
            ),
        }

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.agent_selection = ""  # Set in reset
        self.frames = 0

        if self.render_mode == "human":
            pygame.init()  # type: ignore
            self.screen_width = 600
            self.screen_height = int(self.screen_width * (ARENA_HEIGHT / ARENA_WIDTH))
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Man Coverage Env")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)

    def observe(self, agent):
        my_state = self.agent_states[agent]
        opponent_id = [a for a in self.possible_agents if a != agent][0]
        opponent_state = self.agent_states[opponent_id]

        if agent == "offensive_agent_0":
            obs = np.concatenate(
                [
                    my_state,  # my_x, my_y, my_vx, my_vy
                    opponent_state[:2],  # opponent_x, opponent_y
                    self.offensive_agent_target_point,  # target_x, target_y for route
                ]
            ).astype(np.float32)
        else:  # defensive_agent_0
            obs = np.concatenate(
                [
                    my_state,  # my_x, my_y, my_vx, my_vy
                    opponent_state[:2],  # opponent_x, opponent_y
                ]
            ).astype(np.float32)
        return obs

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def _update_physics(self, agent, action):
        accel = np.clip(action, -MAX_ACCELERATION, MAX_ACCELERATION).astype(np.float32)

        self.agent_states[agent][2:] += (
            accel * TIME_STEP
        )  # Update velocity: v_new = v_old + a * dt
        self.agent_states[agent][2:] = np.clip(
            self.agent_states[agent][2:], -MAX_VELOCITY, MAX_VELOCITY
        )  # Clip velocity

        self.agent_states[agent][:2] += (
            self.agent_states[agent][2:] * TIME_STEP
        )  # Update position: x_new = x_old + v * dt

        # Keep agents within arena boundaries
        self.agent_states[agent][0] = np.clip(
            self.agent_states[agent][0], 0, ARENA_WIDTH
        )
        self.agent_states[agent][1] = np.clip(
            self.agent_states[agent][1], 0, ARENA_HEIGHT
        )

        # If agent hits boundary, zero out velocity in that direction
        if (
            self.agent_states[agent][0] == 0
            or self.agent_states[agent][0] == ARENA_WIDTH
        ):
            self.agent_states[agent][2] = 0  # vx = 0
        if (
            self.agent_states[agent][1] == 0
            or self.agent_states[agent][1] == ARENA_HEIGHT
        ):
            self.agent_states[agent][3] = 0  # vy = 0

    def _calculate_rewards_and_done(self):
        off_agent = "offensive_agent_0"
        def_agent = "defensive_agent_0"

        off_pos = self.agent_states[off_agent][:2]
        def_pos = self.agent_states[def_agent][:2]

        distance_to_target = np.linalg.norm(off_pos - self.offensive_agent_target_point)
        distance_between_agents = np.linalg.norm(off_pos - def_pos)

        # --- Offensive Agent Reward & Done ---
        # 1. Reward for following the route.
        reward_route_follow = (
            -distance_to_target * 0.1
        )  # Small penalty for distance to current target.

        # Check if offensive agent reached its current target point, which ends the episode.
        if distance_to_target < 1.0:  # Threshold for reaching waypoint
            if np.array_equal(
                self.offensive_agent_target_point, SLANT_ROUTE_WAYPOINT_1
            ):
                self.offensive_agent_target_point = SLANT_ROUTE_END.astype(
                    np.float32
                ).copy()
                reward_route_follow += 10.0  # Reward for reaching first waypoint
            elif np.array_equal(self.offensive_agent_target_point, SLANT_ROUTE_END):
                reward_route_follow += 20.0  # Major reward for reaching final target
                self.terminations[off_agent] = True  # Task accomplished
                self.terminations[def_agent] = True  # End for defender too

        # 2. Reward for maintaining distance from the defender.
        reward_distance = (
            distance_between_agents * 0.05
        )  # Continuous reward for being further away

        self.rewards[off_agent] = float(reward_route_follow + reward_distance)

        # --- Defensive Agent Reward & Done ---
        # The defensive agent's reward is based purely on staying close to the offensive agent.
        # A negative distance serves as a penalty for being far away.
        def_reward = -distance_between_agents * 0.2

        self.rewards[def_agent] = float(def_reward)

    def reset(self, seed=None, options=None):
        # Standard PettingZoo reset logic
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        if seed is not None:
            # self._seed(seed) # PettingZoo expects _seed for reproducibility
            np.random.seed(seed)  # Use numpy's global random seed if a seed is given

        self.frames = 0

        # Dictionaries are re-initialized for the new episode
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Initial positions and velocities
        self.agent_states["offensive_agent_0"] = np.concatenate(
            [SLANT_ROUTE_START, [0.0, 0.0]]
        ).astype(np.float32)
        # Defensive agent starts near the offensive agent, but not too close
        def_start_x = SLANT_ROUTE_START[0] + np.random.uniform(-2, 2)
        def_start_y = SLANT_ROUTE_START[1] + np.random.uniform(1, 3)  # Slightly behind
        self.agent_states["defensive_agent_0"] = np.array(
            [def_start_x, def_start_y, 0.0, 0.0], dtype=np.float32
        )
        self.agent_states["defensive_agent_0"][0] = np.clip(
            self.agent_states["defensive_agent_0"][0], 0, ARENA_WIDTH
        )
        self.agent_states["defensive_agent_0"][1] = np.clip(
            self.agent_states["defensive_agent_0"][1], 0, ARENA_HEIGHT
        )

        self.offensive_agent_target_point = SLANT_ROUTE_WAYPOINT_1.astype(
            np.float32
        ).copy()

        # PettingZoo's reset() expects to return (observation, info) for the first agent
        # However, it's also common to just prepare the state and let the loop call .last()
        # For compatibility with some wrappers/training loops, we ensure all agents get an initial observation stored.
        # The training loop will typically call .last() or iterate via agent_iter()

        # No explicit return needed for reset by PettingZoo standard, but observations should be ready
        return {agent: self.observe(agent) for agent in self.agents}, {
            agent: self.infos[agent] for agent in self.agents
        }

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # If agent is done, pass a None action to PettingZoo's AECEnv.action_for_done_agent
            # and then return, as per PettingZoo docs for handling done agents.
            self._was_dead_step(action)
            return

        current_agent = self.agent_selection

        # Ensure action is not None before processing
        if (
            action is None
        ):  # Can happen if agent was already done from a previous multi-agent step logic
            action = self.action_spaces[
                current_agent
            ].sample()  # Or a default no-op action

        self._update_physics(current_agent, action)

        # Accumulate rewards and check termination/truncation for the current agent
        # In AEC, rewards are from the perspective of the current agent, after its action.

        if self._agent_selector.is_last():
            # All agents have taken their turn in this "frame"
            self._calculate_rewards_and_done()  # Calculate rewards and dones for all agents based on the new state

            self.frames += 1
            if self.frames >= EPISODE_LENGTH:
                self.truncations = {agent: True for agent in self.agents}
                self.infos[current_agent]["status"] = "episode_truncated"
        else:
            self._clear_rewards()  # Clears rewards for all agents before recalculation by _calculate_rewards_and_done if it's last agent.

        # For the current agent, assign its specific reward and done status from the global calculation
        # This is tricky in AEC if rewards are interdependent and calculated once per "frame".
        # The reward attributed to current_agent should be the one calculated for it in _calculate_rewards_and_done.

        # After _calculate_rewards_and_done (if it was the last agent), self.rewards, self.terminations are updated for everyone.
        # We fetch the specific values for the current_agent.
        self._cumulative_rewards[self.agent_selection] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()

        # Select next agent
        self.agent_selection = self._agent_selector.next()

    def render(self):
        if self.render_mode != "human":
            return []  # Return empty list if not rendering to comply with some wrappers

        self.screen.fill((20, 20, 20))  # Dark background

        scale_x = self.screen_width / ARENA_WIDTH
        scale_y = self.screen_height / ARENA_HEIGHT

        # Draw Arena Boundaries (optional, for clarity)
        pygame.draw.rect(
            self.screen, (50, 50, 50), (0, 0, self.screen_width, self.screen_height), 1
        )

        # Draw Agents
        off_color = (100, 150, 255)  # Light Blue for Offensive
        def_color = (255, 100, 100)  # Light Red for Defensive
        agent_radius = int(0.4 * min(scale_x, scale_y))  # Adjusted radius

        # Offensive Agent
        off_pos = self.agent_states["offensive_agent_0"][:2]
        pygame.draw.circle(
            self.screen,
            off_color,
            (int(off_pos[0] * scale_x), int(off_pos[1] * scale_y)),
            agent_radius,
        )
        pygame.draw.circle(
            self.screen,
            (200, 200, 255),  # Lighter inner circle
            (int(off_pos[0] * scale_x), int(off_pos[1] * scale_y)),
            int(agent_radius * 0.6),
        )

        # Defensive Agent
        def_pos = self.agent_states["defensive_agent_0"][:2]
        pygame.draw.circle(
            self.screen,
            def_color,
            (int(def_pos[0] * scale_x), int(def_pos[1] * scale_y)),
            agent_radius,
        )
        pygame.draw.circle(
            self.screen,
            (255, 200, 200),  # Lighter inner circle
            (int(def_pos[0] * scale_x), int(def_pos[1] * scale_y)),
            int(agent_radius * 0.6),
        )

        # Draw Slant Route Waypoints for visualization
        route_color = (0, 200, 0)  # Green
        pygame.draw.circle(
            self.screen,
            route_color,
            (int(SLANT_ROUTE_START[0] * scale_x), int(SLANT_ROUTE_START[1] * scale_y)),
            5,
        )
        pygame.draw.circle(
            self.screen,
            route_color,
            (
                int(SLANT_ROUTE_WAYPOINT_1[0] * scale_x),
                int(SLANT_ROUTE_WAYPOINT_1[1] * scale_y),
            ),
            5,
        )
        pygame.draw.circle(
            self.screen,
            route_color,
            (int(SLANT_ROUTE_END[0] * scale_x), int(SLANT_ROUTE_END[1] * scale_y)),
            5,
        )

        # Line to current offensive target
        pygame.draw.line(
            self.screen,
            (100, 255, 100, 100),  # semi-transparent green
            (int(off_pos[0] * scale_x), int(off_pos[1] * scale_y)),
            (
                int(self.offensive_agent_target_point[0] * scale_x),
                int(self.offensive_agent_target_point[1] * scale_y),
            ),
            1,
        )

        # Display basic info
        y_offset = 10
        info_texts = [
            f"Frame: {self.frames}/{EPISODE_LENGTH}",
            f"Off Agent: ({off_pos[0]:.1f}, {off_pos[1]:.1f}) R: {self._cumulative_rewards['offensive_agent_0']:.2f}",
            f"Def Agent: ({def_pos[0]:.1f}, {def_pos[1]:.1f}) R: {self._cumulative_rewards['defensive_agent_0']:.2f}",
            f"Current Agent: {self.agent_selection}",
        ]
        for text_line in info_texts:
            text_surface = self.font.render(text_line, True, (200, 200, 200))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 20

        if any(self.terminations.values()) or any(self.truncations.values()):
            end_text = "EPISODE ENDED"
            if self.terminations["offensive_agent_0"]:
                end_text = "OFFENSE SCORED!"
            elif any(self.truncations.values()):
                end_text = "EPISODE TRUNCATED (MAX STEPS)"

            end_surface = self.font.render(end_text, True, (255, 255, 0))
            text_rect = end_surface.get_rect(
                center=(self.screen_width / 2, self.screen_height / 2)
            )
            self.screen.blit(end_surface, text_rect)

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        return [
            pygame.surfarray.array3d(self.screen)
        ]  # For gymnasium compatibility if needed

    def close(self):
        if self.render_mode == "human":
            pygame.quit()  # type: ignore


# --- PettingZoo API Compliance Helpers & Manual Test ---
def env(render_mode=None):
    internal_env = ManCoverageEnv(render_mode=render_mode)
    # Apply PettingZoo wrappers for API conformance
    internal_env = wrappers.OrderEnforcingWrapper(internal_env)
    return internal_env


if __name__ == "__main__":
    # Manual test
    # pemerge-next-line
    # env_instance = ManCoverageEnv(render_mode="human") # Use direct class for easier debugging
    env_instance = env(render_mode="human")  # Use wrapped env for API testing

    # Test reset
    # observations, infos = env_instance.reset() # PettingZoo reset usually doesn't return obs/info directly for all agents like Gym
    env_instance.reset()

    # print(f"Initial observations: {observations}") # This might be empty depending on wrapper

    max_cycles = 5000
    for step_num in range(max_cycles):
        # Ensure there are agents to iterate over. If env_instance.agents is empty, it means they all terminated.
        if not env_instance.agents:
            print(
                f"All agents are done. Resetting environment. Current step_num: {step_num}"
            )
            env_instance.reset()
            if not env_instance.agents:  # Should be repopulated by reset
                print("Error: env_instance.agents still empty after reset.")
                break
            # After reset, the agent_iter needs to be called on a valid environment state

        for (
            agent_id
        ) in env_instance.agent_iter():  # Use agent_iter for proper agent cycling
            observation, reward, termination, truncation, info = env_instance.last()

            if termination or truncation:
                action = None  # Agent is done, PettingZoo expects None action
            else:
                action = env_instance.action_space(
                    agent_id
                ).sample()  # Sample random action

            env_instance.step(action)

        # Check for global done condition to reset
        # Needs careful handling in AEC as agents terminate/truncate individually.
        # If all agents are done, then reset.
        if (
            not env_instance.agents
        ):  # This means all agents are done and removed from self.agents by AECEnv
            print(f"All agents done at outer loop check step {step_num}. Resetting.")
            env_instance.reset()
            # observations, infos = env_instance.reset()
            # print(f"Reset observations: {observations}")

        # Handle Pygame events for closing window
        if env_instance.render_mode == "human":
            quit_pygame = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_pygame = True
            if quit_pygame:
                env_instance.close()
                break  # Break from outer simulation loop

        if not env_instance.agents:  # if reset happened because all agents were done
            print("Restarting outer loop after reset and Pygame event check.")
            continue

    env_instance.close()
