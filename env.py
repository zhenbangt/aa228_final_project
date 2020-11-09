"""
Adapted from https://github.com/datamllab/rlcard/blob/master/rlcard/envs/env.py
and https://raw.githubusercontent.com/datamllab/rlcard/master/rlcard/envs/leducholdem.py
and https://github.com/datamllab/rlcard/blob/master/rlcard/games/leducholdem/game.py.
"""


from copy import copy
import json
import os

import numpy as np
import rlcard

from rlcard.games.leducholdem import Dealer
from rlcard.games.leducholdem import Player
from rlcard.games.leducholdem import Judger
from rlcard.games.leducholdem import Round
from rlcard.games.limitholdem import Game
from rlcard.utils import *


class LeducholdemGame(Game):
    def __init__(self, allow_step_back=False):
        """Initialize the class leducholdem Game"""
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        """ No big/small blind
        # Some configarations of the game
        # These arguments are fixed in Leduc Hold'em Game
        # Raise amount and allowed times
        self.raise_amount = 2
        self.allowed_raise_num = 2
        self.num_players = 2
        """
        # Some configarations of the game
        # These arguments can be specified for creating new games

        # Small blind and big blind
        self.small_blind = 1
        self.big_blind = 2 * self.small_blind

        # Raise amount and allowed times
        self.raise_amount = self.big_blind
        self.allowed_raise_num = 2

        self.num_players = 2

    def init_game(self):
        """Initialilze the game of Limit Texas Hold'em
        This version supports two-player limit texas hold'em
        Returns:
            (tuple): Tuple containing:
                (dict): The first state of the game
                (int): Current player's id
        """
        # Initilize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initilize two players to play the game
        self.players = [
            rlcard.games.leducholdem.Player(i, self.np_random)
            for i in range(self.num_players)
        ]

        # Initialize a judger class which will decide who wins in the end
        self.judger = Judger(self.np_random)

        # Prepare for the first round
        for i in range(self.num_players):
            self.dealer.shuffle()
            self.players[i].hand = self.dealer.deal_card()
        # Randomly choose a small blind and a big blind
        s = self.np_random.randint(0, self.num_players)
        b = (s + 1) % self.num_players
        self.players[b].in_chips = self.big_blind
        self.players[s].in_chips = self.small_blind
        self.public_card = None
        # The player with small blind plays the first
        self.game_pointer = s

        # Initilize a bidding round, in the first round, the big blind and the small blind needs to
        # be passed to the round for processing.
        self.round = Round(
            raise_amount=self.raise_amount,
            allowed_raise_num=self.allowed_raise_num,
            num_players=self.num_players,
            np_random=self.np_random,
        )

        self.round.start_new_round(
            game_pointer=self.game_pointer, raised=[p.in_chips for p in self.players]
        )

        # Count the round. There are 2 rounds in each game.
        self.round_counter = 0

        # Save the hisory for stepping back to the last state.
        self.history = []

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def step(self, action):
        """Get the next state
        Args:
            action (str): a specific action. (call, raise, fold, or check)
        Returns:
            (tuple): Tuple containing:
                (dict): next player's state
                (int): next plater's id
        """
        if self.allow_step_back:
            # First snapshot the current state
            r = copy(self.round)
            r_raised = copy(self.round.raised)
            gp = self.game_pointer
            r_c = self.round_counter
            d_deck = copy(self.dealer.deck)
            p = copy(self.public_card)
            ps = [copy(self.players[i]) for i in range(self.num_players)]
            ps_hand = [copy(self.players[i].hand) for i in range(self.num_players)]
            self.history.append((r, r_raised, gp, r_c, d_deck, p, ps, ps_hand))

        # Then we proceed to the next round
        self.game_pointer = self.round.proceed_round(self.players, action)

        # If a round is over, we deal more public cards
        if self.round.is_over():
            # For the first round, we deal 1 card as public card. Double the raise amount for the second round
            if self.round_counter == 0:
                self.dealer.shuffle()
                self.public_card = self.dealer.deal_card()
                self.round.raise_amount = 2 * self.raise_amount

            self.round_counter += 1
            self.round.start_new_round(self.game_pointer)

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def get_state(self, player):
        """Return player's state
        Args:
            player_id (int): player id
        Returns:
            (dict): The state of the player
        """
        chips = [self.players[i].in_chips for i in range(self.num_players)]
        legal_actions = self.get_legal_actions()
        state = self.players[player].get_state(self.public_card, chips, legal_actions)
        state["current_player"] = self.game_pointer

        return state

    def is_over(self):
        """Check if the game is over
        Returns:
            (boolean): True if the game is over
        """
        alive_players = [1 if p.status == "alive" else 0 for p in self.players]
        # If only one player is alive, the game is over.
        if sum(alive_players) == 1:
            return True

        # If all rounds are finshed
        if self.round_counter >= 2:
            return True
        return False

    def get_payoffs(self):
        """Return the payoffs of the game
        Returns:
            (list): Each entry corresponds to the payoff of one player
        """
        chips_payoffs = self.judger.judge_game(self.players, self.public_card)
        payoffs = np.array(chips_payoffs) / (self.big_blind)
        return payoffs

    def step_back(self):
        """Return to the previous state of the game
        Returns:
            (bool): True if the game steps back successfully
        """
        if len(self.history) > 0:
            (
                self.round,
                r_raised,
                self.game_pointer,
                self.round_counter,
                d_deck,
                self.public_card,
                self.players,
                ps_hand,
            ) = self.history.pop()
            self.round.raised = r_raised
            self.dealer.deck = d_deck
            for i, hand in enumerate(ps_hand):
                self.players[i].hand = hand
            return True
        return False


class Env:
    """
    The base Env class. For all the environments in RLCard,
    we should base on this class and implement as many functions
    as we can.
    """

    def __init__(self, config):
        """Initialize the environment

        Args:
            config (dict): A config dictionary. All the fields are
                optional. Currently, the dictionary includes:
                'seed' (int) - A environment local random seed.
                'env_num' (int) - If env_num>1, the environemnt wil be run
                  with multiple processes. Note the implementatino is
                  in `vec_env.py`.
                'allow_step_back' (boolean) - True if allowing
                 step_back.
                'allow_raw_data' (boolean) - True if allow
                 raw obs in state['raw_obs'] and raw legal actions in
                 state['raw_legal_actions'].
                'single_agent_mode' (boolean) - True if single agent mode,
                 i.e., the other players are pretrained models.
                'active_player' (int) - If 'singe_agent_mode' is True,
                 'active_player' specifies the player that does not use
                  pretrained models.
                There can be some game specific configurations, e.g., the
                number of players in the game. These fields should start with
                'game_', e.g., 'game_player_num' we specify the number of
                players in the game. Since these configurations may be game-specific,
                The default settings shpuld be put in the Env class. For example,
                the default game configurations for Blackjack should be in
                'rlcard/envs/blackjack.py'
                TODO: Support more game configurations in the future.
        """
        self.allow_step_back = self.game.allow_step_back = config.get(
            "allow_step_back", False
        )
        self.allow_raw_data = config.get("allow_raw_data", False)
        self.record_action = config.get("record_action", False)
        if self.record_action:
            self.action_recorder = []

        # Game specific configurations
        # Currently only support blackjack
        # TODO support game configurations for all the games
        supported_envs = ["blackjack"]
        if self.name in supported_envs:
            _game_config = self.default_game_config.copy()
            for key in config:
                if key in _game_config:
                    _game_config[key] = config[key]
            self.game.configure(_game_config)

        # Get the number of players/actions in this game
        self.player_num = self.game.get_player_num()
        self.action_num = self.game.get_action_num()

        # A counter for the timesteps
        self.timestep = 0

        # Modes
        self.single_agent_mode = config.get("single_agent_mode", False)
        self.active_player = config.get("active_player", 0)

        # Load pre-trained models if single_agent_mode=True
        if self.single_agent_mode:
            self.model = self._load_model()
            # If at least one pre-trained agent needs raw data, we set self.allow_raw_data = True
            for agent in self.model.agents:
                if agent.use_raw:
                    self.allow_raw_data = True
                    break

        # Set random seed, default is None
        self._seed(config.get("seed", None))

    def reset(self):
        """
        Reset environment in single-agent mode
        Call `_init_game` if not in single agent mode
        """
        if not self.single_agent_mode:
            return self._init_game()

        while True:
            state, player_id = self.game.init_game()
            while not player_id == self.active_player:
                self.timestep += 1
                action, _ = self.model.agents[player_id].eval_step(
                    self._extract_state(state)
                )
                if not self.model.agents[player_id].use_raw:
                    action = self._decode_action(action)
                state, player_id = self.game.step(action)

            if not self.game.is_over():
                break

        return self._extract_state(state)

    def step(self, action, raw_action=False):
        """Step forward

        Args:
            action (int): The action taken by the current player
            raw_action (boolean): True if the action is a raw action

        Returns:
            (tuple): Tuple containing:

                (dict): The next state
                (int): The ID of the next player
        """
        if not raw_action:
            action = self._decode_action(action)
        if self.single_agent_mode:
            return self._single_agent_step(action)

        self.timestep += 1
        # Record the action for human interface
        if self.record_action:
            self.action_recorder.append([self.get_player_id(), action])
        next_state, player_id = self.game.step(action)

        return self._extract_state(next_state), player_id

    def step_back(self):
        """Take one step backward.

        Returns:
            (tuple): Tuple containing:

                (dict): The previous state
                (int): The ID of the previous player

        Note: Error will be raised if step back from the root node.
        """
        if not self.allow_step_back:
            raise Exception(
                "Step back is off. To use step_back, please set allow_step_back=True in rlcard.make"
            )

        if not self.game.step_back():
            return False

        player_id = self.get_player_id()
        state = self.get_state(player_id)

        if self.record_action:
            self.action_recorder.pop()

        return state, player_id

    def set_agents(self, agents):
        """
        Set the agents that will interact with the environment.
        This function must be called before `run`.

        Args:
            agents (list): List of Agent classes
        """
        if self.single_agent_mode:
            raise ValueError(
                "Setting agent in single agent mode or human mode is not allowed."
            )

        self.agents = agents
        # If at least one agent needs raw data, we set self.allow_raw_data = True
        for agent in self.agents:
            if agent.use_raw:
                self.allow_raw_data = True
                break

    def run(self, is_training=False):
        """
        Run a complete game, either for evaluation or training RL agent.

        Args:
            is_training (boolean): True if for training purpose.

        Returns:
            (tuple) Tuple containing:

                (list): A list of trajectories generated from the environment.
                (list): A list payoffs. Each entry corresponds to one player.

        Note: The trajectories are 3-dimension list. The first dimension is for different players.
              The second dimension is for different transitions. The third dimension is for the contents of each transiton
        """
        if self.single_agent_mode:
            raise ValueError("Run in single agent not allowed.")

        trajectories = [[] for _ in range(self.player_num)]
        state, player_id = self.reset()

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.is_over():
            # Agent plays
            if not is_training:
                action, _ = self.agents[player_id].eval_step(state)
            else:
                action = self.agents[player_id].step(state)

            # Environment steps
            next_state, next_player_id = self.step(
                action, self.agents[player_id].use_raw
            )
            # Save action
            trajectories[player_id].append(action)

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.player_num):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        payoffs = self.get_payoffs()

        # Reorganize the trajectories
        trajectories = reorganize(trajectories, payoffs)

        return trajectories, payoffs

    def is_over(self):
        """Check whether the curent game is over

        Returns:
            (boolean): True if current game is over
        """
        return self.game.is_over()

    def get_player_id(self):
        """Get the current player id

        Returns:
            (int): The id of the current player
        """
        return self.game.get_player_id()

    def get_state(self, player_id):
        """Get the state given player id

        Args:
            player_id (int): The player id

        Returns:
            (numpy.array): The observed state of the player
        """
        return self._extract_state(self.game.get_state(player_id))

    def get_payoffs(self):
        """Get the payoffs of players. Must be implemented in the child class.

        Returns:
            (list): A list of payoffs for each player.

        Note: Must be implemented in the child class.
        """
        raise NotImplementedError

    def get_perfect_information(self):
        """Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state

        Note: Must be implemented in the child class.
        """
        raise NotImplementedError

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.game.np_random = self.np_random
        return seed

    def _init_game(self):
        """Start a new game

        Returns:
            (tuple): Tuple containing:

                (numpy.array): The begining state of the game
                (int): The begining player
        """
        state, player_id = self.game.init_game()
        if self.record_action:
            self.action_recorder = []
        return self._extract_state(state), player_id

    def _load_model(self):
        """Load pretrained/rule model

        Returns:
            model (Model): A Model object
        """
        raise NotImplementedError

    def _extract_state(self, state):
        """Extract useful information from state for RL. Must be implemented in the child class.

        Args:
            state (dict): The raw state

        Returns:
            (numpy.array): The extracted state
        """
        raise NotImplementedError

    def _decode_action(self, action_id):
        """Decode Action id to the action in the game.

        Args:
            action_id (int): The id of the action

        Returns:
            (string): The action that will be passed to the game engine.

        Note: Must be implemented in the child class.
        """
        raise NotImplementedError

    def _get_legal_actions(self):
        """Get all legal actions for current state.

        Returns:
            (list): A list of legal actions' id.

        Note: Must be implemented in the child class.
        """
        raise NotImplementedError

    def _single_agent_step(self, action):
        """Step forward for human/single agent

        Args:
            action (int): The action takem by the current player

        Returns:
            next_state (numpy.array): The next state
        """
        reward = 0.0
        done = False
        self.timestep += 1
        state, player_id = self.game.step(action)
        while not self.game.is_over() and not player_id == self.active_player:
            self.timestep += 1
            action, _ = self.model.agents[player_id].eval_step(
                self._extract_state(state)
            )
            if not self.model.agents[player_id].use_raw:
                action = self._decode_action(action)
            state, player_id = self.game.step(action)

        if self.game.is_over():
            reward = self.get_payoffs()[self.active_player]
            done = True
            state = self.reset()
            return state, reward, done

        return self._extract_state(state), reward, done

    @staticmethod
    def init_game():
        """(This function has been replaced by `reset()`)"""
        raise ValueError("init_game is removed. Please use env.reset()")


class LeducholdemEnv(Env):
    """Leduc Hold'em Environment"""

    def __init__(self, config):
        """Initialize the Limitholdem environment"""
        self.name = "leduc-holdem"
        self.game = LeducholdemGame()
        super().__init__(config)
        self.actions = ["call", "raise", "fold", "check"]
        self.state_shape = [36]

        with open(
            os.path.join(rlcard.__path__[0], "games/leducholdem/card2index.json"), "r"
        ) as file:
            self.card2index = json.load(file)

    def _load_model(self):
        """Load pretrained/rule model

        Returns:
            model (Model): A Model object
        """
        from rlcard import models

        return models.load("leduc-holdem-cfr")

    def _get_legal_actions(self):
        """Get all leagal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        """
        return self.game.get_legal_actions()

    def _extract_state(self, state):
        """Extract the state representation from state dictionary for agent

        Note: Currently the use the hand cards and the public cards. TODO: encode the states

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        """
        extracted_state = {}

        legal_actions = [self.actions.index(a) for a in state["legal_actions"]]
        extracted_state["legal_actions"] = legal_actions

        public_card = state["public_card"]
        hand = state["hand"]
        obs = np.zeros(36)
        obs[self.card2index[hand]] = 1
        if public_card:
            obs[self.card2index[public_card] + 3] = 1
        obs[state["my_chips"] + 6] = 1
        obs[state["all_chips"][1] + 20] = 1
        extracted_state["obs"] = obs

        if self.allow_raw_data:
            extracted_state["raw_obs"] = state
            extracted_state["raw_legal_actions"] = [a for a in state["legal_actions"]]
        if self.record_action:
            extracted_state["action_record"] = self.action_recorder

        return extracted_state

    def get_payoffs(self):
        """Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        """
        return self.game.get_payoffs()

    def _decode_action(self, action_id):
        """Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        """
        legal_actions = self.game.get_legal_actions()
        if self.actions[action_id] not in legal_actions:
            if "check" in legal_actions:
                return "check"
            else:
                return "fold"
        return self.actions[action_id]

    def get_perfect_information(self):
        """Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        """
        state = {}
        state["chips"] = [self.game.players[i].in_chips for i in range(self.player_num)]
        state["public_card"] = (
            self.game.public_card.get_index() if self.game.public_card else None
        )
        state["hand_cards"] = [
            self.game.players[i].hand.get_index() for i in range(self.player_num)
        ]
        state["current_round"] = self.game.round_counter
        state["current_player"] = self.game.game_pointer
        state["legal_actions"] = self.game.get_legal_actions()
        return state
