import numpy as np
import queue
from game import BoardState, GameSimulator, Rules
import logging

logging.basicConfig(level=logging.DEBUG)
mylog = logging.getLogger()

class Problem:
    """
    This is an interface which GameStateProblem implements.
    You will be using GameStateProblem in your code. Please see
    GameStateProblem for details on the format of the inputs and
    outputs.
    """

    def __init__(self, initial_state, goal_state_set: set):
        self.initial_state = initial_state
        self.goal_state_set = goal_state_set

    def get_actions(self, state):
        """
        Returns a set of valid actions that can be taken from this state
        """
        pass

    def execute(self, state, action):
        """
        Transitions from the state to the next state that results from taking the action
        """
        pass

    def is_goal(self, state):
        """
        Checks if the state is a goal state in the set of goal states
        """
        return state in self.goal_state_set

class GameStateProblem(Problem):

    def __init__(self, initial_board_state, goal_board_state, player_idx):
        """
        player_idx is 0 or 1, depending on which player will be first to move from this initial state.

        Inputs for this constructor:
            - initial_board_state: an instance of BoardState
            - goal_board_state: an instance of BoardState
            - player_idx: an element from {0, 1}

        How Problem.initial_state and Problem.goal_state_set are represented:
            - initial_state: ((game board state tuple), player_idx ) <--- indicates state of board and who's turn it is to move
              ---specifically it is of the form: tuple( ( tuple(initial_board_state.state), player_idx ) )

            - goal_state_set: set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))])
              ---in otherwords, the goal_state_set allows the goal_board_state.state to be reached on either player 0 or player 1's
              turn.
        """
        super().__init__(tuple((tuple(initial_board_state.state), player_idx)), set([tuple((tuple(goal_board_state.state), 0)), tuple((tuple(goal_board_state.state), 1))]))
        self.sim = GameSimulator(None)
        self.search_alg_fnc = None
        self.set_search_alg()

    def set_search_alg(self, alg=""):
        """
        If you decide to implement several search algorithms, and you wish to switch between them,
        pass a string as a parameter to alg, and then set:
            self.search_alg_fnc = self.your_method
        to indicate which algorithm you'd like to run.

        TODO: You need to set self.search_alg_fnc here
        """
        if alg =="a_star":
            self.search_alg_fnc = self.a_star_search
        elif alg == "bfs":
            self.search_alg_fnc = self.bfs_search
        else:
            self.search_alg_fnc = self.a_star_search

    def get_actions(self, state: tuple):
        """
        From the given state, provide the set possible actions that can be taken from the state

        Inputs: 
            state: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn

        Outputs:
            returns a set of actions
        """
        s, p = state
        np_state = np.array(s)
        self.sim.game_state.state = np_state
         

        return self.sim.generate_valid_actions(p)

    def execute(self, state: tuple, action: tuple):
        """
        From the given state, executes the given action

        The action is given with respect to the current player

        Inputs: 
            state: is a tuple (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
                and player_idx is the player that is moving this turn
            action: (relative_idx, position), where relative_idx is an index into the encoded_state
                with respect to the player_idx, and position is the encoded position where the indexed piece should move to.
        Outputs:
            the next state tuple that results from taking action in state
        """
        s, p = state
        k, v = action
        offset_idx = p * 6
        return tuple((tuple( s[i] if i != offset_idx + k else v for i in range(len(s))), (p + 1) % 2))

    ## TODO: Implement your search algorithm(s) here as methods of the GameStateProblem.
    ##       You are free to specify parameters that your method may require.
    ##       However, you must ensure that your method returns a list of (state, action) pairs, where
    ##       the first state and action in the list correspond to the initial state and action taken from
    ##       the initial state, and the last (s,a) pair has s as a goal state, and a=None, and the intermediate
    ##       (s,a) pairs correspond to the sequence of states and actions taken from the initial to goal state.
    ##
    ## NOTE: Here is an example of the format:
    ##       [(s1, a1),(s2, a2), (s3, a3), ..., (sN, aN)] where
    ##          sN is an element of self.goal_state_set
    ##          aN is None
    ##          All sK for K=1...N are in the form (encoded_state, player_idx), where encoded_state is a tuple of 12 integers,
    ##              effectively encoded_state is the result of tuple(BoardState.state)
    ##          All aK for K=1...N are in the form (int, int)
    ##
    ## NOTE: The format of state is a tuple: (encoded_state, player_idx), where encoded_state is a tuple of 12 integers
    ##       (mirroring the contents of BoardState.state), and player_idx is 0 or 1, indicating the player that is
    ##       moving in this state.
    ##       The format of action is a tuple: (relative_idx, position), where relative_idx the relative index into encoded_state
    ##       with respect to player_idx, and position is the encoded position where the piece should move to with this action.
    ## NOTE: self.get_actions will obtain the current actions available in current game state.
    ## NOTE: self.execute acts like the transition function.
    ## NOTE: Remember to set self.search_alg_fnc in set_search_alg above.
    ## 
    """ Here is an example:
    
    def my_snazzy_search_algorithm(self):
        ## Some kind of search algorithm
        ## ...
        return solution ## Solution is an ordered list of (s,a)
    """
    def bfs_search(self):
        start = self.initial_state        
        frontier = queue.Queue()
        frontier.put((start, []))
        visited = set()

        while not frontier.empty():
            current_state, path  = frontier.get()
            encoded_state, player_idx = current_state
            if current_state in self.goal_state_set:
                path = path + [(current_state, None)]
                return path

            visited.add(current_state)
            valid_actions = self.get_actions(current_state)
            for action in valid_actions:
                offset_idx = player_idx * 6 ## Either 0 or 6
                idx, pos = action
                encoded_state_list = list(encoded_state)
                encoded_state_list[idx + offset_idx] = pos
                next_encoded_state = tuple(encoded_state_list)
                next_player_idx = 1 - player_idx

                if (next_encoded_state, next_player_idx) not in visited:
                    new_path = path + [(current_state, action)]
                    frontier.put(((next_encoded_state, next_player_idx), new_path))
                    visited.add((next_encoded_state, next_player_idx))
        return []
    
    def heuristic(self, state):
        encoded_state, player_idx = state
        # Chebyshev distance of ball state
        encoded_goal_state, player_idx = list(self.goal_state_set)[player_idx]
        if player_idx == 1:
            return sum(1 for i, val in enumerate(encoded_state[6:11]) if not np.array_equal(val, encoded_goal_state[6:11][i]))
        else:
            return sum(1 for i, val in enumerate(encoded_state[0:5]) if not np.array_equal(val, encoded_goal_state[0:5][i]))
    
    def a_star_search(self):
        start = self.initial_state
        frontier = queue.PriorityQueue()
        mylog.critical("started getting heuristic")
        heuristic = self.heuristic(start)
        mylog.critical("done getting heuristic")
        mylog.critical("started putting into queue")
        frontier.put((0 + heuristic, 0, start, []), block=False, timeout=5.0)  # (priority, cost, state, path)
        mylog.critical("done putting into queue")
        visited = set()
        cost_so_far = {start: 0}

        while not frontier.empty():
            heuristic, cost, current_state, path = frontier.get()
            logging.critical("" + str(heuristic) + " " + str(cost) + " " + str(current_state) + " ")
            encoded_state, player_idx = current_state

            if current_state in self.goal_state_set:
                return path + [(current_state, None)]

            visited.add(current_state)
            valid_actions = self.get_actions(current_state)

            for action in valid_actions:
                offset_idx = player_idx * 6  # Either 0 or 6
                idx, pos = action
                encoded_state_list = list(encoded_state)
                encoded_state_list[idx + offset_idx] = pos
                next_encoded_state = tuple(encoded_state_list)
                next_player_idx = 1 - player_idx
                next_state = (next_encoded_state, next_player_idx)
                new_cost = cost + 1

                if next_state not in visited or new_cost < cost_so_far.get(next_state, float('inf')):
                    cost_so_far[next_state] = new_cost
                    priority = new_cost + self.heuristic(next_state)
                    new_path = path + [(current_state, action)]
                    frontier.put((priority, new_cost, next_state, new_path))
                    visited.add(next_state)

        return []
