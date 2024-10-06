import numpy as np

class BoardState:
    """
    Represents a state in the game
    """

    def __init__(self):
        """
        Initializes a fresh game state
        """
        self.N_ROWS = 8
        self.N_COLS = 7

        self.state = np.array([1,2,3,4,5,3,50,51,52,53,54,52])
        self.decode_state = [self.decode_single_pos(d) for d in self.state]

    def update(self, idx, val):
        """
        Updates both the encoded and decoded states
        """
        #Update ball position if ball piece got updated
        self.state[idx] = val
        self.decode_state[idx] = self.decode_single_pos(self.state[idx])

    def make_state(self):
        """
        Creates a new decoded state list from the existing state array
        """
        decoded_states = []
        for d in self.state:
            decoded_state = self.decode_single_pos(d)
            decoded_states.append(decoded_state)
        return decoded_states

    def encode_single_pos(self, cr: tuple):
        """
        Encodes a single coordinate (col, row) -> Z

        Input: a tuple (col, row)
        Output: an integer in the interval [0, 55] inclusive

        TODO: You need to implement this.
        """
        return cr[1] * (self.N_ROWS - 1) + cr[0]
     
    def decode_single_pos(self, n: int):
        """
        Decodes a single integer into a coordinate on the board: Z -> (col, row)

        Input: an integer in the interval [0, 55] inclusive
        Output: a tuple (col, row)

        TODO: You need to implement this.
        """
        return (n%(self.N_ROWS-1), n//(self.N_ROWS-1))

    def is_termination_state(self):
        """
        Checks if the current state is a termination state. Termination occurs when
        one of the player's move their ball to the opposite side of the board.

        You can assume that `self.state` contains the current state of the board, so
        check whether self.state represents a termainal board state, and return True or False.
        
        TODO: You need to implement this.
        """
        if(self.is_valid()):
            white_ball = self.state[5]
            black_ball = self.state[11]
            [white_col, white_row] = self.decode_single_pos(white_ball)
            [black_col, black_row] = self.decode_single_pos(black_ball)
            if (white_row == self.N_ROWS-1 and white_col in (0, self.N_COLS-1)) or (black_row == 0 and black_col in (0, self.N_COLS-1)):
                return True
        return False

    def is_valid(self):
        """
        Checks if a board configuration is valid. This function checks whether the current
        value self.state represents a valid board configuration or not. This encodes and checks
        the various constrainsts that must always be satisfied in any valid board state during a game.

        If we give you a self.state array of 12 arbitrary integers, this function should indicate whether
        it represents a valid board configuration.

        Output: return True (if valid) or False (if not valid)
        
        TODO: You need to implement this.
        """
        seen_positions = set()
        ball_positions = [self.state[5], self.state[11]]
        for idx, pos in enumerate(self.state):
            [col, row] = self.decode_single_pos(pos)
            if col < 0 or col > self.N_COLS-1 or row < 0 or row > self.N_ROWS-1:
                return False
            if(idx != 5 and idx != 11):
                if(pos in seen_positions):
                    return False
                seen_positions.add(pos)
        for ball_position in ball_positions:
            if(ball_position not in seen_positions):
                return False
        return True

class Rules:

    @staticmethod
    def single_piece_actions(board_state:BoardState, piece_idx):
        """
        Returns the set of possible actions for the given piece, assumed to be a valid piece located
        at piece_idx in the board_state.state.

        Inputs:
            - board_state, assumed to be a BoardState
            - piece_idx, assumed to be an index into board_state, identfying which piece we wish to
              enumerate the actions for.

        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that piece_idx can move to during this turn.
        
        TODO: You need to implement this.
        """
        curr_pos = board_state.state[piece_idx]
        [col, row] = board_state.decode_single_pos(curr_pos)
        pos_moves = set()
        block_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        for move in block_moves:
            pos_row = row + move[0]
            pos_col = col + move[1]
            if 0 <= pos_row < board_state.N_ROWS and 0 <= pos_col < board_state.N_COLS:
                idx = board_state.encode_single_pos([pos_col, pos_row])
                if(idx not in board_state.state):
                    pos_moves.add(idx)
        return pos_moves

    @staticmethod
    def single_ball_actions(board_state:BoardState, player_idx):
        """
        Returns the set of possible actions for moving the specified ball, assumed to be the
        valid ball for plater_idx  in the board_state

        Inputs:
            - board_state, assumed to be a BoardState
            - player_idx, either 0 or 1, to indicate which player's ball we are enumerating over
        
        Output: an iterable (set or list or tuple) of integers which indicate the encoded positions
            that player_idx's ball can move to during this turn.
        
        TODO: You need to implement this.
        """
        def is_oppponent_pos(pos):
            if(player_idx):
                if(pos in board_state.state[0:5]):
                    return True
            else:
                if(pos in board_state.state[6:11]):
                    return True
            return False
  
        def is_clear_path(start, end, direction):
            current = board_state.decode_single_pos(start)
            dest = board_state.decode_single_pos(end)
            while current != dest:
                current = (current[0] + direction[0], current[1] + direction[1])
                if(current == dest):
                    break
                if not (0 <= current[0] < board_state.N_COLS and 0 <= current[1] < board_state.N_ROWS):
                    return False
                if(is_oppponent_pos(board_state.encode_single_pos(current))):
                    return False
            return True
        
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # Horizontal and vertical
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
        ]

        possible_actions = set()
        start = board_state.state[11] if player_idx else board_state.state[5]
        possible_moves_to_explore = []
        if player_idx:
            possible_moves_to_explore = board_state.state[6:11]
        else:
            possible_moves_to_explore = board_state.state[0:5]
        
        def explore_in_paths(start, visited):
            for move in possible_moves_to_explore:
                if move in visited:
                    continue
                for direction in directions:
                    if(is_clear_path(start, move, direction)):
                        possible_actions.add(move)
                        visited.add(move)
                        explore_in_paths(move, visited)

        explore_in_paths(start, set([start]))
        return possible_actions

class GameSimulator:
    """
    Responsible for handling the game simulation
    """

    def __init__(self, players):
        self.game_state = BoardState()
        self.current_round = -1 ## The game starts on round 0; white's move on EVEN rounds; black's move on ODD rounds
        self.players = players

    def run(self):
        """
        Runs a game simulation
        """
        while not self.game_state.is_termination_state():
            ## Determine the round number, and the player who needs to move
            self.current_round += 1
            player_idx = self.current_round % 2
            ## For the player who needs to move, provide them with the current game state
            ## and then ask them to choose an action according to their policy
            action, value = self.players[player_idx].policy( self.game_state.make_state() )
            print(f"Round: {self.current_round} Player: {player_idx} State: {tuple(self.game_state.state)} Action: {action} Value: {value}")

            if not self.validate_action(action, player_idx):
                ## If an invalid action is provided, then the other player will be declared the winner
                if player_idx == 0:
                    return self.current_round, "BLACK", "White provided an invalid action"
                else:
                    return self.current_round, "WHITE", "Black probided an invalid action"

            ## Updates the game state
            self.update(action, player_idx)

        ## Player who moved last is the winner
        if player_idx == 0:
            return self.current_round, "WHITE", "No issues"
        else:
            return self.current_round, "BLACK", "No issues"

    def generate_valid_actions(self, player_idx: int):
        """
        Given a valid state, and a player's turn, generate the set of possible actions that player can take

        player_idx is either 0 or 1

        Input:
            - player_idx, which indicates the player that is moving this turn. This will help index into the
              current BoardState which is self.game_state
        Outputs:
            - a set of tuples (relative_idx, encoded position), each of which encodes an action. The set should include
              all possible actions that the player can take during this turn. relative_idx must be an
              integer on the interval [0, 5] inclusive. Given relative_idx and player_idx, the index for any
              piece in the boardstate can be obtained, so relative_idx is the index relative to current player's
              pieces. Pieces with relative index 0,1,2,3,4 are block pieces that like knights in chess, and
              relative index 5 is the player's ball piece.
            
        TODO: You need to implement this.
        """
        possible_actions = set()
        
        ball_actions = Rules.single_ball_actions(self.game_state, player_idx)
        for action in ball_actions:
            possible_actions.add((5, action))
        
        for idx in range(4):
            piece_idx = idx + 6 if player_idx else idx
            piece_actions = Rules.single_piece_actions(self.game_state, piece_idx)
            for action in piece_actions:
                possible_actions.add((idx, action))
        
        return possible_actions

    def validate_action(self, action: tuple, player_idx: int):
        """
        Checks whether or not the specified action can be taken from this state by the specified player

        Inputs:
            - action is a tuple (relative_idx, encoded position)
            - player_idx is an integer 0 or 1 representing the player that is moving this turn
            - self.game_state represents the current BoardState

        Output:
            - if the action is valid, return True
            - if the action is not valid, raise ValueError
        
        TODO: You need to implement this.
        """
        if not self.game_state.is_valid():
            raise ValueError("board state not valid")
        ball_action = True if action[0] == 5 else False
        if ball_action:
            if action[1] in Rules.single_ball_actions(self.game_state, player_idx):
                return True
            else:
                raise ValueError(f'ball action {action[1]} not possible for player: {player_idx}')
        else:
            piece_idx = action[0]
            if action[1] in Rules.single_piece_actions(self.game_state, piece_idx):
                return True
            else:
                raise ValueError(f'piece action {action[1]} not possible for player: {player_idx}')
    
    def update(self, action: tuple, player_idx: int):
        """
        Uses a validated action and updates the game board state
        """
        offset_idx = player_idx * 6 ## Either 0 or 6
        idx, pos = action
        self.game_state.update(offset_idx + idx, pos)
