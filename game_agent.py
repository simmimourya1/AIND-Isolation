"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def minimize_opponent_hits(game, player):
    center_position = (0,0);
    left_blank_spaces = len(game.get_blank_spaces())
    player_moves_left = len(game.get_legal_moves(player))
    opponent_moves_left = len(game.get_legal_moves(game.get_opponent(player)))

    #maximize those blank spaces which actually aren't contributing for opponent moves.
    not_my_moves = left_blank_spaces - player_moves_left
    if opponent_moves_left >= 0.5 *not_my_moves:
        return float(player_moves_left- 2 * opponent_moves_left)
    elif opponent_moves_left >= 0.3 *not_my_moves:
        return float(player_moves_left - 1.2* opponent_moves_left)
    else:
        return float(player_moves_left - opponent_moves_left)



def simplest_aggressive(game, player):
    # not submitted
    # If won
    if game.is_winner(player):
        return float("inf")

    # If lost
    if game.is_loser(player):
        return float("-inf")

    player_moves_left = len(game.get_legal_moves(player))
    opponent_moves_left = len(game.get_legal_moves(game.get_opponent(player)))

    return float(player_moves_left - 2*opponent_moves_left)


def maximize_winning(game, player):

    # not submitted
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = 1.0 * len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if my_moves == 0:
        return float("-inf")

    if opponent_moves == 0:
        return float("inf")

    return my_moves/opponent_moves


def minimize_losing(game, player):
    # not submitted 
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = len(game.get_legal_moves(player))
    opponent_moves = 1.0 * len(game.get_legal_moves(game.get_opponent(player)))

    if my_moves == 0:
        return float("-inf")

    if opponent_moves == 0:
        return float("inf")

    return -opponent_moves/my_moves


def weighted_combination(game, player):
    # This heuristic is a weighted combination of maximize_winning and minimize_losing heuristics.
    # submitted 
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    my_moves = len(game.get_legal_moves(player))
    opponent_moves = len(game.get_legal_moves(game.get_opponent(player)))

    return my_moves*my_moves - 1.5*opponent_moves*opponent_moves

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return weighted_combination(game, player)




class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if len(game.get_legal_moves(self)) == 0:
            return (-1,-1)

        if self.search_depth <= 0:
            self.iterative = True

        best_move = (None,(-1,-1))

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                for depth in range(1,game.width * game.height):
                    best_move = self.minimax(game, depth) if self.method == 'minimax' \
                        else self.alphabeta(game,depth)
                    if best_move[0] == float("inf"):
                        break
            else:
                best_move = self.minimax(game, self.search_depth) if self.method == 'minimax' \
                    else self.alphabeta(game, self.search_depth)

        except Timeout:
            return best_move[1]

        # Return the best move from the last completed search iteration
        return best_move[1]


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
            
        # Get legal moves for active player
        legal_moves = game.get_legal_moves()
        
        # Search depth reaches terminal case
        if depth == 0:
            # Score
            return self.score(game, self), (-1, -1)

        # If no legal moves are left, game ends
        if not legal_moves:
            # -inf or +inf 
            return game.utility(self), (-1, -1)

        best_move = None

        if maximizing_player:
            best_score = float("-inf")
            # check for every legal move.
            for move in legal_moves:
                next_state = game.forecast_move(move)
                score, _ = self.minimax(next_state, depth - 1, False)
                # Best for maximizing player is highest score
                if score > best_score:
                    best_score, best_move = score, move
        else:
            best_score = float("inf")
            for move in legal_moves:
                next_state = game.forecast_move(move)
                score, _ = self.minimax(next_state, depth - 1, True)
                # Best for minimizing player is lowest score
                if score < best_score:
                    best_score, best_move = score, move
        return best_score, best_move


        # Minimax for a given game state returns all the valid moves. It then stimulates
        # all the valid moves on copies of game state and evaluates each game state and 
        # returns the best move.

        # def min_play(game_state):
        #     return(min(map(lambda move: max_play(game_state.forecast_move(move).score()), game.get_legal_moves(player))))


        # def max_play(game_state):
        #     return(max(map(lambda move: min_play(game_state.forecast_move(move).score()), game.get_legal_moves(player))))
    
            
        # return max(map(lambda move: (move, min_play(game_state.forecast_move(move).score())), game.get_legal_moves(player)), key = lambda x: x[1])


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
            
        # Get legal moves for active player
        legal_moves = game.get_legal_moves()
        
        # Search depth reaches terminal case
        if depth == 0:
            # Score
            return self.score(game, self), (-1, -1)

        # If no legal moves are left, game ends
        if not legal_moves:
            # -inf or +inf 
            return game.utility(self), (-1, -1)


        best_move = None
        if maximizing_player:
            # Maximizing player wants highest score.
            best_score = float("-inf")
            for move in legal_moves:
                next_state = game.forecast_move(move)
                # Forecast_move switches the active player
                score, _ = self.alphabeta(next_state, depth - 1, alpha, beta, False)
                if score > best_score:
                    best_score, best_move = score, move
                # Prune
                if best_score >= beta:
                    return best_score, best_move
                # Update alpha
                alpha = max(alpha, best_score)
        else:
            # Minimizing player wants lowest score.
            best_score = float("inf")
            for move in legal_moves:
                next_state = game.forecast_move(move)
                score, _ = self.alphabeta(next_state, depth - 1, alpha, beta, True)
                if score < best_score:
                    best_score, best_move = score, move
                # Prune
                if best_score <= alpha:
                    return best_score, best_move
                # Update beta
                beta = min(beta, best_score)
        return best_score, best_move