import numpy
import random, util
from game import Agent


#     ********* Reflex agent- sections a and b *********
class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
        ------------------------------------------------------------------------------
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current GameState (pacman.py) and the proposed action
        and returns a number, where higher numbers are better.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        return betterEvaluationFunction(successorGameState)


#     ********* Evaluation functions *********

def scoreEvaluationFunction(gameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
    """
    return gameState.getScore()


######################################################################################
# b: implementing a better heuristic function
def betterEvaluationFunction(gameState):
    """

    The betterEvaluationFunction takes in a GameState (pacman.py) and should return a number, where higher numbers are better.

    A GameState specifies the full game state, including the food, capsules, agent configurations and more.
    Following are a few of the helper methods that you can use to query a GameState object to gather information about
    the present state of Pac-Man, the ghosts and the maze:

    gameState.getLegalActions():
    gameState.getPacmanState():
    gameState.getGhostStates():
    gameState.getNumAgents():
    gameState.getScore():
    The GameState class is defined in pacman.py and you might want to look into that for other helper methods.
    """

    score = gameState.getScore()
    food_grid = gameState.getFood()
    longest_road = food_grid.height + food_grid.width  # longest manheten distance.
    score -= longest_road * len(gameState.getCapsules())  # giving the number of pills left some values.
    num_food = gameState.getNumFood()

    capsules_distances = [util.manhattanDistance(gameState.getPacmanPosition(), capsule) for capsule in
                          gameState.getCapsules()]
    closest_capsule_dist = 1
    if len(capsules_distances) > 0:
        closest_capsule_dist = min(capsules_distances)
    capsules = gameState.getCapsules()
    capsule_value = closest_capsule_dist

    scared_value = 0
    ghost_distance = 0
    num_of_ghosts = len(gameState.getGhostStates())
    for ghost_state in gameState.getGhostStates():
        if ghost_state.scaredTimer > 0:
            scared_value = util.manhattanDistance(gameState.getPacmanPosition(), ghost_state.configuration.pos)
        else:
            curr_ghost_distance = util.manhattanDistance(gameState.getPacmanPosition(), ghost_state.configuration.pos)
            if curr_ghost_distance <= 1:
                return -100000000
            ghost_distance += curr_ghost_distance
    if num_of_ghosts == 0:
        ghost_distance /= 1

    food_distances = []
    food_grid = gameState.getFood()
    for x in range(food_grid.width):
        for y in range(food_grid.height):
            if food_grid[x][y] is True:
                food_distances.append(util.manhattanDistance(gameState.getPacmanPosition(), (x, y)))

    closest_food_list = []
    closest_food_value = 0
    total_food_dist = 0
    if (num_food > 0):
        for _ in range(3):
            if len(food_distances) != 0:
                closest_food_list.append(min(food_distances))
                food_distances.remove(closest_food_list[-1])
        closest_food_value = random.choice(closest_food_list)
        total_food_dist = sum(food_distances) / num_food
    N_score = 1000000
    N_scared = 50
    if (num_food >= 0.3 * food_grid.width * food_grid.height):
        N_capsules = 5  # if food is more than 30%+- then chase capsules more
    else:
        N_capsules = 20
    N_closest_food = 12
    N_total_food = 5
    N_ghosts = 1
    return N_score * (score) ** 3 - N_capsules * capsule_value - N_scared * (scared_value) ** 2 - N_closest_food * (
        closest_food_value) ** 2 - N_total_food * (total_food_dist) + N_ghosts * (ghost_distance) ** 2


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxAgent, AlphaBetaAgent & both ExpectimaxAgents.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='betterEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# c: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.getScore():
            Returns the score corresponding to the current state of the game

          gameState.isWin():
            Returns True if it's a winning state

          gameState.isLose():
            Returns True if it's a losing state

          self.depth:
            The depth to which search should continue

        """

        # BEGIN_YOUR_CODE
        def G(gameState):
            return gameState.isWin() or gameState.isLose()

        def U(gameState):
            if gameState.isWin():
                return numpy.inf
            if gameState.isLose():
                return -numpy.inf

        def Turn(agent_index):
            if agent_index + 1 < gameState.getNumAgents():
                return agent_index + 1
            else:
                return 0

        # The heuristic evaluation function
        evalFumc = self.evaluationFunction

        def GetMinMaxAction(gameState, agent_index, depth):
            # we reached a win or a lose situation.
            if G(gameState):
                return (U(gameState), None)
            # end of search depth.
            if depth == 0:
                return (evalFumc(gameState), None)
            if agent_index == 0:
                # Pacmans turn
                CurrMax = -numpy.inf
                MaxAction = None
                # if there are no agents every call we should go one layer deeper.
                if gameState.getNumAgents() == 1:
                    depth -= 1
                for move in gameState.getLegalActions(agent_index):
                    v = GetMinMaxAction(gameState.generateSuccessor(agent_index, move), Turn(agent_index), depth)
                    if CurrMax <= v[0]:
                        CurrMax = v[0]
                        MaxAction = move
                return (CurrMax, MaxAction)
            else:
                # Ghosts turn
                CurrMin = numpy.inf
                MinAction = None
                for move in gameState.getLegalActions(agent_index):
                    if Turn(agent_index) == 0:
                        # the next turn will be pacmans so go one depth lower.
                        v = GetMinMaxAction(gameState.generateSuccessor(agent_index, move), Turn(agent_index),
                                            depth - 1)
                    else:
                        # next turn is another ghost so stay in same depth.
                        v = GetMinMaxAction(gameState.generateSuccessor(agent_index, move), Turn(agent_index), depth)
                    if CurrMin >= v[0]:
                        CurrMin = v[0]
                        MinAction = move
                return (CurrMin, MinAction)

        return GetMinMaxAction(gameState, 0, self.depth)[1]
        # END_YOUR_CODE


######################################################################################
# d: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        # BEGIN_YOUR_CODE
        def G(gameState):
            return gameState.isWin() or gameState.isLose()

        def U(gameState):
            if gameState.isWin():
                return numpy.inf
            if gameState.isLose():
                return -numpy.inf

        def Turn(agent_index):
            if agent_index + 1 < gameState.getNumAgents():
                return agent_index + 1
            else:
                return 0

        # The heuristic evaluation function
        evalFumc = self.evaluationFunction

        def GetMinMaxActionAlphaBeta(gameState, agent_index, depth, alpha, beta):
            # we reached a win or a lose situation.
            if G(gameState):
                return (U(gameState), None)
            # end of search depth.
            if depth == 0:
                return (evalFumc(gameState), None)
            if agent_index == 0:
                # Pacmans turn
                CurrMax = -numpy.inf
                MaxAction = None
                # if there are no agents every call we should go one layer deeper.
                if gameState.getNumAgents() == 1:
                    depth -= 1
                for move in gameState.getLegalActions(agent_index):
                    v = GetMinMaxActionAlphaBeta(gameState.generateSuccessor(agent_index, move), Turn(agent_index),
                                                 depth, alpha, beta)
                    if CurrMax <= v[0]:
                        CurrMax = v[0]
                        MaxAction = move
                    alpha = max(CurrMax, alpha)
                    if CurrMax >= beta:
                        return (numpy.inf, move)
                return (CurrMax, MaxAction)
            else:
                # Ghosts turn
                CurrMin = numpy.inf
                MinAction = None
                for move in gameState.getLegalActions(agent_index):
                    if Turn(agent_index) == 0:
                        # the next turn will be pacmans so go one depth lower.
                        v = GetMinMaxActionAlphaBeta(gameState.generateSuccessor(agent_index, move), Turn(agent_index),
                                                     depth - 1, alpha, beta)
                    else:
                        # next turn is another ghost so stay in same depth.
                        v = GetMinMaxActionAlphaBeta(gameState.generateSuccessor(agent_index, move), Turn(agent_index),
                                                     depth, alpha, beta)
                    if CurrMin >= v[0]:
                        CurrMin = v[0]
                        MinAction = move
                    if Turn(agent_index) == 0:
                        beta = min(CurrMin, beta)
                        if CurrMin <= alpha:
                            return (-numpy.inf, move)
                return (CurrMin, MinAction)

        return GetMinMaxActionAlphaBeta(gameState, 0, self.depth, -numpy.inf, numpy.inf)[1]
        # END_YOUR_CODE


######################################################################################
# e: implementing random expectimax

class RandomExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their legal moves.
        """

        # BEGIN_YOUR_CODE
        def G(gameState):
            return gameState.isWin() or gameState.isLose()

        def U(gameState):
            if gameState.isWin():
                return numpy.inf
            if gameState.isLose():
                return -numpy.inf

        def Turn(agent_index):
            if agent_index + 1 < gameState.getNumAgents():
                return agent_index + 1
            else:
                return 0

        def UniformProbability(gameState, agent_index):
            moves = gameState.getLegalActions(agent_index)
            states = [gameState.generateSuccessor(agent_index, move) for move in moves]
            return [(state, 1 / len(moves)) for state in states]

        # The heuristic evaluation function
        evalFumc = self.evaluationFunction

        def GetExpectimaxAction(gameState, agent_index, depth, Probabilistic):
            # we reached a win or a lose situation.
            if G(gameState):
                return (U(gameState), None)
            # end of search depth.
            if depth == 0:
                return (evalFumc(gameState), None)
            if Probabilistic:
                values = []
                for c, p in UniformProbability(gameState, agent_index):
                    if Turn(agent_index) == 0:
                        values.append(p * GetExpectimaxAction(c, Turn(agent_index), depth - 1, False)[0])
                    else:
                        values.append(p * GetExpectimaxAction(c, Turn(agent_index), depth, False)[0])
                return (sum(values), None)
            if agent_index == 0:
                # Pacmans turn
                CurrMax = -numpy.inf
                MaxAction = None
                # if there are no agents every call we should go one layer deeper.
                if gameState.getNumAgents() == 1:
                    depth -= 1
                for move in gameState.getLegalActions(agent_index):
                    v = GetExpectimaxAction(gameState.generateSuccessor(agent_index, move), Turn(agent_index), depth,
                                            False)
                    if CurrMax <= v[0]:
                        CurrMax = v[0]
                        MaxAction = move
                return (CurrMax, MaxAction)
            else:
                # Ghosts turn
                CurrMin = numpy.inf
                MinAction = None
                for move in gameState.getLegalActions(agent_index):
                    # next turn is another ghost so stay in same depth.
                    v = GetExpectimaxAction(gameState.generateSuccessor(agent_index, move), agent_index,
                                            depth, True)
                    if CurrMin >= v[0]:
                        CurrMin = v[0]
                        MinAction = move
                return (CurrMin, MinAction)

        return GetExpectimaxAction(gameState, 0, self.depth, False)[1]
        # END_YOUR_CODE


######################################################################################
# f: implementing directional expectimax

class DirectionalExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as using the DirectionalGhost distribution to choose from their legal moves.
        """

        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE


######################################################################################
# I: implementing competition agent

class CompetitionAgent(MultiAgentSearchAgent):
    """
      Your competition agent
    """

    def getAction(self, gameState):
        """
          Returns the action using self.depth and self.evaluationFunction

        """

        # BEGIN_YOUR_CODE
        raise Exception("Not implemented yet")
        # END_YOUR_CODE
