# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

# und noch wegen q5
import math

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** MY CODE HERE ***"

        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        if foodDistances:
            closestFood = min(foodDistances)
        else:
            closestFood = 1

        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        ghostPenalty = 0

        for i, ghostDist in enumerate(ghostDistances):
            if newScaredTimes[i] > 0:
                ghostPenalty += 200 / (ghostDist + 1)
            else:
                ghostPenalty -= 10 / (ghostDist + 1)

        foodReward = 10 / closestFood

        return successorGameState.getScore() + foodReward + ghostPenalty

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    # meine hilfsfunktion
    def minmax(self,gameState: GameState, depth:int, current:int):
        if gameState.isWin() or gameState.isLose() or depth==0:
            return self.evaluationFunction(gameState)

        numagents =gameState.getNumAgents() # anz. agenten
        nextactor = (current +1)% numagents


        if current==0:
            maxeval=float('-inf')
            for act in gameState.getLegalActions(0):
                eval = self.minmax(gameState.generateSuccessor(0, act), depth - 1 if nextactor == 0 else depth, nextactor)
                maxeval = max(eval, maxeval)
            return maxeval
        else:
            mineval = float('+inf')
            for act in gameState.getLegalActions(current):
                eval = self.minmax(gameState.generateSuccessor(current, act), depth - 1 if nextactor == 0 else depth, nextactor)
                mineval = min(eval, mineval)
            return mineval


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** MY CODE HERE ***"

        legalActions = gameState.getLegalActions(0)
        if len(legalActions) == 0:
            return Directions.STOP

        bestaction = Directions.STOP

        bestscore = float('-inf')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            score = self.minmax(successor, self.depth, 1)

            if score > bestscore:
                bestscore = score
                bestaction = action

        return bestaction
        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** MY CODE HERE ***"
        legal_actions = gameState.getLegalActions(0)
        if len(legal_actions) == 0:
            return Directions.STOP

        alpha = float('-inf')
        beta = float('inf')
        best_action = None
        v = float('-inf')

        for action in legal_actions:
            value = self.min_Value(gameState.generateSuccessor(0, action), self.depth, 1, alpha, beta)
            if value > v:
                v = value
                best_action = action
            alpha = max(alpha, v)

        return best_action
        #util.raiseNotDefined()
    
    ##########################
    ##  HILFSFUNKTIONEN     ##
    ##########################
    def min_Value(self, gameState: GameState, depth: int, current: int, alpha: int, beta: int):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        mineval = float('inf')
        numAgents = gameState.getNumAgents()
        next_agent = (current + 1) % numAgents
        next_depth = depth - 1 if next_agent== 0 else depth

        for action in gameState.getLegalActions(current):
            if next_agent == 0:
                eval = self.max_Value(gameState.generateSuccessor(current, action), next_depth, next_agent, alpha, beta)
            else:
                eval = self.min_Value(gameState.generateSuccessor(current, action), next_depth, next_agent, alpha, beta)
            mineval = min(mineval, eval)
            if mineval < alpha:
                return mineval
            beta = min(beta, mineval)
        return mineval
    
    def max_Value(self, gameState: GameState, depth: int, current: int, alpha: int, beta: int):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        maxeval = float('-inf')
        numAgents = gameState.getNumAgents()
        next_agent = (current + 1) % numAgents
        next_depth = depth - 1 if next_agent == 0 else depth

        for action in gameState.getLegalActions(current):
            if next_agent == 0:
                eval = self.max_Value(gameState.generateSuccessor(current, action), next_depth, next_agent, alpha, beta)
            else:
                eval = self.min_Value(gameState.generateSuccessor(current, action), next_depth, next_agent, alpha, beta)
            maxeval = max(maxeval, eval)
            if maxeval > beta:
                return maxeval
            alpha = max(alpha, maxeval)
        return maxeval



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def max_Value(self, gameState: GameState, depth: int, current: int):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        numAgents = gameState.getNumAgents()
        nextActor = (current + 1) % numAgents

        maxEval = float('-inf')
        for action in gameState.getLegalActions(current):
            successor = gameState.generateSuccessor(current, action)
            eval = self.exp_Value(successor, depth - 1 if nextActor == 0 else depth, nextActor)
            maxEval = max(maxEval, eval)
        return maxEval
    
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** MY CODE HERE ***"

        legalActions = gameState.getLegalActions(0)
        if not legalActions:
            return Directions.STOP

        bestAction = None
        bestValue = float('-inf')

        for action in legalActions:
            successor = gameState.generateSuccessor(0, action)
            actionValue = self.exp_Value(successor, self.depth, 1)

            if actionValue > bestValue:
                bestValue = actionValue
                bestAction = action

        return bestAction

        #util.raiseNotDefined()
    
    def exp_Value(self, gameState: GameState, depth: int, current: int):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        numAgents = gameState.getNumAgents()
        nextActor = (current + 1) % numAgents

        legalActions = gameState.getLegalActions(current)
        totalValue = 0

        for action in legalActions:
            successor = gameState.generateSuccessor(current, action)
            if nextActor == 0:
                totalValue += self.max_Value(successor, depth - 1 if nextActor == 0 else depth, nextActor)
            else:
                totalValue += self.exp_Value(successor, depth - 1 if nextActor == 0 else depth, nextActor)

        return totalValue / len(legalActions)



def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** MY CODE HERE ***"
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** MY CODE HERE ***"

    foodDistances = [euclideanDistance(newPos, food) for food in newFood.asList()]
    if foodDistances:
        closestFood = min(foodDistances)
    else:
        closestFood = 1

    ghostDistances = [euclideanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    ghostPenalty = 0
    for i, ghostDist in enumerate(ghostDistances):
        if newScaredTimes[i] > 0:
            ghostPenalty += 200 / (ghostDist + 1)
        else:
            ghostPenalty -= 10 / (ghostDist + 1)

    foodReward = 10 / closestFood

    return successorGameState.getScore() + foodReward + ghostPenalty


def euclideanDistance(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    #util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
