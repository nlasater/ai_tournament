# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'Attacker', second = 'Defender'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

# action selection agent

class Action():
  """ This is composed from code from baselineTeam.py """
  
  """ This class serves as a general structure for both offensive and defensive agents.
  
      This will form the skeleton of our agent.
      
  """
  
  
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
    
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights
    
  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}
  
# offensive strategy:
class getOffAction(Action):
  """ This is the offensive behaviors. This extends some methods of Action"""
  def __init__ (self, agent, index, gameState, init_food):
    self.agent = agent
    self.index = index
    self.count = 0
    self.agent.distancer.getMazeDistances()
    self.init_food = init_food
    
    # establish the player zones
    bound = (gameState.data.layout.width - 2)/2
    
    if not self.agent.red: #if we are NOT the red zone
      bound = bound+1
      
    # feed this into self:
    self.boundary = []
    for i in range(1, (gameState.data.layout.height - 1)): # iterate along boundary line and add walkable boundaries to the list
      if not gameState.hasWall(bound, i):
        self.boundary.append((bound, i))
        
    
  def getFeatures (self, state, action):
    """ 
    List of features:
    
    sucScore = current player score
    returnDist = distance to nearest boundary
    eaten = food eaten so far
    foodDist = distance to nearest food
    capDist = distance to nearest capsule
    distFromGhost = distance from enemy ghost (0 if > 5)
    nomPacmanDist = distance to nommable enemy pacman (0 if > 4)
    
    """
    
    feats = util.Counter()
    successor = self.getSuccessor(state,action)
    
    # compute suc's score
    feats['sucScore'] = self.agent.getScore(successor)
    
    # compute current position:
    curr_pos = successor.getAgentState(self.index).getPosition()
    
    # find the distance to boundary
    boMin = float("inf")
    for i in range(len(self.boundary)):
      dist = self.agent.getMazeDistance(curr_pos, self.boundary[i])
      if dist < boMin:
        boMin = dist
    
    feats['returnDist'] = boMin
    
    
    # food eaten so far:
    feats['eaten'] = self.init_food - self.agent.getFood(successor).count()
    
    # find distance to nearest food (that we can eat)
    if (self.agent.getFood(successor).count()) > 0:
      closest = float("inf")
      for f in self.agent.getFood(successor).asList():
        dist = self.agent.getMazeDistance(curr_pos, f)
        if dist < closest:
          closest = dist
    feats['foodDist'] = closest
    
    # distance to nearest capsule:
    if len(self.agent.getCapsules(successor)) > 0:
      closest = float("inf")
      for f in self.agent.getCapsules(successor):
        dist = self.agent.getMazeDistance(curr_pos, f)
        if dist < closest:
          closest = dist
    feats['capDist'] = closest
    
    
    # distance to nearest enemy ghost:
    oppStates = []
    for i in self.agent.getOpponents(successor):
      oppStates.append(successor.getAgentState(i))
    valid = filter(lambda x: not x.isPacman and x.getPosition() != None, oppStates)
    if len(valid) > 0:
      posits = [agent.getPosition() for agent in valid]
      closest = min(posits, key=lambda x: self.agent.getMazeDistance(curr_pos, x))
      closeDist = self.agent.getMazeDistance(curr_pos, closest)
      if closeDist <= 5:
        feats['distFromGhost'] = closeDist

    else:
      # since we don't know the actual positions, we can use this:
      prb = []
      for i in self.agent.getOpponents(successor):
        prb.append(successor.getAgentDistances()[i]) # get the distance (not position, but close enough)
      feats['distFromGhost'] = min(prb) # choose the smallest option for the closest ghost

        
    
  
    # this will signal to the attacker to eat the enemy player iff hes a ghost and enemy is close
    # first get all enemy states:
    enemies = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
    # then get only the pacmans:
    enPacman = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
    if len(enPacman) > 0: # if we have at least one pacman
      posits = [agent.getPosition() for agent in enPacman] # get all the positions
      closest = min(posits, key=lambda x: self.agent.getMazeDistance(curr_pos, x)) # find the closest one
      closeDist = self.agent.getMazeDistance(curr_pos, closest) # find the dist to said closest one
      if closeDist < 4:
        # print(CurrentPosition,closest,closestDist)
        feats['nomPacmanDist'] = closeDist
    else:
      feats['nomPacmanDist'] = 0 # if theres no pacman, return the default 0
      # not really sure this is needed, shouldn't counters return 0 if no value set?
        
    
    # finally, return:
    return feats
  
  def getWeights(self, state, action):
    """ Get the weights for the features """
    # get successor:
    successor = self.getSuccessor(state, action)
    curr_pos = successor.getAgentState(self.index).getPosition()
    
    # see how much we've eaten:
    eaten = self.init_food - self.agent.getFood(successor).count()
    
    # if we are pacman:
    if state.getAgentState(self.index).isPacman:
      
      # test to see if opp is scared, if so we can go for the ghost if we're close
      allEnem = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
      enemyGhosts = []
      # parse for only enemy ghosts:
      for g in allEnem:
        if not g.isPacman: 
          enemyGhosts.append(g)

      for gh in enemyGhosts:
        if gh.scaredTimer > 0:
          if gh.scaredTimer > 12: # we have lots of time to get there:
            return {'sucScore': 110, 'foodDist': -10, 'nomPacmanDist': 0, 'distFromGhost': -1, 'capDist': 0, 'returnDist': 10-3*eaten, 'eaten': 350}

          elif gh.scaredTimer > 6:
            return {'sucScore': 110+5*eaten, 'foodDist': -5, 'nomPacmanDist': 0, 'distFromGhost': -1, 'capDist': -10, 'returnDist': -5-4*eaten, 'eaten': 100}


        # if theyre not scared:
        else:
          return {'sucScore': 110, 'foodDist': -10, 'nomPacmanDist': 0, 'distFromGhost': 20, 'capDist': -15, 'returnDist': -15, 'eaten': 0}
        
    # if we are a ghost:
    else:
      # if we are close to an enemy pacman, nom it:
      # first get all enemy states:
      enemies = [successor.getAgentState(i) for i in self.agent.getOpponents(successor)]
      # then get only the pacmans:
      enPacman = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
      if len(enPacman) > 0: # if we have at least one pacman
        posits = [agent.getPosition() for agent in enPacman] # get all the positions
        closest = min(posits, key=lambda x: self.agent.getMazeDistance(curr_pos, x)) # find the closest one
        closeDist = self.agent.getMazeDistance(curr_pos, closest) # find the dist to said closest one
        if closeDist < 4:
            return {'sucScore': 0, 'foodDist': -1, 'nomPacmanDist': -8, 'distFromGhost': 0, 'capDist': 0, 'returnDist': 0, 'eaten': 10}

      
      # otherwise, return the same as above:
      return {'sucScore': 110, 'foodDist': -10, 'nomPacmanDist': 0, 'distFromGhost': 20, 'capDist': -15, 'returnDist': -15, 'eaten': 0}
    
  
  
    """
    FROM HERE ON NEEDS TO BE EDITED

    DO NOT SUBMIT UNTIL THIS HAS BEEN CHANGED, SO IT DOESNT READ AS A COPY

    """
  def allSimulation(self, depth, gameState, decay):
    new_state = gameState.deepCopy()
    if depth == 0:
        result_list = []
        actions = new_state.getLegalActions(self.index)
        actions.remove(Directions.STOP)

        reversed_direction = Directions.REVERSE[new_state.getAgentState(self.index).configuration.direction]
        if reversed_direction in actions and len(actions) > 1:
            actions.remove(reversed_direction)
        a = random.choice(actions)
        next_state = new_state.generateSuccessor(self.index, a)
        result_list.append(self.evaluate(next_state, Directions.STOP))
        return max(result_list)

    # Get valid actions
    result_list = []
    actions = new_state.getLegalActions(self.index)
    current_direction = new_state.getAgentState(self.index).configuration.direction
    # The agent should not use the reverse direction during simulation

    reversed_direction = Directions.REVERSE[current_direction]
    if reversed_direction in actions and len(actions) > 1:
        actions.remove(reversed_direction)

    # Randomly chooses a valid action
    for a in actions:
        # Compute new state and update depth
        next_state = new_state.generateSuccessor(self.index, a)
        result_list.append(
            self.evaluate(next_state, Directions.STOP) + decay * self.allSimulation(depth - 1, next_state, decay))
    return max(result_list)

  def MTCS(self, depth, gameState, decay):
    """
    Random simulate some actions for the agent. The actions other agents can take
    are ignored, or, in other words, we consider their actions is always STOP.
    The final state from the simulation is evaluated.
    """
    new_state = gameState.deepCopy()
    value = self.evaluate(new_state, Directions.STOP)
    decay_index = 1
    while depth > 0:

        # Get valid actions
        actions = new_state.getLegalActions(self.index)
        # The agent should not stay put in the simulation
        # actions.remove(Directions.STOP)
        current_direction = new_state.getAgentState(self.agent.index).configuration.direction
        # The agent should not use the reverse direction during simulation
        reversed_direction = Directions.REVERSE[new_state.getAgentState(self.agent.index).configuration.direction]
        if reversed_direction in actions and len(actions) > 1:
            actions.remove(reversed_direction)
        # Randomly chooses a valid action
        a = random.choice(actions)
        # Compute new state and update depth
        new_state = new_state.generateSuccessor(self.agent.index, a)
        value = value + decay ** decay_index * self.evaluate(new_state, Directions.STOP)
        depth -= 1
        decay_index += 1
    # Evaluate the final simulation state
    return value


  def chooseAction(self, gameState):
    start = time.time()

    # Get valid actions. Randomly choose a valid one out of the best (if best is more than one)
    actions = gameState.getLegalActions(self.agent.index)
    actions.remove(Directions.STOP)
    feasible = []
    for a in actions:
        value = 0
        # for i in range(0, 10):
        #     value += self.randomSimulation1(2, new_state, 0.8) / 10
        # fvalues.append(value)
        value = self.allSimulation(2, gameState.generateSuccessor(self.agent.index, a), 0.7)
        feasible .append(value)

    bestAction = max(feasible)
    possibleChoice = filter(lambda x: x[0] == bestAction, zip(feasible, actions))
    #print 'eval time for offensive agent %d: %.4f' % (self.agent.index, time.time() - start)
    return random.choice(possibleChoice)[1]



class getDefensiveActions(Action):
  # Load the denfensive information
  def __init__(self, agent, index, gameState, init_food):
    self.index = index
    self.agent = agent
    self.DenfendList = {}

    if self.agent.red:
        middle = (gameState.data.layout.width - 2) / 2
    else:
        middle = ((gameState.data.layout.width - 2) / 2) + 1
    self.boundary = []
    for i in range(1, gameState.data.layout.height - 1):
        if not gameState.hasWall(middle, i):
            self.boundary.append((middle, i))

    self.target = None
    self.lastObservedFood = None
    # Update probabilities to each patrol point.
    self.DefenceProbability(gameState)

  def DefenceProbability(self, gameState):
    """
    This method calculates the minimum distance from our patrol
    points to our pacdots. The inverse of this distance will
    be used as the probability to select the patrol point as
    target.
    """
    total = 0

    for position in self.boundary:
        food = self.agent.getFoodYouAreDefending(gameState).asList()
        closestFoodDistance=min(self.agent.getMazeDistance(position,f) for f in food)
        if closestFoodDistance == 0:
            closestFoodDistance = 1
        self.DenfendList[position] = 1.0 / float(closestFoodDistance)
        total += self.DenfendList[position]

    # Normalize.
    if total == 0:
        total = 1
    for x in self.DenfendList.keys():
        self.DenfendList[x] = float(self.DenfendList[x]) / float(total)

  def selectPatrolTarget(self):
    """
    Select some patrol point to use as target.
    """

    maxProb=max(self.DenfendList[x] for x in self.DenfendList.keys())
    bestTarget = filter(lambda x: self.DenfendList[x] == maxProb, self.DenfendList.keys())
    return random.choice(bestTarget)

  def chooseAction(self, gameState):

    # start = time.time()

    DefendingList = self.agent.getFoodYouAreDefending(gameState).asList()
    if self.lastObservedFood and len(self.lastObservedFood) != len(DefendingList):
        self.DefenceProbability(gameState)
    myPos = gameState.getAgentPosition(self.index)
    if myPos == self.target:
        self.target = None

    # Visible enemy , keep chasing.
    enemies=[gameState.getAgentState(i) for i in self.agent.getOpponents(gameState)]
    inRange = filter(lambda x: x.isPacman and x.getPosition() != None,enemies)
    if len(inRange)>0:
        eneDis,enemyPac = min([(self.agent.getMazeDistance(myPos,x.getPosition()), x) for x in inRange])
        self.target=enemyPac.getPosition()
        #for x in inRange:
            #if self.agent.getMazeDistance(myPos,x.getPosition())==closestGhost:
                #self.target=x.getPosition()
                #print(self.target)

    elif self.lastObservedFood != None:
        eaten = set(self.lastObservedFood) - set(self.agent.getFoodYouAreDefending(gameState).asList())
        if len(eaten)>0:
           closestFood, self.target = min([(self.agent.getMazeDistance(myPos,f),f) for f in eaten])

    self.lastObservedFood = self.agent.getFoodYouAreDefending(gameState).asList()


    # We have only a few dots.
    if self.target == None and len(self.agent.getFoodYouAreDefending(gameState).asList()) <= 4:
        food = self.agent.getFoodYouAreDefending(gameState).asList() + self.agent.getCapsulesYouAreDefending(gameState)
        self.target = random.choice(food)

    # Random patrolling
    elif self.target == None:
        self.target = self.selectPatrolTarget()

    actions = gameState.getLegalActions(self.index)
    feasible = []
    fvalues = []
    for a in actions:
        new_state = gameState.generateSuccessor(self.index, a)
        if not a == Directions.STOP and not new_state.getAgentState(self.index).isPacman:
            newPosition = new_state.getAgentPosition(self.index)
            feasible.append(a)
            fvalues.append(self.agent.getMazeDistance(newPosition, self.target))

    # Randomly chooses between ties.
    best = min(fvalues)
    ties = filter(lambda x: x[0] == best, zip(fvalues, feasible))

    # print 'eval time for defender agent %d: %.4f' % (self.index, time.time() - start)
    return random.choice(ties)[1]


class Attacker(CaptureAgent):
  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  def registerInitialState(self, gameState):
    self.init_food = gameState.getRedFood().count()
    CaptureAgent.registerInitialState(self, gameState)
    self.DefenceStatus = getDefensiveActions(self, self.index, gameState, self.init_food)
    self.OffenceStatus = getOffAction(self, self.index, gameState, self.init_food)

  def chooseAction(self, gameState):
    self.enemies = self.getOpponents(gameState)
    invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]


    if  self.getScore(gameState) >= 13:
        return self.DefenceStatus.chooseAction(gameState)
    else:
        return self.OffenceStatus.chooseAction(gameState)


class Defender(CaptureAgent):
  def __init__(self, index):
    CaptureAgent.__init__(self, index)

  def registerInitialState(self, gameState):
    self.init_food = gameState.getRedFood().count()
    CaptureAgent.registerInitialState(self, gameState)
    self.DefenceStatus = getDefensiveActions(self, self.index, gameState, self.init_food)
    self.OffenceStatus = getOffAction(self, self.index, gameState, self.init_food)

  def chooseAction(self, gameState):
    self.enemies = self.getOpponents(gameState)
    invaders = [a for a in self.enemies if gameState.getAgentState(a).isPacman]
    numInvaders = len(invaders)

    # Check if we have the poison active.
    scaredTimes = [gameState.getAgentState(enemy).scaredTimer for enemy in self.enemies]
    # if numInvaders == 0 and self.getScore(gameState) < 10:
    #     return self.OffenceStatus.chooseAction(gameState)
    # else:
    #     print(self.DefenceStatus.target, "Target is ..........")
    return self.DefenceStatus.chooseAction(gameState)

