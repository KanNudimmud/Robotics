# -----------
# User Instructions:
#
# Modify the the search function so that it returns
# a shortest path as follows:
#
# [['>', 'v', ' ', ' ', ' ', ' '],
#  [' ', '>', '>', '>', '>', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', 'v'],
#  [' ', ' ', ' ', ' ', ' ', '*']]
#
# Where '>', '<', '^', and 'v' refer to right, left,
# up, and down motions. Note that the 'v' should be
# lowercase. '*' should mark the goal cell.
#
# You may assume that all test cases for this function
# will have a path from init to goal.
# ----------

grid = [[0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1, 0]]
init = [0, 0]
goal = [len(grid) - 1, len(grid[0]) - 1]
cost = 1

delta = [[-1, 0],  # go up
         [0, -1],  # go left
         [1, 0],  # go down
         [0, 1]]  # go right

delta_name = ['^', '<', 'v', '>']

def search(grid, init, goal, cost):
    # ----------------------------------------
    # modify code below
    # ----------------------------------------
    closed = [[0 for row in range(len(grid[0]))] for col in range(len(grid))]
    closed[init[0]][init[1]] = 1

    x = init[0]
    y = init[1]
    g = 0

    expand = [[' ' for row in range(len(grid[0]))] for col in range(len(grid))]
    movement = [[-1 for row in range(len(grid[0]))] for col in range(len(grid))]

    open = [[g, x, y]]

    found = False  # flag that is set when search is complete
    resign = False  # flag set if we can't find expand

    while not found and not resign:
        if len(open) == 0:
            resign = True
            return 'fail'
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            x = next[1]
            y = next[2]
            g = next[0]

            if x == goal[0] and y == goal[1]:
                found = True
            else:
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            open.append([g2, x2, y2])
                            closed[x2][y2] = 1
                            movement[x2][y2] = i
    x = goal[0]
    y = goal[1]
    expand[x][y] = '*'
    flag = 0
    while flag == 0:
        x3 = x - delta[movement[x][y]][0]
        y3 = y - delta[movement[x][y]][1]
        expand[x3][y3] = delta_name[movement[x][y]]

        x = x3
        y = y3

        if (x == init[0] and y == init[1]):
            flag = 1

    for i in range(len(expand)):
        print expand[i]
    return expand  # make sure you return the shortest path

search(grid, init, goal, cost) # TRUE

# -----------
# User Instructions:
#
# Modify the the search function so that it becomes
# an A* search algorithm as defined in the previous
# lectures.
#
# Your function should return the expanded grid
# which shows, for each element, the count when
# it was expanded or -1 if the element was never expanded.
#
# If there is no path from init to goal,
# the function should return the string 'fail'
# ----------

grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]]
heuristic = [[9, 8, 7, 6, 5, 4],
             [8, 7, 6, 5, 4, 3],
             [7, 6, 5, 4, 3, 2],
             [6, 5, 4, 3, 2, 1],
             [5, 4, 3, 2, 1, 0]]

init = [0, 0]
goal = [len(grid) - 1, len(grid[0]) - 1]
cost = 1

delta = [[-1, 0],  # go up
         [0, -1],  # go left
         [1, 0],  # go down
         [0, 1]]  # go right

delta_name = ['^', '<', 'v', '>']


def search(grid, init, goal, cost, heuristic):
    # ----------------------------------------
    # modify the code below
    # ----------------------------------------
    closed = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]
    closed[init[0]][init[1]] = 1

    expand = [[-1 for col in range(len(grid[0]))] for row in range(len(grid))]
    action = [[-1 for col in range(len(grid[0]))] for row in range(len(grid))]

    x = init[0]
    y = init[1]
    g = 0

    open = [[g, x, y]]

    found = False  # flag that is set when search is complete
    resign = False  # flag set if we can't find expand
    count = 0

    while not found and not resign:
        if len(open) == 0:
            resign = True
            return "Fail"
        else:
            open.sort()
            open.reverse()
            next = open.pop()
            x = next[1]
            y = next[2]
            g = next[0]
            expand[x][y] = count
            count += 1

            if x == goal[0] and y == goal[1]:
                found = True
            else:
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost
                            open.append([g2, x2, y2])
                            closed[x2][y2] = 1

    return expand

# ----------
# User Instructions:
#
# Create a function compute_value which returns
# a grid of values. The value of a cell is the minimum
# number of moves required to get from the cell to the goal.
#
# If a cell is a wall or it is impossible to reach the goal from a cell,
# assign that cell a value of 99.
# ----------

grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]]
goal = [len(grid) - 1, len(grid[0]) - 1]
cost = 1  # the cost associated with moving from a cell to an adjacent one

delta = [[-1, 0],  # go up
         [0, -1],  # go left
         [1, 0],  # go down
         [0, 1]]  # go right

delta_name = ['^', '<', 'v', '>']


def compute_value(grid, goal, cost):
    value = [[99 for col in range(len(grid[0]))] for row in range(len(grid))]
    policy = [[" " for col in range(len(grid[0]))] for row in range(len(grid))]

    change = True
    while change:
        change = False

        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if goal[0] == x and goal[1] == y:
                    if value[x][y] > 0:
                        value[x][y] = 0
                        policy[x][y] = "*"
                        change = True
                elif grid[x][y] == 0:
                    for a in range(len(delta)):
                        x2 = x + delta[a][0]
                        y2 = y + delta[a][1]

                        if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]) and grid[x2][y2] == 0:
                            v2 = value[x2][y2] + cost

                            if v2 < value[x][y]:
                                change = True
                                value[x][y] = v2
                                policy[x][y] = delta_name[a]

    # make sure your function returns a grid of values as
    # demonstrated in the previous video.
    print(policy)
    return value

compute_value(grid, goal, cost) # TRUE

# User Instructions:
#
# Write a function optimum_policy that returns
# a grid which shows the optimum policy for robot
# motion. This means there should be an optimum
# direction associated with each navigable cell from
# which the goal can be reached.
#
# Unnavigable cells as well as cells from which
# the goal cannot be reached should have a string
# containing a single space (' '), as shown in the
# previous video. The goal cell should have '*'.
# ----------

grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]]
init = [0, 0]
goal = [len(grid) - 1, len(grid[0]) - 1]
cost = 1  # the cost associated with moving from a cell to an adjacent one

delta = [[-1, 0],  # go up
         [0, -1],  # go left
         [1, 0],  # go down
         [0, 1]]  # go right

delta_name = ['^', '<', 'v', '>']


def optimum_policy(grid, goal, cost):
    # ----------------------------------------
    # modify code below
    # ----------------------------------------
    value = [[99 for row in range(len(grid[0]))] for col in range(len(grid))]

    policy = [[' ' for row in range(len(grid[0]))] for col in range(len(grid))]

    movement = [[-1 for row in range(len(grid[0]))] for col in range(len(grid))]

    change = True

    while change:
        change = False

        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if goal[0] == x and goal[1] == y:
                    if value[x][y] > 0:
                        value[x][y] = 0

                        change = True

                elif grid[x][y] == 0:
                    for a in range(len(delta)):
                        x2 = x + delta[a][0]
                        y2 = y + delta[a][1]

                        if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]) and grid[x2][y2] == 0:
                            v2 = value[x2][y2] + cost

                            if v2 < value[x][y]:
                                change = True
                                value[x][y] = v2
                                movement[x][y] = a

    for i in range(len(movement)):
        for j in range(len(movement[0])):
            if movement[i][j] == -1:
                if i == goal[0] and j == goal[1]:
                    policy[i][j] = '*'
            elif movement[i][j] == 0:
                policy[i][j] = '^'
            elif movement[i][j] == 1:
                policy[i][j] = '<'
            elif movement[i][j] == 2:
                policy[i][j] = 'v'
            elif movement[i][j] == 3:
                policy[i][j] = '>'

    for i in range(len(policy)):
        print
        policy[i]

    return policy

optimum_policy(grid, goal, cost)    # TRUE

# ----------
# User Instructions:
#
# Implement the function optimum_policy2D below.
#
# You are given a car in grid with initial state
# init. Your task is to compute and return the car's
# optimal path to the position specified in goal;
# the costs for each motion are as defined in cost.
#
# There are four motion directions: up, left, down, and right.
# Increasing the index in this array corresponds to making a
# a left turn, and decreasing the index corresponds to making a
# right turn.

forward = [[-1, 0],  # go up
           [0, -1],  # go left
           [1, 0],  # go down
           [0, 1]]  # go right
forward_name = ['up', 'left', 'down', 'right']

# action has 3 values: right turn, no turn, left turn
action = [-1, 0, 1]
action_name = ['R', '#', 'L']

# EXAMPLE INPUTS:
# grid format:
#     0 = navigable space
#     1 = unnavigable space
grid = [[1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1]]

init = [4, 3, 0]  # given in the form [row,col,direction]
# direction = 0: up
#             1: left
#             2: down
#             3: right

goal = [2, 0]  # given in the form [row,col]

cost = [2, 1, 20]  # cost has 3 values, corresponding to making


# a right turn, no turn, and a left turn

# EXAMPLE OUTPUT:
# calling optimum_policy2D with the given parameters should return
# [[' ', ' ', ' ', 'R', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', '#'],
#  ['*', '#', '#', '#', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', ' '],
#  [' ', ' ', ' ', '#', ' ', ' ']]
# ----------

# ----------------------------------------
# modify code below
# ----------------------------------------

def optimum_policy2D(grid, init, goal, cost):
    value = [[[999 for row in range(len(grid[0]))] for col in range(len(grid))],
             [[999 for row in range(len(grid[0]))] for col in range(len(grid))],
             [[999 for row in range(len(grid[0]))] for col in range(len(grid))],
             [[999 for row in range(len(grid[0]))] for col in range(len(grid))]]
    policy = [[[' ' for row in range(len(grid[0]))] for col in range(len(grid))],
              [[' ' for row in range(len(grid[0]))] for col in range(len(grid))],
              [[' ' for row in range(len(grid[0]))] for col in range(len(grid))],
              [[' ' for row in range(len(grid[0]))] for col in range(len(grid))]]
    policy2D = [[' ' for row in range(len(grid[0]))] for col in range(len(grid))]

    change = True

    while change:
        change = False
        for x in range(len(grid)):
            for y in range(len(grid[0])):
                for orientation in range(4):
                    if goal[0] == x and goal[1] == y:
                        if value[orientation][x][y] > 0:
                            value[orientation][x][y] = 0
                            policy[orientation][x][y] = '*'
                            change = True
                    elif grid[x][y] == 0:
                        for i in range(3):
                            o2 = (orientation + action[i]) % 4
                            x2 = x + forward[o2][0]
                            y2 = y + forward[o2][1]
                            if x2 >= 0 and x2 < len(grid) and y2 >= 0 \
                                    and y2 < len(grid[0]) and grid[x2][y2] == 0:
                                v2 = value[o2][x2][y2] + cost[i]
                                if v2 < value[orientation][x][y]:
                                    change = True
                                    value[orientation][x][y] = v2
                                    policy[orientation][x][y] = action_name[i]

    x = init[0]
    y = init[1]
    orientation = init[2]
    policy2D[x][y] = policy[orientation][x][y]
    while policy[orientation][x][y] != '*':
        if policy[orientation][x][y] == '#':
            o2 = orientation
        elif policy[orientation][x][y] == 'R':
            o2 = (orientation - 1) % 4
        elif policy[orientation][x][y] == 'L':
            o2 = (orientation + 1) % 4
        x = x + forward[o2][0]
        y = y + forward[o2][1]
        orientation = o2
        policy2D[x][y] = policy[orientation][x][y]

    return policy2D


for i in optimum_policy2D(grid, init, goal, cost):
    print(i)       # TRUE


