''' Supervisor controller

Supervisor, which spawns 3 different sized blue cubes randomly within a 0.5x0.5 m wide square. It monitors which cubes are in the basket, and it checks if they are on top of eachother in the proper, decreasing size order. If all conditions are met it respawns the cubes up to MAX_ROUNDS times.

'''
# Imports
import numpy as np
from controller import Supervisor

# =============================== PARAMETERS ==============================
SEED = 42
N_CUBES = 3
MAX_ROUNDS = 5                    # total cube sets 
# cube
CUBE_Z = 0.45                     # spawn height of the cube centre (greater than CUBE_SIZE/2 to avoid collision with the floor)
SPAWN_X_RANGE = (0.20, 0.70)      # Reasonably large arbitrary range       
SPAWN_Y_RANGE = (0.05, 0.55)      # Reasonably large arbitrary range
SIZE_RANGE_BIG = (0.06,0.075)     # big cube size range
SIZE_RANGE_MEDIUM = (0.035,0.05)  # medium cube size range
SIZE_RANGE_SMALL = (0.015,0.025)   # small cube size range
#SIZE_BIG = 0.07
#SIZE_MEDIUM = 0.045
#SIZE_SMALL = 0.02

# ============================== INITIALIZATION ============================
# set random seed
rng = np.random.default_rng(seed=SEED)
# create the supervisor instance.
supervisor = Supervisor()
# get the time step of the current world.
timestep = int(supervisor.getBasicTimeStep())
# Get root node
root_node = supervisor.getRoot()
children_field = root_node.getField('children')
# Get basket node
basket_node = supervisor.getFromDef("BASKET")
basket_sx, basket_sy = 0.4, 0.4
#basket_sx, basket_sy = supervisor.getPosition(basket_node)[0], supervisor.getPosition(basket_node)[1]

class SimEnded(Exception):
    """Raised when Webots stops the simulation (step returns -1)."""
def step(n=1):
    for _ in range(n):
        if supervisor.step(timestep) == -1:
            raise SimEnded

# Geometry related functions
def to_local(node, p_world):
    """World point -> node's local frame (uses the node's global pose)."""
    t = np.array(node.getPosition())
    R = np.array(node.getOrientation()).reshape(3, 3)
    return R.T @ (np.asarray(p_world, dtype=float) - t)

def in_basket_xy(p_world, margin=0.02):
    """Checks if an object is inside the basket in XY plane, adding a margin of half the cube size."""
    q = to_local(basket_node, p_world)
    return abs(q[0]) <= basket_sx / 2 - margin and abs(q[1]) <= basket_sy / 2 - margin

# Spawn cubes
def sample_position():
    """Samples a random position for a cube within the spawn area."""
    for _ in range(100):  # Try up to 100 times to find a valid position
        x = rng.uniform(*SPAWN_X_RANGE)
        y = rng.uniform(*SPAWN_Y_RANGE)
        p = (x, y, CUBE_Z)
        return p
    raise RuntimeError("Could not find a valid spawn position")

def sample_size():
    """Samples a random size for a cube within the size range"""
    c0 = rng.uniform(*SIZE_RANGE_BIG)
    c1 = rng.uniform(*SIZE_RANGE_MEDIUM)
    c2 = rng.uniform(*SIZE_RANGE_SMALL)
    s= (c0, c1, c2)
    return s

def spawn_cubes():
    nodes = []
    s = sample_size()
    for i in range(N_CUBES):
        p = sample_position()
        node_str = (f'DEF CUBE_{i} cube {{ translation {p[0]} {p[1]} {p[2]} '
                    f'color {0} {0} {0.498039} size {s[i]} {s[i]} {s[i]} name "cube_{i}" }}')
        children_field.importMFNodeFromString(-1, node_str)
        nodes.append(supervisor.getFromDef(f"CUBE_{i}"))
    return nodes

# Monitoring
def cube_status(node):
    p = node.getPosition()
    v = node.getVelocity()
    still = np.linalg.norm(v[:3]) < 0.01 and np.linalg.norm(v[3:]) < 0.01
    in_basket = in_basket_xy(p) and still
    return {"pos": p, "in_basket": in_basket}

# delete cubes
def clear_cubes(nodes):
    for n in nodes:
        if n is not None:
            n.remove()
    step()

# Main loop

try:
    step()
    for r in range(MAX_ROUNDS):
        nodes = spawn_cubes()
        placed = [0.0] * N_CUBES
        result = None
        while result is None:
            step()
            # Check if all cubes are in the basket
            st = [cube_status(n) for n in nodes]
            size_field=[]
            for i, s in enumerate(st):
                size_field.append(nodes[i].getField("size"))
                match i:
                    case 0:
                        #compare to size/2+basket hight(0.001)
                        if s["in_basket"] and round(s["pos"][2],3)== round(size_field[i].getSFVec3f()[1]/2+0.001,3): 
                            if placed[i] == 0:
                                print(f"CUBE_{i} is in the basket.")
                            placed[i] = placed[i] + 1
                    case 1:
                        if s["in_basket"] and round(s["pos"][2],3)==round(size_field[i].getSFVec3f()[1]/2+size_field[i-1].getSFVec3f()[1]+0.001,3):
                            if placed[i] == 0:
                                print(f"CUBE_{i} is in the basket.")
                            placed[i] = placed[i] + 1
                    case 2:
                        if s["in_basket"] and round(s["pos"][2],3)==round(size_field[i].getSFVec3f()[1]/2+size_field[i-1].getSFVec3f()[1]+size_field[i-2].getSFVec3f()[1]+0.001,3):     #compare big cube size+medium cube size+small cube size/2 
                            if placed[i] == 0:
                                print(f"Cube_{i} is in the basket.")
                            placed[i] = placed[i] + 1                    
                    
            if all(placed):
                result = ("success", "all cubes in basket")
                print(f"Round {r+1}: All cubes are in the basket.")
                break
        clear_cubes(nodes)
except SimEnded:
    pass

supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
