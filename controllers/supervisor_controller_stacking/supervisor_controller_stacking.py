''' Supervisor controller

Supervisor, which spawns 3 different sized blue cubes randomly within a 0.5x0.5 m wide square. It monitors which cubes are in the basket, and it checks if they are on top of eachother in the proper, decreasing size order. If all conditions are met it respawns the cubes up to MAX_ROUNDS times.

'''
# Imports
import numpy as np
import math
from controller import Supervisor

# =============================== PARAMETERS ==============================
SEED = 42
N_CUBES = 3
MAX_ROUNDS = 5                    # total cube sets
ROUND_TIMEOUT = 300.0             # [s] sim time before a round is declared failed
MIN_CUBE_SPACING=0.09             # ~math.hypot(0.0375,0.0375)+math.hypot(0.025,0.025) 
# cube
CUBE_Z = 0.05                     # spawn height of the cube centre (greater than CUBE_SIZE/2 to avoid collision with the floor)
SPAWN_X_RANGE = (0.20, 0.70)      # Reasonably large arbitrary range       
SPAWN_Y_RANGE = (0.05, 0.55)      # Reasonably large arbitrary range
SIZE_RANGE_BIG = (0.06,0.075)     # big cube size range
SIZE_RANGE_MEDIUM = (0.035,0.05)  # medium cube size range
SIZE_RANGE_SMALL = (0.015,0.025)   # small cube size range
Z_TOL= 0.003                       # 2 mm vertical settling tolerance

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
BASKET_TOP_Z = basket_node.getPosition()[2] + 0.0005  # center 0.001 + half-thickness 0.0005

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
def sample_position(taken):
    """Samples a random position for a cube within the spawn area, ensuring it is not too close to another cube."""
    for _ in range(100):  # Try up to 100 times to find a valid position
        x = rng.uniform(*SPAWN_X_RANGE)
        y = rng.uniform(*SPAWN_Y_RANGE)
        p = (x, y, CUBE_Z)
        if math.hypot(x,y)>0.8:
            continue
        if any(math.hypot(p[0] - q[0], p[1] - q[1]) < MIN_CUBE_SPACING for q in taken):
            continue
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
    """Spawns N_CUBES number of cubes of different sizes"""
    positions, nodes = [],[]
    s = sample_size()
    for i in range(N_CUBES):
        p = sample_position(positions)
        positions.append(p)
        node_str = (f'DEF CUBE_{i} cube {{ translation {p[0]} {p[1]} {p[2]} '
                    f'color {0} {0} {0.498039} size {s[i]} {s[i]} {s[i]} name "cube_{i}" }}')
        children_field.importMFNodeFromString(-1, node_str)
        nodes.append(supervisor.getFromDef(f"CUBE_{i}"))
    return nodes, positions, s

# Monitoring
def cube_status(node):
    """Checks if cube is in the basket and is still"""
    p = node.getPosition()
    v = node.getVelocity()
    still = np.linalg.norm(v[:3]) < 0.01 and np.linalg.norm(v[3:]) < 0.01
    in_basket = in_basket_xy(p) and still
    return {"pos": p, "in_basket": in_basket}

def is_stacked_on(upper_pos, upper_size, lower_pos, lower_size):
    """Checks both XY overlap and expected Z height of upper cube on lower cube."""
    xy_ok = (
        abs(upper_pos[0] - lower_pos[0]) <= lower_size / 2
        and abs(upper_pos[1] - lower_pos[1]) <= lower_size / 2
    )
    expected_z = lower_pos[2] + lower_size / 2 + upper_size / 2
    z_ok = abs(upper_pos[2] - expected_z) <= Z_TOL
    return xy_ok and z_ok

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
        nodes, positions, sizes = spawn_cubes()
        placed = [0.0] * N_CUBES
        result = None
        round_start=supervisor.getTime()
        while result is None:
            step()
            # Check if all cubes are in the basket
            st = [cube_status(n) for n in nodes]

            # Cube 0 (big): resting flat on the basket floor
            c0_z_ok = abs(st[0]["pos"][2] - (BASKET_TOP_Z + sizes[0] / 2)) <= Z_TOL
            placed[0] = st[0]["in_basket"] and c0_z_ok

            # Cube 1 (medium): stacked on Cube 0
            placed[1] = placed[0] and st[1]["in_basket"] and is_stacked_on(
                st[1]["pos"], sizes[1], st[0]["pos"], sizes[0]
            )

            # Cube 2 (small): stacked on Cube 1
            placed[2] = placed[1] and st[2]["in_basket"] and is_stacked_on(
                st[2]["pos"], sizes[2], st[1]["pos"], sizes[1]
            )                  
                    
            if all(placed):
                result = ("success", "all cubes in basket")
                print(f"Round {r+1}: All cubes are in the basket.")
                break
            if supervisor.getTime() - round_start > ROUND_TIMEOUT:
                result=("timeout", "round timed out")
                print(f"Round {r+1}: Timed out.")
                break
        clear_cubes(nodes)
except SimEnded:
    pass

supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)
