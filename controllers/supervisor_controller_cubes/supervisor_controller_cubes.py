# Supervisor controller
# Imports
import numpy as np
from controller import Supervisor

# =============================== PARAMETERS ==============================
SEED = 42
N_CUBES = 5
MAX_ROUNDS = 5                    # total cube sets 
ROUND_TIMEOUT = 300.0             # [s] sim time before a round is declared failed
# cube
COLORS = [[0,0, 0.498039],[0.666667, 0, 0],[0, 0.333333, 0],[1, 0.333333, 1],[1, 1, 0]]     # Blue, Red, Green, Pink, Yellow
CUBE_SIZE = 0.04                  # CUBE_EDGE_LENGTH [m]
CUBE_Z = 0.03                    # spawn height of the cube centre (greater than CUBE_SIZE/2 to avoid collision with the floor)
SPAWN_X_RANGE = (0.20, 0.70)      # Reasonably large arbitrary range       
SPAWN_Y_RANGE = (0.05, 0.55)      # Reasonably large arbitrary range

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
    """Samples a random position for a cube within the spawn area, ensuring it is not too close to the basket."""
    for _ in range(100):  # Try up to 100 times to find a valid position
        x = rng.uniform(*SPAWN_X_RANGE)
        y = rng.uniform(*SPAWN_Y_RANGE)
        p = (x, y, CUBE_Z)
        return p
    raise RuntimeError("Could not find a valid spawn position")

def spawn_cubes():
    positions, nodes = [], []
    for i in range(N_CUBES):
        p = sample_position()
        positions.append(p)
        c = COLORS[i % len(COLORS)]
        node_str = (f'DEF CUBE_{i} cube {{ translation {p[0]} {p[1]} {p[2]} '
                    f'color {c[0]} {c[1]} {c[2]} name "cube_{i}" }}')
        children_field.importMFNodeFromString(-1, node_str)
        nodes.append(supervisor.getFromDef(f"CUBE_{i}"))
    return nodes, positions

# Monitoring
def cube_status(node):
    p = node.getPosition()
    v = node.getVelocity()
    still = np.linalg.norm(v[:3]) < 0.01 and np.linalg.norm(v[3:]) < 0.01
    q = to_local(basket_node, p)
    level = (q[2]<=0.023)
    in_basket = in_basket_xy(p) and still and level
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
        nodes, positions = spawn_cubes()
        placed = [0.0] * N_CUBES
        result = None
        while result is None:
            step()
            # Check if all cubes are in the basket
            st = [cube_status(n) for n in nodes]
            for i, s in enumerate(st):
                if s["in_basket"]:
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