import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import numpy as np
import heapq



rows, cols = 10, 10
grid = np.zeros((rows, cols), dtype=int)

obstacles = [(3,3), (3,4), (3,5), (6,7), (7,7), (8,7), (4,6), (5,6), (2,2)]
for obs in obstacles:
    grid[obs] = 1

start = (0, 0)
goal  = (9, 9)

# ---------- A* Search ----------
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    open_heap = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0, start, [start]))
    closed = set()

    while open_heap:
        f, g, current, path = heapq.heappop(open_heap)
        if current == goal:
            return path
        if current in closed:
            continue
        closed.add(current)

        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = current[0]+dr, current[1]+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                new_g = g + 1
                new_path = path + [(nr, nc)]
                heapq.heappush(open_heap, (new_g + heuristic((nr,nc), goal), new_g, (nr,nc), new_path))
    return None

# ---------- DFS Backtracking ----------
def dfs(grid, start, goal):
    stack = [(start, [start])]
    visited = set()

    while stack:
        current, path = stack.pop()
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)

        for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
            nr, nc = current[0]+dr, current[1]+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                stack.append(((nr, nc), path + [(nr, nc)]))
    return None

# ---------- Paths ----------
paths = [astar(grid, start, goal), dfs(grid, start, goal)]
algo_names = ["A*", "Backtracking"]
current_algo = [0]
anim = [None]

# ---------- Visualization ----------
fig, ax = plt.subplots(figsize=(7.5, 7.5))
plt.subplots_adjust(bottom=0.3)

ax.imshow(grid, cmap='Greys', origin='upper')
title = ax.set_title(f"Algorithm: {algo_names[current_algo[0]]}", fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(np.arange(-0.5, cols, 1))
ax.set_yticks(np.arange(-0.5, rows, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True, linewidth=0.5, color='black')

# Start and Goal markers
start_dot = ax.plot(start[1], start[0], 'bo', markersize=10, label='Start')[0]
goal_dot  = ax.plot(goal[1],  goal[0],  'ro', markersize=10, label='Goal')[0]

# Legend (placed below the grid)
legend_elements = [
    start_dot,
    goal_dot,
    plt.Line2D([0], [0], marker='s', color='black', markersize=8, linestyle='None', label='Obstacle'),
    plt.Line2D([0], [0], color='limegreen', lw=3, label='Path')
]
ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.25),
          fontsize=10, ncol=4, frameon=False)

# Trail and car marker (square-shaped)
trail_line, = ax.plot([], [], '-', color='limegreen', linewidth=3)
car_marker, = ax.plot([], [], marker='s', color='orange', markersize=14, markeredgecolor='black')

# Step label above plot
step_label = fig.text(0.5, 0.94, "", ha='center', fontsize=12, color='darkblue', fontweight='bold')

# ---------- Animation Functions ----------
def init():
    trail_line.set_data([], [])
    car_marker.set_data([], [])
    step_label.set_text("")
    return trail_line, car_marker, step_label

def update(frame):
    path = paths[current_algo[0]]
    if path:
        x_coords = [p[1] for p in path]
        y_coords = [p[0] for p in path]

        trail_line.set_data(x_coords[:frame+1], y_coords[:frame+1])
        car_marker.set_data([x_coords[frame]], [y_coords[frame]])
        step_label.set_text(f"Step: {frame + 1} / {len(path)}")
    return trail_line, car_marker, step_label

# ---------- Initial Animation ----------
anim[0] = animation.FuncAnimation(
    fig, update, frames=len(paths[current_algo[0]]),
    init_func=init, interval=300, blit=False, repeat=False
)

# ---------- Button Setup ----------
ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
next_button = Button(ax_button, 'Next')

def on_next(event):
    current_algo[0] = (current_algo[0] + 1) % len(paths)
    title.set_text(f"Algorithm: {algo_names[current_algo[0]]}")
    trail_line.set_data([], [])
    car_marker.set_data([], [])
    step_label.set_text("")

    new_path = paths[current_algo[0]]
    anim[0] = animation.FuncAnimation(
        fig, update, frames=len(new_path),
        init_func=init, interval=300, blit=False, repeat=False
    )
    plt.draw()

next_button.on_clicked(on_next)

plt.show()
