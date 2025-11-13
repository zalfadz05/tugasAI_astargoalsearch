# =========================================
# Algoritma Pencarian Berbasis Tujuan (A*)
# =========================================

import heapq
import matplotlib.pyplot as plt
import numpy as np

# Representasi peta (0 = jalan bebas, 1 = halangan)
grid = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
]

# Titik awal dan tujuan
start = (0, 0)
goal = (4, 4)

# Fungsi heuristik (Manhattan Distance)
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Fungsi utama algoritma A*
def astar_search(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_list = []
    heapq.heappush(open_list, (0 + heuristic(start, goal), 0, start, [start]))
    visited = set()

    while open_list:
        f, g, current, path = heapq.heappop(open_list)

        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return path, g

        # Gerakan ke atas, bawah, kiri, kanan
        directions = [(-1,0), (1,0), (0,-1), (0,1)]

        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy

            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                neighbor = (nx, ny)
                new_cost = g + 1
                f = new_cost + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f, new_cost, neighbor, path + [neighbor]))

    return None, None

# Jalankan pencarian
path, cost = astar_search(grid, start, goal)

# Tampilkan hasil di terminal
if path:
    print("✅ Jalur ditemukan:")
    print(path)
    print("Total langkah:", cost)
else:
    print("❌ Tidak ada jalur ditemukan.")

# -----------------------------
# VISUALISASI HASIL
# -----------------------------
grid_np = np.array(grid)

plt.figure(figsize=(6,6))
plt.imshow(grid_np, cmap='binary')

# Gambar jalur hasil A*
if path:
    for (x, y) in path:
        plt.scatter(y, x, c='cyan', s=200, marker='o')

# Gambar titik start & goal
plt.scatter(start[1], start[0], c='green', s=300, marker='*', label='Start')
plt.scatter(goal[1], goal[0], c='red', s=300, marker='*', label='Goal')

plt.title("A* Path Planning with heapq", fontsize=14)
plt.legend()
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()
