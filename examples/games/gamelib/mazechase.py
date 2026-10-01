# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Maze chase: the engine and the BFS teacher.

A grid maze with a pellet in every open cell and two or three chasers.
Each tick the player moves one cell (up, down, left or right; into a wall it stays and the tick still passes),
then every chaser whose turn it is moves one cell.
Eating every pellet wins; a chaser on the player's cell, or a swap of cells with one, loses.
A game is a pure function of its seed and the moves played: the layout uses one random stream and the chasers another.

The chasers differ in speed and manner: the hunter goes straight for the player and misses a turn one tick in four;
the wanderer and the drifter move one tick in four, mostly at random and now and then toward the player.
`MazeView` is everything a player sees; the teacher reads it and nothing else.
"""

import random
from collections import deque
from dataclasses import dataclass

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ORDER = ("up", "down", "left", "right")
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
PELLET_SCORE = 10
RECENT = 5
KINDS = ("hunter", "wanderer", "drifter")
# which ticks each kind of chaser moves on (tick % 4)
SCHEDULE = {"hunter": "1110", "wanderer": "1000", "drifter": "0010"}


@dataclass(frozen=True)
class MazeConfig:
    width: int = 15  # odd: passages sit on odd rows and columns
    height: int = 9
    chasers: int = 2  # 1 to 3 of KINDS, in that order
    braid: float = 1.0  # the share of dead ends opened into loops
    loops: int = 3  # further walls between passages knocked through
    hunter: str = "1110"  # the hunter's moves over four ticks (1 moves, 0 waits): "1010" is half speed
    max_ticks: int = 300


@dataclass(frozen=True)
class MazeView:
    """A snapshot: the walls, who is where, what is left, and how the game stands."""

    grid: tuple  # rows of '#' and ' '
    player: tuple
    chasers: tuple  # ((row, col, kind), ...)
    pellets: frozenset
    score: int
    tick: int
    recent: tuple  # the last moves, oldest first
    status: str  # playing, won, lost, timeout
    max_ticks: int
    hunter: str = "1110"  # the hunter's pace, which the teacher is told

    @property
    def legal(self):
        return legal_moves(self.grid, self.player)


def legal_moves(grid, pos):
    r, c = pos
    return [d for d in ORDER if grid[r + DIRS[d][0]][c + DIRS[d][1]] != "#"]


def neighbours(grid, pos):
    r, c = pos
    return [(d, (r + DIRS[d][0], c + DIRS[d][1])) for d in legal_moves(grid, pos)]


def distances(grid, sources):
    """Steps from the nearest source to every reachable open cell."""
    dist = {s: 0 for s in sources}
    queue = deque(sources)
    while queue:
        cell = queue.popleft()
        for _, nxt in neighbours(grid, cell):
            if nxt not in dist:
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
    return dist


def make_layout(seed, cfg):
    """(grid rows, the player's start, the chasers' starts): a depth-first maze with its dead ends opened into loops."""
    if cfg.width % 2 == 0 or cfg.height % 2 == 0 or cfg.width < 5 or cfg.height < 5:
        raise ValueError("width and height must be odd and at least 5")
    rng = random.Random(f"maze:{seed}")
    h, w = cfg.height, cfg.width
    grid = [["#"] * w for _ in range(h)]
    cells = [(r, c) for r in range(1, h, 2) for c in range(1, w, 2)]
    start = rng.choice(cells)
    grid[start[0]][start[1]] = " "
    stack = [start]
    while stack:
        r, c = stack[-1]
        options = [
            (r + dr, c + dc)
            for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2))
            if 0 < r + dr < h and 0 < c + dc < w and grid[r + dr][c + dc] == "#"
        ]
        if not options:
            stack.pop()
            continue
        nr, nc = rng.choice(options)
        grid[(r + nr) // 2][(c + nc) // 2] = " "
        grid[nr][nc] = " "
        stack.append((nr, nc))
    for r, c in cells:
        walls = [d for d in ORDER if grid[r + DIRS[d][0]][c + DIRS[d][1]] == "#"]
        if len(walls) == 3 and rng.random() < cfg.braid:  # a dead end: knock through to a neighbouring passage
            inside = [d for d in walls if 0 < r + 2 * DIRS[d][0] < h and 0 < c + 2 * DIRS[d][1] < w]
            d = rng.choice(inside)
            grid[r + DIRS[d][0]][c + DIRS[d][1]] = " "
    inner = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if (r % 2) != (c % 2) and grid[r][c] == "#"  # a wall segment between two passage cells
    ]
    for r, c in rng.sample(inner, min(cfg.loops, len(inner))):
        grid[r][c] = " "
    rows = tuple("".join(row) for row in grid)
    player = (h - 2, 1 + 2 * ((w // 2) // 2))  # the bottom row, near the middle
    far = sorted(distances(rows, [player]).items(), key=lambda kv: (-kv[1], kv[0]))
    starts = []
    for cell, _ in far:
        if all(abs(cell[0] - s[0]) + abs(cell[1] - s[1]) >= 4 for s in starts):
            starts.append(cell)
        if len(starts) == cfg.chasers:
            break
    return rows, player, tuple(starts)


class Chaser:
    def __init__(self, kind, pos):
        self.kind, self.pos, self.prev = kind, pos, None


class MazeChase:
    """One game. `step(direction)` plays a tick and returns what happened: {"ate", "bumped", "caught"}."""

    def __init__(self, seed, config=MazeConfig()):
        self.seed, self.config = seed, config
        self.grid, self.player, starts = make_layout(seed, config)
        self.chasers = [Chaser(KINDS[i], pos) for i, pos in enumerate(starts)]
        self.pellets = {
            (r, c) for r, row in enumerate(self.grid) for c, ch in enumerate(row) if ch == " " and (r, c) != self.player
        }
        self.rng = random.Random(f"chasers:{seed}")
        self.score, self.tick, self.status = 0, 0, "playing"
        self.moves = []

    @property
    def over(self):
        return self.status != "playing"

    def pace(self, kind):
        return self.config.hunter if kind == "hunter" else SCHEDULE[kind]

    def view(self):
        return MazeView(
            grid=self.grid,
            player=self.player,
            chasers=tuple((*c.pos, c.kind) for c in self.chasers),
            pellets=frozenset(self.pellets),
            score=self.score,
            tick=self.tick,
            recent=tuple(self.moves[-RECENT:]),
            status=self.status,
            max_ticks=self.config.max_ticks,
            hunter=self.config.hunter,
        )

    def _chaser_next(self, chaser):
        options = neighbours(self.grid, chaser.pos)
        if chaser.kind != "hunter" and self.rng.random() < 0.8:
            forward = [o for o in options if o[1] != chaser.prev] or options  # it turns back only at a dead end
            return self.rng.choice(forward)[1]
        dist = distances(self.grid, [self.player])
        return min(options, key=lambda o: (dist.get(o[1], 10**6), ORDER.index(o[0])))[1]

    def step(self, direction):
        if self.over:
            raise ValueError("the game is over")
        if direction not in DIRS:
            raise ValueError(f"not a direction: {direction!r}")
        old = self.player
        nxt = (old[0] + DIRS[direction][0], old[1] + DIRS[direction][1])
        bumped = self.grid[nxt[0]][nxt[1]] == "#"
        if not bumped:
            self.player = nxt
        self.moves.append(direction)
        self.tick += 1
        ate = self.player in self.pellets
        if ate:
            self.pellets.discard(self.player)
            self.score += PELLET_SCORE
        caught = any(c.pos == self.player for c in self.chasers)
        if not caught and not self.pellets:
            self.status = "won"
            return {"ate": ate, "bumped": bumped, "caught": False}
        if not caught:
            for chaser in self.chasers:
                if self.pace(chaser.kind)[(self.tick - 1) % 4] != "1":
                    continue
                before = chaser.pos
                chaser.pos, chaser.prev = self._chaser_next(chaser), before
                if chaser.pos == self.player or (chaser.pos == old and before == self.player):
                    caught = True
        if caught:
            self.status = "lost"
        elif self.tick >= self.config.max_ticks:
            self.status = "timeout"
        return {"ate": ate, "bumped": bumped, "caught": caught}


def turns_taken(kind, tick, t, hunter="1110"):
    """How many moves a chaser of this kind has made when the player has made `t` more moves, from tick `tick`."""
    pace = hunter if kind == "hunter" else SCHEDULE[kind]
    return sum(pace[(tick + i) % 4] == "1" for i in range(t))


def teacher_move(view):
    """The BFS teacher: the first step of the shortest path to the nearest pellet it can reach with room to spare, and
    when there is none, to the safest place it can reach.

    The rules are known, so the search is exact about how fast a chaser can be: a cell the player reaches on its `t`-th
    move is safe if no chaser could be standing on it by then, given the distance it has to cover and the ticks it
    moves on (every chaser is assumed to come straight at it, the worst case). The slack of a cell is how many moves
    the nearest chaser would still need. A breadth-first search through the safe cells gives each its `t` and the first
    move of its shortest path; the target is the nearest pellet with slack 2 or more, or with none the cell with the
    most slack. Ties between directions go in the order up, down, left, right."""
    grid, start, inf = view.grid, view.player, 10**6
    fields = [(distances(grid, [(r, c)]), kind) for r, c, kind in view.chasers]

    def slack(cell, t):
        return min((d.get(cell, inf) - turns_taken(kind, view.tick, t, view.hunter) for d, kind in fields), default=99)

    reach = {start: (0, None)}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        t, first = reach[cell]
        for d, nxt in neighbours(grid, cell):
            if nxt not in reach and slack(nxt, t + 1) > 0:
                reach[nxt] = (t + 1, first or d)
                queue.append(nxt)
    order = sorted((t, cell) for cell, (t, _) in reach.items() if cell != start)
    for t, cell in order:
        if cell in view.pellets and slack(cell, t) >= 2:
            return reach[cell][1]
    if order:
        return reach[max(order, key=lambda tc: (slack(tc[1], tc[0]), -tc[0], tc[1]))[1]][1]
    options = neighbours(grid, start)
    return max(options, key=lambda o: (min((d.get(o[1], inf) for d, _ in fields), default=inf), -ORDER.index(o[0])))[0]


def random_move(view, rng, legal_only=False):
    return rng.choice(view.legal if legal_only else list(ORDER))
