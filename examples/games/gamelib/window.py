# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""The live window: pygame shows the frames the renderer draws, and turns the keyboard into moves for a human player.

pygame is imported here and nowhere else, so everything but the window runs without it (`pip install pygame-ce`).
"""

import time

from .hangman import ALPHABET
from .render import HEIGHT, WIDTH

MAZE_KEYS = {
    "up": ("UP", "w"),
    "down": ("DOWN", "s"),
    "left": ("LEFT", "a"),
    "right": ("RIGHT", "d"),
}


class Window:
    def __init__(self, title, scale=2):
        try:
            import pygame
        except ImportError:
            raise SystemExit("the live window needs pygame: pip install pygame-ce (or run with --headless)") from None
        self.pg = pygame
        pygame.init()
        self.scale = scale
        self.screen = pygame.display.set_mode((WIDTH * scale, HEIGHT * scale))
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()

    def show(self, image):
        surface = self.pg.image.frombuffer(image.tobytes(), image.size, "RGB")
        self.screen.blit(surface, (0, 0))
        self.pg.display.flip()

    def poll(self):
        """The keys pressed since the last call, as names ('left', 'a', 'space', 'escape', ...), and whether to quit."""
        pg, keys, quit_ = self.pg, [], False
        for event in pg.event.get():
            if event.type == pg.QUIT:
                quit_ = True
            elif event.type == pg.KEYDOWN:
                keys.append(pg.key.name(event.key))
        return keys, quit_

    def wait(self, ms):
        """Sleep `ms` while keeping the window responsive; returns (keys, quit) collected meanwhile."""
        end, keys, quit_ = time.perf_counter() + ms / 1000, [], False
        while time.perf_counter() < end:
            got, q = self.poll()
            keys += got
            quit_ = quit_ or q
            self.clock.tick(60)
        return keys, quit_

    def close(self):
        self.pg.quit()


def maze_direction(keys):
    """The direction a list of key names asks for (the last one wins), or None."""
    asked = None
    for key in keys:
        for direction, names in MAZE_KEYS.items():
            if key in {n.lower() for n in names}:
                asked = direction
    return asked


def hangman_letter(keys):
    """The first letter typed, or None."""
    return next((k for k in keys if len(k) == 1 and k in ALPHABET), None)
