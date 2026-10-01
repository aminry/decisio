# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the decisio project
"""Two small games for watching a decision model play, and the teachers that label them.

mazechase and hangman are the engines (deterministic under a seed; the entropy solver uses numpy);
render draws their frames with Pillow; window shows them live with pygame.
"""
