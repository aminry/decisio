/**
 * Site-level strings that are not part of the game contract.
 *
 * Changed from upstream (ably-labs/jev-pong, Apache-2.0): the sponsor, gateway and vendor strings are removed. The
 * page names the lanes it was recorded with, from the recording itself.
 */

export const SITE_NAME = 'Pong lanes';

export const UPSTREAM_URL = 'https://github.com/ably-labs/jev-pong';

/** The one line that describes the whole thing. */
export const TAGLINE = 'Pong where the ball moves one step per model decision. Slow model, slow ball.';

export const CREDITS = 'System One request format · AI SDK evaluate';

/** The window the results strip counts decisions over, in replay seconds. */
export const WINDOW_SECONDS = 12;
