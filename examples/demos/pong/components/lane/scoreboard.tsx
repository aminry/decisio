'use client';

/**
 * The score, above the middle of the court.
 *
 * Big enough to read from across the room and mono so the digits never shuffle
 * (design/SPEC.md, must-not-get-wrong e). The two sides are split by the same
 * dashed hairline the court draws down its own centre, which is what ties the
 * scoreboard to the game under it.
 *
 * The names are the players', in the case they typed them: this is a cabinet
 * with two people on it, not a table of categories, so they are not set in
 * small caps like a provider tag.
 *
 * It is never orange: orange is the first lane's ball, trail and latency number
 * bar, and nothing else.
 */

export interface ScoreboardProps {
  /** Left paddle: the human. */
  you: { name: string; score: number };
  /** Right paddle: the model. */
  them: { name: string; score: number };
  size?: 'play' | 'watch';
}

const SIZES = {
  play: {
    number: 'text-[48px] lg:text-[68px]',
    name: 'text-[13px] lg:text-[15px]',
    rule: 'h-9 lg:h-12',
  },
  watch: {
    number: 'text-[40px] lg:text-[56px]',
    name: 'text-[13px] lg:text-[14px]',
    rule: 'h-8 lg:h-10',
  },
} as const;

export function Scoreboard({ you, them, size = 'play' }: ScoreboardProps) {
  const s = SIZES[size];
  return (
    <div className="flex w-full items-center justify-center gap-4 lg:gap-7">
      <Side name={you.name} score={you.score} align="right" sizes={s} />
      <span aria-hidden className={`border-court-line ${s.rule} border-l border-dashed`} />
      <Side name={them.name} score={them.score} align="left" sizes={s} />
    </div>
  );
}

function Side({
  name,
  score,
  align,
  sizes,
}: {
  name: string;
  score: number;
  align: 'left' | 'right';
  sizes: (typeof SIZES)[keyof typeof SIZES];
}) {
  return (
    <div
      className={`flex min-w-0 flex-1 items-baseline gap-2.5 lg:gap-4 ${
        align === 'right' ? 'justify-end' : 'flex-row-reverse justify-end'
      }`}
    >
      <span className={`text-fg-muted min-w-0 truncate font-medium ${sizes.name}`}>{name}</span>
      <span className={`mono text-fg ${sizes.number} leading-none font-medium tracking-[-0.03em]`}>
        {score}
      </span>
    </div>
  );
}

export default Scoreboard;
