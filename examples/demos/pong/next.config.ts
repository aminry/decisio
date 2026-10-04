import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  // The dev indicator would sit on top of a recording; the agent rule files are not part of this demo.
  devIndicators: false,
  agentRules: false,
};

export default nextConfig;
