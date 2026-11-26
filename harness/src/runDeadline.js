// A wall-clock run budget, independent of Policy responses and gameplay pauses.
function startRunDeadline(agent, durationMs, { setTimer = setTimeout, clearTimer = clearTimeout } = {}) {
  if (!Number.isInteger(durationMs) || durationMs < 0 || durationMs > 2147483647) {
    throw new Error('runDurationMs must be an integer in [0, 2147483647]');
  }
  if (durationMs === 0) return () => {};
  let active = true;
  const timer = setTimer(() => {
    if (!active) return;
    active = false;
    if (!agent.running || agent.stopRequested) return;
    agent.runDurationLimitReached = true;
    agent.stop('run_duration_limit');
  }, durationMs);
  timer.unref?.();
  return () => {
    active = false;
    clearTimer(timer);
  };
}

module.exports = { startRunDeadline };
