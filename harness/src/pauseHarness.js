class PauseHarness {
  constructor(options = {}) {
    this.sessionState = options.sessionState;
    this.recordEvent = typeof options.recordEvent === "function" ? options.recordEvent : null;
  }

  transition(phase, gamePaused, metadata = {}) {
    this.sessionState.setPauseState(phase, gamePaused);
    if (this.recordEvent) {
      void this.recordEvent("pause_harness_transition", {
        phase,
        gamePaused: Boolean(gamePaused),
        ...metadata
      });
    }
  }

  getStatus() {
    return { ...this.sessionState.pauseHarness };
  }
}

module.exports = {
  PauseHarness
};
