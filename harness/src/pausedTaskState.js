const PAUSED_TASK_STATES = Object.freeze({
  NORMAL_GAMEPLAY: "NORMAL_GAMEPLAY",
  CUTSCENE_OR_TRANSITION: "CUTSCENE_OR_TRANSITION",
  MISSION_FAILED: "MISSION_FAILED",
  RETRY_MENU: "RETRY_MENU",
  QUIT_CONFIRMATION: "QUIT_CONFIRMATION",
  MISSION_SUCCESS: "MISSION_SUCCESS",
  UNKNOWN: "UNKNOWN"
});

function resultText(result = {}) {
  return [
    result.text,
    ...(Array.isArray(result.matches?.failure) ? result.matches.failure : []),
    ...(Array.isArray(result.matches?.success) ? result.matches.success : [])
  ].filter(Boolean).join(" ").toLowerCase();
}

function classifyPausedTaskState(result = {}) {
  const text = resultText(result);
  if (/(是否确定.*退出|确定要退出|退出任务将导致|are you sure.*(?:quit|exit))/i.test(text)) {
    return PAUSED_TASK_STATES.QUIT_CONFIRMATION;
  }
  if (Object.values(PAUSED_TASK_STATES).includes(result.state)) {
    return result.state;
  }
  if (result.signal === "failure" || result.failure === true) {
    return PAUSED_TASK_STATES.MISSION_FAILED;
  }
  if (result.signal === "success" || result.success === true) {
    return PAUSED_TASK_STATES.MISSION_SUCCESS;
  }
  if (/(mission failed|任务失败|任务失败了|失败)/i.test(text)) {
    return PAUSED_TASK_STATES.MISSION_FAILED;
  }
  if (/(retry|重试|重新开始)/i.test(text)) {
    return PAUSED_TASK_STATES.RETRY_MENU;
  }
  if (/(quit confirmation|确认退出|are you sure)/i.test(text)) {
    return PAUSED_TASK_STATES.QUIT_CONFIRMATION;
  }
  if (/(cutscene|过场|剧情|loading|加载中)/i.test(text)) {
    return PAUSED_TASK_STATES.CUTSCENE_OR_TRANSITION;
  }
  return PAUSED_TASK_STATES.UNKNOWN;
}

function isRecoveryTaskState(state) {
  return [
    PAUSED_TASK_STATES.MISSION_FAILED,
    PAUSED_TASK_STATES.RETRY_MENU,
    PAUSED_TASK_STATES.QUIT_CONFIRMATION,
    PAUSED_TASK_STATES.UNKNOWN
  ].includes(state);
}

module.exports = {
  PAUSED_TASK_STATES,
  classifyPausedTaskState,
  isRecoveryTaskState
};
