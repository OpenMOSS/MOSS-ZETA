(() => {
  "use strict";
  const en = {
    "navDemos": "Demos",
    "demoFilterLabel": "Filter recorded demos",
    "demoAll": "All",
    "demoLand": "Driving & movement",
    "demoWater": "Water",
    "skip": "Skip to content",
    "navigation": "Main navigation",
    "navApproach": "Approach",
    "thinkingBranches": "Reasoning branches",
    "steadyTitle": "Keep executing",
    "steadySubtitle": "Empty think · Keep current state",
    "smallTitle": "Update locally",
    "smallSubtitle": "Small think · Revise the judgment",
    "brainModeTitle": "Plan the next stage",
    "brainSubtitle": "Major update · Call Brain",
    "illustrationBadge": "Protocol illustration",
    "copy": "Copy",
    "stateEffectLabel": "STATE EFFECT",
    "exampleNote": "Illustrative responses for continuation, local updates and Brain calls.",
    "backTop": "Back to top",
    "projectSubtitle": "Towards Real-time AGI in the Frontier AAA Game with Zeroth Embodied Task Agent",
    "projectSummary": "Powered by MOSS-ZETA-preview, ZETA connects real-time visual control, selective reasoning and persistent memory to pursue long-horizon tasks in GTA V.",
    "projectCode": "Code on GitHub",
    "projectDemos": "Gameplay demos",
    "featuredVideoLabel": "ZETA project demonstration, 69 seconds with audio",
    "featuredFallback": "Your browser does not support video playback.",
    "projectVideoCaption": "ZETA: perception, reasoning and control",
    "newDemosTitle": "ZETA in action",
    "newDemosIntro": "From pursuits and firefights to tunnel riding and a dive into the sea.",
    "abstractTitle": "Real-time control. Long-horizon tasks.",
    "abstractOne": "Steering needs immediate corrections, while the destination may have been set minutes earlier. ZETA connects these time scales: a visual Policy provides fast gamepad control as System 1, while an external Brain handles stage planning and task memory as System 2.",
    "abstractTwo": "Initialized from Qwen3.5-9B, the Policy learns control through action pretraining, then learns when to continue, revise a judgment or call Brain through instruction tuning. Dual-route inference reduces context-switch waiting, the 200ms scheduler places actions in their execution windows, and the Harness maintains task state throughout interaction.",
    "protocolExplore": "Explore the three reasoning branches and output examples",
    "implementationTitle": "Code and implementation",
    "implementationIntro": "The Harness, inference infrastructure and 200ms scheduler are maintained together in one repository.",
    "implementationRepository": "GitHub repository",
    "implementationRepositoryDetail": "Explore the ZETA implementation and setup",
    "implementationInfraDetail": "Model serving and dual-route inference",
    "implementationHarnessDetail": "Task state, memory and execution handoff",
    "demoCombat": "Combat",
    "categoryLand": "Driving & movement",
    "categoryWater": "Water",
    "categoryCombat": "Combat",
    "demoEscapeTitle": "Escape the police",
    "demoEscapeDescription": "A continuous escape from a two-star pursuit until the wanted level clears and the objective changes to returning the car.",
    "demoEscapeAlt": "Escape the police",
    "demoSniperTitle": "Track and snipe a target",
    "demoSniperDescription": "Track a distant moving target through the scope, adjust aim and fire.",
    "demoSniperAlt": "Track and snipe a target",
    "demoTunnelTitle": "Ride through a four-star pursuit",
    "demoTunnelAlt": "Ride through a four-star pursuit",
    "demoTunnelDescription": "Ride through curved pipes, a construction area and a metro station during a four-star pursuit.",
    "demoDiveTitle": "Dive from the pier",
    "demoDiveAlt": "Dive from the pier",
    "demoDiveDescription": "Jump from the pier, swim underwater, then return to the surface.",
    "demoChaseTitle": "Vault a fence in pursuit",
    "demoChaseAlt": "Vault a fence in pursuit",
    "demoChaseDescription": "Chase a target into a yard, vault the fence and continue across the rooftops.",
    "demoAircombatTitle": "Fire from a helicopter",
    "demoAircombatAlt": "Fire from a helicopter",
    "demoAircombatDescription": "Track a pursuing helicopter from the side door and keep firing until it catches fire.",
    "demoHillchaseTitle": "Chase a plane across the hills",
    "demoHillchaseAlt": "Chase a plane across the hills",
    "demoHillchaseDescription": "Follow a smoking aircraft across the hills, descend steep slopes and jump a roadside barrier.",
    "demoDemolitionTitle": "Pull down a hillside structure",
    "demoDemolitionAlt": "Pull down a hillside structure",
    "demoDemolitionDescription": "Tow a hillside structure with a pickup, followed by its visible collapse.",
    "paperStoryTitle": "Inside the ZETA paper",
    "paperStoryIntro": "From model and data to the inference and execution system behind sustained interaction.",
    "tocFramework": "Framework",
    "tocData": "Data",
    "tocTraining": "Training",
    "tocHarness": "Harness",
    "tocRuntime": "Real-time inference",
    "frameworkHeading": "Coordinating decisions across three time scales",
    "frameworkOne": "The Policy predicts short gamepad-control segments from the current image and recent observation–response history. The same autoregressive model decides whether to retain its judgment, revise a local plan or ask Brain to reconsider the current stage.",
    "frameworkTwo": "The stage instruction specifies what to accomplish, local reasoning interprets how to proceed now, and task memory preserves confirmed progress, facts and constraints. Updating them at different time scales connects short control steps to a persistent task context.",
    "dataHeading": "From gameplay recordings to aligned training samples",
    "dataOne": "Action pretraining uses approximately <strong>4,000 hours</strong> of human GTA V gamepad recordings: roughly 2,000 hours each of free play and task-directed gameplay. Instruction tuning annotates approximately <strong>1,200 hours</strong> selected from the task-directed portion of the same corpus. Hours refer to original recordings before augmentation.",
    "dataTwo": "Video frames and recorded controller events are aligned on one timeline, then combined with stage instructions, local reasoning and task memory. The current Policy input includes the full frame, same-frame minimap, available OCR, recent history and the effective task state.",
    "trainingHeading": "Two-stage training for actions and update decisions",
    "trainingOne": "The Policy is initialized from Qwen3.5-9B. Stage I adapts it to gamepad control through visual–action sequences. Stage II fine-tunes the same Policy to condition actions on objectives, local judgments and progress, while learning when to continue, update reasoning or request Brain. Brain and OCR supply planning and visual-text context to the Policy.",
    "trainingTwo": "Action pretraining uses 20-round windows. Instruction tuning uses windows of up to 16 rounds and retains the latest eight when advancing. Only selected current assistant responses receive target supervision; task memory, supplied state and retained history provide context. Even the closing tag of an empty think is supervised, making continuation an explicit model decision.",
    "trainingThree": "Training applies repeated-action loss decay with γ=0.5 to consecutive equal ground-truth whole actions, while reasoning and Brain payloads retain unit weight. Sampling-phase augmentation and expert continuations after neutral-action history address action transitions and recovery of useful control.",
    "harnessHeading": "Integrating context updates with ongoing control",
    "harnessOne": "A nonempty local think supplies the Policy’s new judgment; Brain refreshes memory while retaining the stage instruction. Periodic refresh can update both memory and reasoning. In the handoff configuration, the active service continues control while a standby service prepares an independent session for the revised context.",
    "harnessTwo": "The new context becomes active only after the standby produces a complete legal action and confirms continuation-prefix readiness. A full Brain call first stops submissions, settles outstanding actions, releases inputs and confirms that the game is paused; it then reviews all three state fields and resumes from a fresh observation after committing the result.",
    "runtimeHeading": "Dual-route inference on a 200ms control clock",
    "runtimeOne": "While one TP4 route generates the current response, the other prepares the next known prefix. Preparation excludes unfinished responses and future observations. Reuse requires matching tokens and ordered images; window trims reconstruct the selected 16/8 history exactly. The two default routes fit on one eight-H200 node.",
    "runtimeTwo": "With the paper’s delay-aware offset, an observation at t conditions controls for (t+200ms, t+400ms]. Early states wait for the start, late actions retain the original window end, and expired states are discarded. The 200ms cadence is the target observation/control schedule; preparation, generation and execution retain their own dependencies.",
    "efficiencyHeading": "Reducing model, image-processing and scheduling overhead",
    "efficiencyOne": "The serving path combines compiled execution, CUDA Graphs, shared image features, streamlined multimodal token processing and asynchronous scheduling. The plot below shows independent component-removal controls and TP scaling at a shared Graph512 limit.",
    "efficiencyResults": "At the shared Graph512 limit, mean request latencies for TP1, TP2, TP4 and TP8 are <strong>244.7, 196.6, 179.4 and 178.7 ms</strong>, respectively. Expanding TP4 to Graph8192 reduces its mean to <strong>168.9 ms</strong>.",
    "figureConditions": "Experimental settings and timing scope",
    "navPaperStory": "Paper",
    "projectReport": "Paper & method"
};
  const originals = new Map();
  document.querySelectorAll("[data-i18n]").forEach(el => originals.set(el, el.innerHTML));
  const attributeOriginals = new Map();
  for (const attribute of ["aria", "alt"]) {
    document.querySelectorAll(`[data-i18n-${attribute}]`).forEach(el => {
      const name = attribute === "aria" ? "aria-label" : "alt";
      attributeOriginals.set(el, {name, value: el.getAttribute(name), key: el.dataset[attribute === "aria" ? "i18nAria" : "i18nAlt"]});
    });
  }
  let language = "zh", mode = "steady", toastTimer;
  try { language = localStorage.getItem("zeta-language") === "en" ? "en" : "zh"; } catch (_) { /* Storage is optional. */ }
  const queryLanguage = new URLSearchParams(location.search).get("lang");
  if (["zh", "en"].includes(queryLanguage)) language = queryLanguage;
  const action = "<a>+0+8+0+0A;A;A;A;A;A</a>";
  const modes = {
    steady: {
      label: "EMPTY THINK",
      zh: {title: "没有新判断，继续执行。", description: "空 think 表示沿用当前判断，并继续输出手柄动作。", thought: "", effects: "当前阶段目标、任务记忆和局部判断保持不变。"},
      en: {title: "No new judgment. Keep executing.", description: "An empty think retains the current judgment and continues gamepad control.", thought: "", effects: "Retain the current stage instruction, task memory and local reasoning."}
    },
    small: {
      label: "LOCAL UPDATE",
      zh: {title: "新的线索，更新判断与记忆。", description: "Policy 生成新的局部判断，并请求 Brain 刷新任务记忆；当前阶段目标保持不变，更新可以与持续控制并行准备。", thought: "前方路径仍可通行，沿当前路线继续接近目标。", effects: "局部判断来自 Policy，任务记忆来自 Brain；通过运行时交接后共同生效，阶段目标不变。"},
      en: {title: "New evidence. Refresh reasoning and memory.", description: "Policy proposes a local judgment and requests a task-memory refresh from Brain. The stage instruction remains fixed while preparation can overlap ongoing control.", thought: "The path ahead remains open; continue along this route toward the goal.", effects: "Local reasoning comes from Policy and task memory from Brain. They take effect through runtime handoff while the stage instruction stays fixed."}
    },
    brain: {
      label: "BRAIN CALL",
      zh: {title: "阶段变化，请求更高层规划。", description: "专用标记发起完整 Brain 审视：运行时暂停执行，Brain 核对证据后返回阶段目标、任务记忆与初始局部判断。", thought: '当前阶段已完成，需要确认下一阶段目标。<|call_brain|>{"instruction":"前往下一处任务地点","small_think":"先确认目标位置，再选择路线。"}', effects: "Brain 返回阶段目标、任务记忆与局部判断，提交新状态后恢复控制。"},
      en: {title: "A stage change. Request higher-level planning.", description: "The marker requests a full Brain review. The harness pauses execution while Brain checks the evidence and returns the stage instruction, task memory and initial local reasoning.", thought: 'The current stage is complete; confirm the next goal.<|call_brain|>{"instruction":"Reach the next task location","small_think":"Confirm the destination, then choose a route."}', effects: "Brain returns the stage instruction, task memory and local reasoning; control resumes after committing the updated context."}
    }
  };

  function renderMode() {
    const panel = document.getElementById("thinking-panel");
    if (!panel) return;
    const content = modes[mode][language];
    document.getElementById("branch-label").textContent = modes[mode].label;
    document.getElementById("branch-title").textContent = content.title;
    document.getElementById("branch-description").textContent = content.description;
    document.getElementById("branch-code").textContent = `<think>${content.thought}</think>${mode === "brain" ? "<a>+0+0+0+0;;;;;</a>" : action}`;
    document.getElementById("branch-effects").textContent = content.effects;
    panel.setAttribute("aria-labelledby", `tab-${mode}`);
    document.querySelectorAll("[data-mode]").forEach(button => {
      const selected = button.dataset.mode === mode;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
    });
  }
  function publicUrl(value) {
    if (typeof value !== "string" || !value) return null;
    try { const url = new URL(value); return url.protocol === "https:" ? url : null; } catch (_) { return null; }
  }
  function applyPublicLinks() {
    const config = window.ZETA_CONFIG || {};
    const repo = publicUrl(config.repositoryUrl);
    const codeLink = document.getElementById("code-resource");
    if (repo && ["github.com", "www.github.com"].includes(repo.hostname) && /^\/[^/]+\/[^/]+\/?$/.test(repo.pathname) && codeLink) {
      codeLink.href = repo.href;
      codeLink.querySelector("strong").textContent = language === "en" ? "Source on GitHub" : "GitHub 源代码";
      codeLink.querySelector("small").textContent = language === "en" ? "Explore the repository" : "查看仓库与实现";
    }
    const paper = publicUrl(config.paperUrl);
    if (paper) document.querySelectorAll("[data-public-paper]").forEach(link => { link.href = paper.href; });
  }
  function setLanguage(next) {
    language = next;
    document.documentElement.lang = next === "en" ? "en" : "zh-CN";
    originals.forEach((original, el) => {
      el.innerHTML = next === "en" ? (en[el.dataset.i18n] ?? original) : original;
    });
    attributeOriginals.forEach(({name, value, key}, el) => el.setAttribute(name, next === "en" ? (en[key] ?? value) : value));
    document.querySelectorAll("[data-locale]").forEach(el => { el.hidden = el.dataset.locale !== next; });
    const toggle = document.getElementById("language-toggle");
    if (toggle) { toggle.textContent = next === "en" ? "中文" : "EN"; toggle.setAttribute("aria-label", next === "en" ? "切换到中文" : "Switch to English"); }
    document.title = "ZETA — Zeroth Embodied Task Agent";
    const description = next === "en" ? "Visual decisions, adaptive reasoning, persistent memory and gamepad control in open-world games." : "ZETA：探索开放世界游戏中的视觉决策、分层思考、持续记忆与手柄控制。";
    document.querySelector('meta[name="description"]')?.setAttribute("content", description);
    try { localStorage.setItem("zeta-language", next); } catch (_) { /* No storage required. */ }
    renderMode();
    applyPublicLinks();
    document.dispatchEvent(new CustomEvent("zeta:languagechange"));
  }
  document.getElementById("language-toggle")?.addEventListener("click", () => setLanguage(language === "zh" ? "en" : "zh"));
  const tabs = [...document.querySelectorAll("[data-mode]")];
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => { mode = tab.dataset.mode; renderMode(); });
    tab.addEventListener("keydown", event => {
      let next;
      if (["ArrowDown", "ArrowRight"].includes(event.key)) next = (index + 1) % tabs.length;
      if (["ArrowUp", "ArrowLeft"].includes(event.key)) next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      mode = tabs[next].dataset.mode; renderMode(); tabs[next].focus();
    });
  });
  function notify(message) {
    const toast = document.getElementById("toast");
    if (!toast) return;
    clearTimeout(toastTimer); toast.textContent = message; toast.classList.add("visible");
    toastTimer = setTimeout(() => toast.classList.remove("visible"), 2500);
  }
  document.getElementById("copy-example")?.addEventListener("click", async () => {
    const code = document.getElementById("branch-code");
    try {
      await navigator.clipboard.writeText(code.textContent);
      notify(language === "en" ? "Example copied" : "示例已复制");
    } catch (_) {
      const selection = window.getSelection(); const range = document.createRange();
      range.selectNodeContents(code); selection.removeAllRanges(); selection.addRange(range);
      notify(language === "en" ? "Text selected — use your system copy command" : "已选中文本，可使用系统复制快捷键");
    }
  });
  const tablist = document.querySelector('.mode-list[role="tablist"]');
  if (tablist) {
    const mobile = window.matchMedia("(max-width: 760px)");
    const orient = () => tablist.setAttribute("aria-orientation", mobile.matches ? "horizontal" : "vertical");
    orient(); mobile.addEventListener("change", orient);
  }
  setLanguage(language);
})();
