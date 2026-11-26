const PAUSE_MENU_MARKERS = Object.freeze([
  "地图",
  "简讯",
  "统计",
  "设置",
  "游戏",
  "在线"
]);

function compactOcrText(ocr) {
  return (ocr?.results || [])
    .map((result) => String(result?.text || ""))
    .join(" ")
    .replace(/\s+/g, "");
}

function detectGtaPauseMenu(ocr, { requirePositiveHeader = false } = {}) {
  const text = compactOcrText(ocr);
  const rawText = (ocr?.results || []).map((result) => String(result?.text || "")).join(" ");
  const chineseMarkers = PAUSE_MENU_MARKERS.filter((marker) => text.includes(marker));
  // Keep word boundaries: ordinary words containing "map"/"game" are not menu tabs.
  const englishMarkers = ["MAP", "BRIEF", "STATS", "SETTINGS", "GAME", "ONLINE"]
    .filter((marker) => new RegExp(`\\b${marker}\\b`, "i").test(rawText));
  const englishEvidence = /Grand\s*Theft\s*Auto\s*V/i.test(rawText) || /Hide\s+Menu/i.test(rawText);
  const matchedMarkers = [...chineseMarkers, ...(englishEvidence ? englishMarkers : [])];
  const hasMenuFooter = text.includes("隐藏菜单")
    || text.includes("RockstarGames账户")
    || text.includes("返回")
    || /Hide\s+Menu/i.test(rawText);
  const hasSpecificMenuFooter = text.includes("隐藏菜单")
    || text.includes("RockstarGames账户")
    || /Hide\s+Menu/i.test(rawText);
  // GTA's cutscene pause layer has no tabs. Require the paired, trailing
  // pause-menu/back footer, rather than treating an arbitrary "back" as pause.
  const pausedCutscene = /暂停菜单[^\u4e00-\u9fff]{0,12}返回[^\u4e00-\u9fff]{0,8}$/.test(text)
    || /\bPAUSE\s+MENU\b[^\n]{0,20}\bBACK\b(?:\s+[A-Z])?\s*$/i.test(rawText);
  const menuOpen = pausedCutscene || (requirePositiveHeader
      ? matchedMarkers.length >= 3 || (matchedMarkers.length >= 2 && (hasMenuFooter || /GrandTheftAutoV/i.test(text)))
      : matchedMarkers.length >= 2 || (hasMenuFooter && matchedMarkers.length >= 1));
  return {
    pausedCutscene,
    menuOpen,
    // An opening/closing menu can expose its footer before its title/tabs.
    // Missing positive header evidence is not proof that the game has resumed.
    conclusive: Boolean(ocr?.ok) && (menuOpen || !hasSpecificMenuFooter),
    matchedMarkers,
    hasMenuFooter,
    text
  };
}

module.exports = {
  PAUSE_MENU_MARKERS,
  compactOcrText,
  detectGtaPauseMenu
};
