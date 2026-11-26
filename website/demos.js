(() => {
  "use strict";
  const grid = document.getElementById("demo-grid");
  if (!grid) return;
  const cards = [...grid.querySelectorAll("[data-demo-id]")];
  const filters = [...document.querySelectorAll("[data-demo-filter]")];
  const dialog = document.getElementById("demo-dialog");
  const video = document.getElementById("demo-video");
  const featuredVideo = document.getElementById("featured-video");
  const counter = document.getElementById("demo-counter");
  let selectedFilter = "all", current = null, manifest = null;
  const lang = () => document.documentElement.lang === "en" ? "en" : "zh";
  const text = (zh, en) => lang() === "en" ? en : zh;
  const findDemo = id => manifest?.featured?.id === id ? manifest.featured : manifest?.clips.find(clip => clip.id === id);
  function applyFilter(value) {
    selectedFilter = value;
    filters.forEach(button => button.setAttribute("aria-pressed", String(button.dataset.demoFilter === value)));
    let visible = 0;
    cards.forEach(card => { card.hidden = value !== "all" && card.dataset.category !== value; if (!card.hidden) visible++; });
    grid.classList.toggle("filtered", value !== "all");
    counter.textContent = text(`${visible} 段演示`, `${visible} demos`);
  }
  filters.forEach(button => button.addEventListener("click", () => applyFilter(button.dataset.demoFilter)));
  function updateDialog() {
    if (!current) return;
    const row = findDemo(current);
    document.getElementById("demo-dialog-title").textContent = row.title[lang()];
    document.getElementById("demo-description").textContent = row.description[lang()];
    document.getElementById("demo-close").textContent = text("关闭", "Close");
    document.getElementById("demo-previous").textContent = text("上一段", "Previous");
    document.getElementById("demo-next").textContent = text("下一段", "Next");
    video.setAttribute("aria-label", row.title[lang()]);
  }
  function openDemo(id) {
    const row = findDemo(id);
    if (!row || typeof dialog.showModal !== "function") return false;
    featuredVideo?.pause();
    video.pause(); current = id; updateDialog();
    video.poster = row.poster;
    video.src = row.video;
    if (!dialog.open) dialog.showModal();
    video.load();
    const play = video.play();
    if (play?.catch) play.catch(() => { /* Native controls remain available. */ });
    return true;
  }
  document.querySelectorAll("a[data-demo]").forEach(link => link.addEventListener("click", event => {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (openDemo(link.dataset.demo)) event.preventDefault();
  }));
  document.getElementById("demo-close").addEventListener("click", () => dialog.close());
  featuredVideo?.addEventListener("play", () => { if (dialog.open) dialog.close(); });
  dialog.addEventListener("close", () => {
    video.pause(); video.removeAttribute("src"); video.load(); current = null;
  });
  dialog.addEventListener("click", event => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  });
  function step(amount) {
    const featured = manifest?.featured ? [manifest.featured.id] : [];
    const list = [...featured, ...cards.filter(card => !card.hidden).map(card => card.dataset.demoId)];
    const choices = list.includes(current) ? list : [...featured, ...cards.map(card => card.dataset.demoId)];
    const index = choices.indexOf(current);
    if (index >= 0) openDemo(choices[(index + amount + choices.length) % choices.length]);
  }
  document.getElementById("demo-previous").addEventListener("click", () => step(-1));
  document.getElementById("demo-next").addEventListener("click", () => step(1));
  document.addEventListener("zeta:languagechange", () => { applyFilter(selectedFilter); updateDialog(); });
  // Links work as ordinary local MP4 links until enhancement is ready, including
  // file:// use where browser security can prevent fetching the JSON manifest.
  fetch("demos.json").then(response => {
    if (!response.ok) throw Error("Demo manifest unavailable");
    return response.json();
  }).then(value => {
    if (!Array.isArray(value.clips)) return;
    const rows = value.featured ? [value.featured, ...value.clips] : value.clips;
    if (rows.some(row => !/^assets\/demos\/[a-z]+\.mp4$/.test(row.video) || !/^assets\/demos\/[a-z]+\.jpg$/.test(row.poster))) return;
    manifest = value;
  }).catch(() => { /* Direct video links are the complete fallback. */ });
  applyFilter(selectedFilter);
})();
