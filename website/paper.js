(() => {
  "use strict";
  const figures = [...document.querySelectorAll("[data-paper-figure]")];
  if (!figures.length) return;
  const revision = new URL(document.currentScript.src).searchParams.get("v");
  let catalogue = null;
  function render() {
    if (!catalogue) return;
    const locale = document.documentElement.lang === "en" ? "en" : "zh";
    for (const figure of figures) {
      const row = catalogue.find(row => row.id === figure.dataset.paperFigure);
      const copy = row.copy[locale];
      const image = figure.querySelector("[data-figure-image]");
      const imageUrl = `${row.image}?v=${row.image_sha256.slice(0, 12)}`;
      image.src = imageUrl; image.width = row.width; image.height = row.height; image.alt = copy.alt;
      const link = figure.querySelector("[data-figure-link]");
      link.href = imageUrl;
      link.setAttribute("aria-label", (locale === "en" ? "Open full-size figure: " : "打开大图：") + copy.title);
      figure.querySelector("[data-figure-hint]").textContent = locale === "en" ? "View full size" : "点击查看大图";
      figure.querySelector("[data-figure-title]").textContent = copy.title;
      figure.querySelector("[data-figure-description]").textContent = copy.description;
      figure.querySelector("[data-figure-note]").textContent = copy.note;
    }
    document.querySelector(".paper-contents").setAttribute("aria-label", locale === "en" ? "Paper contents" : "论文内容目录");
  }
  document.addEventListener("zeta:languagechange", render);
  fetch("paper-figures.json" + (revision ? `?v=${encodeURIComponent(revision)}` : "")).then(response => {
    if (!response.ok) throw Error("Paper figure catalogue unavailable");
    return response.json();
  }).then(value => {
    const ids = new Set(figures.map(figure => figure.dataset.paperFigure));
    if (!Array.isArray(value.figures) || value.figures.length !== figures.length
      || new Set(value.figures.map(row => row.id)).size !== ids.size
      || value.figures.some(row => !ids.has(row.id) || !row.copy?.zh || !row.copy?.en
        || !/^assets\/paper\/[a-z]+\.png$/.test(row.image)
        || !/^[a-f0-9]{64}$/.test(row.image_sha256))) return;
    catalogue = value.figures; render();
  }).catch(() => { /* Static figures and captions remain available. */ });
})();
