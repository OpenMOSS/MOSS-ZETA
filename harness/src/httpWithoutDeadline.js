const http = require("node:http");
const https = require("node:https");

// No implicit headers/body deadline; the caller's AbortSignal remains authoritative.
function httpWithoutDeadline(url, options = {}) {
  return new Promise((resolve, reject) => {
    const protocol = new URL(url).protocol;
    if (!["http:", "https:"].includes(protocol)) return reject(new Error("Unsupported HTTP protocol"));
    const transport = protocol === "https:" ? https : http;
    const request = transport.request(url, {
      method: options.method || "GET", headers: options.headers,
      signal: options.signal, agent: false
    }, (response) => {
      const body = new Promise((resolveBody, rejectBody) => {
        const chunks = [];
        response.on("data", (chunk) => chunks.push(chunk));
        response.on("end", () => resolveBody(Buffer.concat(chunks).toString("utf8")));
        response.on("error", rejectBody);
      });
      // A cancelled body can reject before the caller starts awaiting text().
      void body.catch(() => {});
      resolve({ ok: response.statusCode >= 200 && response.statusCode < 300,
        status: response.statusCode, statusText: response.statusMessage,
        text: () => body });
    });
    request.setTimeout(0);
    request.on("error", reject);
    request.end(options.body);
  });
}

module.exports = { httpWithoutDeadline };
