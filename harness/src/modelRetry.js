function retryableModelError(error) {
  const status = Number(error?.status);
  if (status >= 400 && status < 500 && ![408, 425, 429].includes(status)) return false;
  return error?.modelProtocolError === true || [408, 425, 429].includes(status) || status >= 500
    || /timed?\s*out|timeout|fetch failed|network|socket|econn|enotfound|eai_again|connection reset|temporar|did not contain message content/i
      .test([error?.name, error?.message, error?.code].filter(Boolean).join(" "));
}

function retryDelay(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason || new Error("request aborted"));
    const done = () => { signal?.removeEventListener("abort", abort); resolve(); };
    const timer = setTimeout(done, ms);
    const abort = () => { clearTimeout(timer); signal.removeEventListener("abort", abort); reject(signal.reason || new Error("request aborted")); };
    signal?.addEventListener("abort", abort, { once: true });
  });
}

module.exports = { retryableModelError, retryDelay };
