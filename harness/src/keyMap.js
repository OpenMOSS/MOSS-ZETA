const NAMED_KEYS = {
  backspace: 0x08,
  tab: 0x09,
  enter: 0x0d,
  return: 0x0d,
  shift: 0x10,
  ctrl: 0x11,
  control: 0x11,
  alt: 0x12,
  pause: 0x13,
  capslock: 0x14,
  caps: 0x14,
  escape: 0x1b,
  esc: 0x1b,
  space: 0x20,
  pageup: 0x21,
  pagedown: 0x22,
  end: 0x23,
  home: 0x24,
  left: 0x25,
  up: 0x26,
  right: 0x27,
  down: 0x28,
  insert: 0x2d,
  delete: 0x2e,
  del: 0x2e,
  meta: 0x5b,
  win: 0x5b,
  windows: 0x5b,
  apps: 0x5d,
  numpad0: 0x60,
  numpad1: 0x61,
  numpad2: 0x62,
  numpad3: 0x63,
  numpad4: 0x64,
  numpad5: 0x65,
  numpad6: 0x66,
  numpad7: 0x67,
  numpad8: 0x68,
  numpad9: 0x69,
  multiply: 0x6a,
  add: 0x6b,
  subtract: 0x6d,
  decimal: 0x6e,
  divide: 0x6f,
  numlock: 0x90,
  scrolllock: 0x91,
  semicolon: 0xba,
  quote: 0xde,
  apostrophe: 0xde,
  comma: 0xbc,
  period: 0xbe,
  slash: 0xbf,
  backslash: 0xdc,
  bracketleft: 0xdb,
  bracketright: 0xdd,
  leftbracket: 0xdb,
  rightbracket: 0xdd,
  minus: 0xbd,
  equal: 0xbb,
  backquote: 0xc0,
  grave: 0xc0
};

for (let i = 1; i <= 12; i += 1) {
  NAMED_KEYS[`f${i}`] = 0x6f + i;
}

function normalizeKeyName(key) {
  if (typeof key !== "string") {
    throw new Error("key must be a string");
  }

  return key.trim().toLowerCase().replace(/\s+/g, "");
}

function keyToVirtualKey(key) {
  const normalized = normalizeKeyName(key);

  if (/^[a-z]$/.test(normalized)) {
    return normalized.toUpperCase().charCodeAt(0);
  }

  if (/^[0-9]$/.test(normalized)) {
    return normalized.charCodeAt(0);
  }

  if (Object.prototype.hasOwnProperty.call(NAMED_KEYS, normalized)) {
    return NAMED_KEYS[normalized];
  }

  throw new Error(`Unsupported key: ${key}`);
}

module.exports = {
  keyToVirtualKey,
  normalizeKeyName,
  SUPPORTED_KEY_NAMES: Object.keys(NAMED_KEYS).sort()
};
