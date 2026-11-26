const { spawnSync } = require("node:child_process");

function commandExists(command) {
  const shellCommand = process.platform === "win32" ? `where ${command}` : `command -v ${command}`;
  const result = spawnSync(process.platform === "win32" ? "cmd.exe" : "sh", [process.platform === "win32" ? "/c" : "-lc", shellCommand], {
    encoding: "utf8",
    timeout: 3000
  });
  return result.status === 0;
}

function isWaylandScreenshotBackend(backend) {
  return backend === "grim" || backend === "gnome-screenshot" || backend === "spectacle";
}

function isPillowScreenshotBackend(backend) {
  return backend === "python-pillow";
}

function screenshotBackendInstallHint(backend) {
  if (backend === "grim") {
    return "Install grim for wlroots/Hyprland/Sway Wayland screenshots, or set AGI_GTA_LINUX_SCREENSHOT=gnome-screenshot, spectacle, or x11 for another desktop backend.";
  }

  if (backend === "gnome-screenshot") {
    return "Install gnome-screenshot for GNOME Wayland screenshots, or set AGI_GTA_LINUX_SCREENSHOT=grim, spectacle, or x11 for another desktop backend.";
  }

  if (backend === "spectacle") {
    return "Install spectacle/kde-spectacle for KDE/Plasma Wayland screenshots, or set AGI_GTA_LINUX_SCREENSHOT=grim, gnome-screenshot, or x11 for another desktop backend.";
  }

  if (backend === "python-pillow") {
    return "Install Python 3 with Pillow ImageGrab support, or set AGI_GTA_LINUX_SCREENSHOT=x11 and install ImageMagick import.";
  }

  return "Set AGI_GTA_LINUX_SCREENSHOT=grim, gnome-screenshot, spectacle, x11, python-pillow, or screenshot-desktop.";
}

function assertLinuxScreenshotBackendAvailable(backend, exists = commandExists) {
  if (isPillowScreenshotBackend(backend)) {
    if (exists("python3")) {
      return;
    }

    throw new Error(`Linux screenshot backend "${backend}" is selected, but command "python3" was not found. ${screenshotBackendInstallHint(backend)}`);
  }

  if (!isWaylandScreenshotBackend(backend)) {
    return;
  }

  if (exists(backend)) {
    return;
  }

  throw new Error(`Linux Wayland screenshot backend "${backend}" is selected, but command "${backend}" was not found. ${screenshotBackendInstallHint(backend)}`);
}

function isWaylandSession(env = process.env) {
  const sessionType = String(env.XDG_SESSION_TYPE || "").trim().toLowerCase();
  if (sessionType === "wayland") {
    return true;
  }
  if (sessionType === "x11") {
    return false;
  }

  return Boolean(env.WAYLAND_DISPLAY);
}

function selectLinuxScreenshotBackend(env = process.env, exists = commandExists) {
  const requested = String(env.AGI_GTA_LINUX_SCREENSHOT || "").trim().toLowerCase();
  const waylandSession = isWaylandSession(env);
  const desktop = `${env.XDG_CURRENT_DESKTOP || ""} ${env.DESKTOP_SESSION || ""}`;

  if (requested === "x11" || requested === "screenshot-desktop") {
    return "screenshot-desktop";
  }

  if (requested === "python-pillow" || requested === "pillow") {
    return "python-pillow";
  }

  if (requested === "grim" || requested === "gnome-screenshot" || requested === "spectacle") {
    return requested;
  }

  if (requested) {
    throw new Error(`Unsupported AGI_GTA_LINUX_SCREENSHOT: ${requested}. Expected grim, gnome-screenshot, spectacle, x11, python-pillow, or screenshot-desktop.`);
  }

  if (waylandSession) {
    if (/gnome/i.test(desktop)) {
      return "gnome-screenshot";
    }
    if (/(kde|plasma)/i.test(desktop)) {
      return "spectacle";
    }
    if (exists("grim")) {
      return "grim";
    }
    if (exists("gnome-screenshot")) {
      return "gnome-screenshot";
    }
    if (exists("spectacle")) {
      return "spectacle";
    }
    return "grim";
  }

  return "screenshot-desktop";
}

module.exports = {
  assertLinuxScreenshotBackendAvailable,
  commandExists,
  isPillowScreenshotBackend,
  isWaylandScreenshotBackend,
  isWaylandSession,
  selectLinuxScreenshotBackend
};
