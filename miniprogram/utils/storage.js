const FAVORITES_KEY = "dlt_favorite_groups_v2";
const LEGACY_KEY = "dlt_random_favorites";
const MAX_GROUPS = 100;
const CLOUD_MIGRATED_KEY = "dlt_favorites_cloud_migrated";
const CLOUD_PENDING_KEY = "dlt_favorites_cloud_pending";
const api = require("./api");

function getFavorites() {
  const current = wx.getStorageSync(FAVORITES_KEY);
  if (Array.isArray(current)) return current;

  const legacy = wx.getStorageSync(LEGACY_KEY) || [];
  const migrated = legacy.map((item) => ({
    id: item.id || `${Date.now()}-${Math.random()}`,
    source: "random",
    sourceLabel: "随机生成",
    createdAt: item.createdAt || "",
    generatedAfterIssue: null,
    tickets: normalizeTickets(item.tickets || []),
    details: null,
    legacy: true,
  }));
  wx.setStorageSync(FAVORITES_KEY, migrated);
  return migrated;
}

function saveFavorite(input) {
  const payload = Array.isArray(input) ? { source: "random", tickets: input } : input;
  const favorite = {
    id: `${Date.now()}-${Math.floor(Math.random() * 10000)}`,
    source: payload.source || "random",
    sourceLabel: payload.sourceLabel || sourceLabel(payload.source),
    createdAt: payload.createdAt || formatTime(new Date()),
    generatedAfterIssue: payload.generatedAfterIssue ? String(payload.generatedAfterIssue) : null,
    tickets: normalizeTickets(payload.tickets || []),
    details: payload.details || null,
    legacy: false,
  };
  const favorites = [favorite, ...getFavorites()].slice(0, MAX_GROUPS);
  wx.setStorageSync(FAVORITES_KEY, favorites);
  markCloudPending(favorite.id, true);
  api.post("/api/v1/favorites", favorite)
    .then(() => markCloudPending(favorite.id, false))
    .catch(() => null);
  return favorite;
}

function removeFavorite(id) {
  const favorites = getFavorites().filter((item) => item.id !== id);
  wx.setStorageSync(FAVORITES_KEY, favorites);
  markCloudPending(id, false);
  api.delete(`/api/v1/favorites/${encodeURIComponent(id)}`).catch(() => null);
  return favorites;
}

async function syncFavorites() {
  const local = getFavorites();
  const migrated = Boolean(wx.getStorageSync(CLOUD_MIGRATED_KEY));
  const pending = new Set(wx.getStorageSync(CLOUD_PENDING_KEY) || []);
  const uploadRows = migrated ? local.filter((item) => pending.has(String(item.id))) : local;
  const uploads = await Promise.all(uploadRows.map((item) => (
    api.post("/api/v1/favorites", item)
      .then(() => {
        markCloudPending(item.id, false);
        return true;
      })
      .catch(() => false)
  )));
  if (!migrated && uploads.every(Boolean)) wx.setStorageSync(CLOUD_MIGRATED_KEY, true);
  try {
    const response = await api.get("/api/v1/favorites?limit=100");
    const merged = new Map();
    (response.groups || []).forEach((item) => merged.set(String(item.id), item));
    local.forEach((item) => merged.set(String(item.id), item));
    const favorites = Array.from(merged.values())
      .sort((a, b) => String(b.createdAt || "").localeCompare(String(a.createdAt || "")))
      .slice(0, MAX_GROUPS);
    wx.setStorageSync(FAVORITES_KEY, favorites);
    return favorites;
  } catch (error) {
    return local;
  }
}

function markCloudPending(id, pending) {
  const values = new Set((wx.getStorageSync(CLOUD_PENDING_KEY) || []).map(String));
  if (pending) values.add(String(id));
  else values.delete(String(id));
  wx.setStorageSync(CLOUD_PENDING_KEY, Array.from(values));
}

function normalizeTickets(tickets) {
  return tickets.map((ticket, index) => {
    const front = normalizeNumbers(ticket.front || ticket.front_text);
    const back = normalizeNumbers(ticket.back || ticket.back_text);
    return {
      id: ticket.id || index + 1,
      front,
      back,
      front_text: front.map(padNumber).join(" "),
      back_text: back.map(padNumber).join(" "),
    };
  });
}

function normalizeNumbers(value) {
  if (Array.isArray(value)) return value.map(Number);
  return String(value || "")
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map(Number);
}

function sourceLabel(source) {
  if (source === "quick_model") return "模型生成";
  if (source === "custom_model") return "自定义模型";
  return "随机生成";
}

function padNumber(value) {
  return String(value).padStart(2, "0");
}

function formatTime(date) {
  const pad = (value) => String(value).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

module.exports = {
  getFavorites,
  saveFavorite,
  removeFavorite,
  syncFavorites,
};
