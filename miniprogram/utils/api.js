const { API_BASE_URL, WECHAT_AUTH_ENABLED } = require("./config");
let authPromise = null;
const responseCache = new Map();
const inFlightGets = new Map();

function clientId() {
  const key = "dlt_client_id";
  let value = wx.getStorageSync(key);
  if (!value) {
    value = `wx-${Date.now()}-${Math.random().toString(36).slice(2, 12)}`;
    wx.setStorageSync(key, value);
  }
  return value;
}

function ensureAuth() {
  if (!WECHAT_AUTH_ENABLED) return Promise.resolve(null);
  const session = wx.getStorageSync("dlt_auth_session");
  if (session?.token && new Date(session.expires_at).getTime() > Date.now() + 60000) {
    return Promise.resolve(session.token);
  }
  if (authPromise) return authPromise;
  authPromise = new Promise((resolve, reject) => {
    wx.login({
      success(login) {
        if (!login.code) {
          reject(new Error("微信登录失败"));
          return;
        }
        wx.request({
          url: `${API_BASE_URL}/api/v1/auth/wechat`,
          method: "POST",
          data: { code: login.code },
          header: { "content-type": "application/json" },
          success(response) {
            if (response.statusCode >= 200 && response.statusCode < 300) {
              wx.setStorageSync("dlt_auth_session", response.data);
              resolve(response.data.token);
              return;
            }
            reject(new Error(response.data?.error || "微信登录失败"));
          },
          fail(error) {
            reject(new Error(error.errMsg || "微信登录失败"));
          },
        });
      },
      fail(error) {
        reject(new Error(error.errMsg || "微信登录失败"));
      },
    });
  }).finally(() => {
    authPromise = null;
  });
  return authPromise;
}

async function request(path, options = {}) {
  const token = await ensureAuth();
  return new Promise((resolve, reject) => {
    wx.request({
      url: `${API_BASE_URL}${path}`,
      method: options.method || "GET",
      data: options.data || {},
      timeout: options.timeout || 20000,
      header: {
        "content-type": "application/json",
        "X-Client-Id": clientId(),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      success(response) {
        if (response.statusCode >= 200 && response.statusCode < 300) {
          resolve(response.data);
          return;
        }
        if (response.statusCode === 401 && WECHAT_AUTH_ENABLED) {
          wx.removeStorageSync("dlt_auth_session");
        }
        reject(new Error(response.data?.error || `请求失败（${response.statusCode}）`));
      },
      fail(error) {
        reject(new Error(error.errMsg || "网络连接失败"));
      },
    });
  });
}

function cacheDuration(path) {
  if (path === "/api/v1/about") return 60 * 60 * 1000;
  if (path === "/api/v1/home") return 10 * 1000;
  if (path.startsWith("/api/v1/statistics/")) return 30 * 1000;
  if (path.startsWith("/api/v1/draws")) return 8 * 1000;
  if (path === "/api/v1/lab/overview") return 3 * 1000;
  return 0;
}

function cloneData(value) {
  return value === undefined ? value : JSON.parse(JSON.stringify(value));
}

function cachedGet(path) {
  const duration = cacheDuration(path);
  const cached = responseCache.get(path);
  if (duration && cached && Date.now() - cached.createdAt < duration) {
    return Promise.resolve(cloneData(cached.value));
  }
  if (inFlightGets.has(path)) return inFlightGets.get(path);
  const promise = request(path)
    .then((value) => {
      if (duration) responseCache.set(path, { value: cloneData(value), createdAt: Date.now() });
      return value;
    })
    .finally(() => inFlightGets.delete(path));
  inFlightGets.set(path, promise);
  return promise;
}

function clearCache() {
  responseCache.clear();
}

function invalidateForMutation(path) {
  if (path.includes("/refresh")) {
    clearCache();
    return;
  }
  if (path.startsWith("/api/v1/lab/")) {
    responseCache.delete("/api/v1/lab/overview");
  }
}

module.exports = {
  get(path) {
    return cachedGet(path);
  },
  post(path, data) {
    return request(path, { method: "POST", data }).then((value) => {
      invalidateForMutation(path);
      return value;
    });
  },
  delete(path) {
    return request(path, { method: "DELETE" }).then((value) => {
      invalidateForMutation(path);
      return value;
    });
  },
  clearCache,
};
