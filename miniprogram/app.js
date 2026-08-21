const api = require("./utils/api");

App({
  globalData: {
    home: null,
    refreshPromise: null,
    lastRefreshAt: 0,
    launched: false,
  },

  onLaunch() {
    this.refreshHistory();
  },

  onShow() {
    if (this.globalData.launched) {
      this.refreshHistory();
    } else {
      this.globalData.launched = true;
    }
  },

  refreshHistory(force = false) {
    const now = Date.now();
    if (!force && this.globalData.refreshPromise && now - this.globalData.lastRefreshAt < 60000) {
      return this.globalData.refreshPromise;
    }
    this.globalData.lastRefreshAt = now;
    this.globalData.refreshPromise = api.post("/api/v1/refresh", {}).catch(() => null);
    return this.globalData.refreshPromise;
  },
});
