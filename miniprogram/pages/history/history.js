const api = require("../../utils/api");

Page({
  data: {
    query: "",
    rows: [],
    total: 0,
    loading: false,
    hasMore: true,
  },

  onLoad() {
    this.loadRows(true);
  },

  onShow() {
    if (!this.hasShown) {
      this.hasShown = true;
      return;
    }
    this.loadRows(true);
  },

  onPullDownRefresh() {
    this.loadRows(true).finally(wx.stopPullDownRefresh);
  },

  onReachBottom() {
    this.loadRows(false);
  },

  onUnload() {
    clearTimeout(this.searchTimer);
    this.destroyed = true;
  },

  onSearchInput(event) {
    this.setData({ query: event.detail.value });
    clearTimeout(this.searchTimer);
    this.searchTimer = setTimeout(() => this.loadRows(true), 300);
  },

  clearSearch() {
    this.setData({ query: "" });
    this.loadRows(true);
  },

  async loadRows(reset) {
    if ((!reset && this.data.loading) || (!reset && !this.data.hasMore)) return;
    const requestId = (this.rowsRequestId || 0) + 1;
    this.rowsRequestId = requestId;
    this.setData({ loading: true });
    const offset = reset ? 0 : this.data.rows.length;
    try {
      const query = encodeURIComponent(this.data.query.trim());
      const data = await api.get(`/api/v1/draws?limit=30&offset=${offset}&q=${query}`);
      if (this.destroyed || requestId !== this.rowsRequestId) return;
      this.setData({
        rows: reset ? data.rows : this.data.rows.concat(data.rows),
        total: data.total,
        hasMore: data.has_more,
      });
    } catch (error) {
      if (requestId !== this.rowsRequestId) return;
      wx.showToast({ title: error.message, icon: "none" });
    } finally {
      if (!this.destroyed && requestId === this.rowsRequestId) this.setData({ loading: false });
    }
  },
});
