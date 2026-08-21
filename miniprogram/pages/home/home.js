const api = require("../../utils/api");

Page({
  data: {
    loading: true,
    refreshing: false,
    homeView: "overview",
    home: null,
    error: "",
  },

  onLoad() {
    this.loadHome();
  },

  onShow() {
    if (this.hasShown) {
      this.loadHome();
    }
    this.hasShown = true;
  },

  onPullDownRefresh() {
    this.refreshData();
  },

  async loadHome() {
    const requestId = (this.homeRequestId || 0) + 1;
    this.homeRequestId = requestId;
    try {
      const home = await api.get("/api/v1/home");
      if (requestId !== this.homeRequestId) return;
      getApp().globalData.home = home;
      this.setData({ home, loading: false, error: "" });
      const refreshPromise = getApp().globalData.refreshPromise;
      if (refreshPromise && this.followedRefreshPromise !== refreshPromise) {
        this.followedRefreshPromise = refreshPromise;
        refreshPromise.then(() => {
          if (this.data.home) this.loadHome();
        });
      }
    } catch (error) {
      if (requestId !== this.homeRequestId) return;
      this.setData({ loading: false, error: error.message });
    }
  },

  async refreshData() {
    if (this.data.refreshing) return;
    this.setData({ refreshing: true });
    try {
      await getApp().refreshHistory(true);
      await this.loadHome();
      wx.showToast({ title: "数据已检查", icon: "success" });
    } catch (error) {
      wx.showToast({ title: error.message, icon: "none" });
    } finally {
      this.setData({ refreshing: false });
      wx.stopPullDownRefresh();
    }
  },

  openRandom() {
    wx.navigateTo({ url: "/pages/random/random" });
  },

  openModelGenerate() {
    wx.navigateTo({ url: "/pages/model-generate/model-generate" });
  },

  openHistory() {
    wx.navigateTo({ url: "/pages/history/history" });
  },

  openInsight() {
    wx.switchTab({ url: "/pages/insight/insight" });
  },

  selectHomeView(event) {
    this.setData({ homeView: event.currentTarget.dataset.view });
  },

  showInfo(event) {
    const explanations = {
      frontHot: {
        title: "前区高频",
        content: `所选最近 ${this.data.home.quick_stats.window} 期中，前区出现次数最多的5个号码。这里只描述历史频次，不表示下期更容易出现。`,
      },
      backHot: {
        title: "后区高频",
        content: `所选最近 ${this.data.home.quick_stats.window} 期中，后区出现次数最多的3个号码。这里只描述历史频次，不表示下期更容易出现。`,
      },
      averageSum: {
        title: "前区平均和值",
        content: `先把每期前区的5个号码相加。例如03、04、14、28、31的和值是3+4+14+28+31=80。“前区平均和值”就是最近 ${this.data.home.quick_stats.window} 期所有前区和值的平均数。当前标记为“${this.data.home.quick_stats.front_sum_level_label}”，是与全部历史期数的长期平均值比较后得出的。蓝色表示偏低、绿色表示接近、红色表示偏高。它用于观察整体大小，不预测下一期。`,
      },
      averageBackSum: {
        title: "后区平均和值",
        content: `先把每期后区的2个号码相加。例如05、07的和值是5+7=12。“后区平均和值”就是最近 ${this.data.home.quick_stats.window} 期所有后区和值的平均数。当前标记为“${this.data.home.quick_stats.back_sum_level_label}”，是与全部历史期数的长期平均值比较后得出的。蓝色表示偏低、绿色表示接近、红色表示偏高。它用于观察整体大小，不预测下一期。`,
      },
    };
    const info = explanations[event.currentTarget.dataset.info];
    if (info) wx.showModal({ ...info, showCancel: false, confirmText: "知道了" });
  },
});
