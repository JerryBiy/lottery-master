const api = require("../../utils/api");
const storage = require("../../utils/storage");

Page({
  data: {
    favorites: [],
    visibleFavorites: [],
    displayLimit: 8,
    evaluating: false,
    about: null,
  },

  onLoad() {
    this.loadAbout();
  },

  onShow() {
    this.loadFavorites();
  },

  onUnload() {
    this.destroyed = true;
  },

  async loadFavorites() {
    const requestId = (this.favoriteRequestId || 0) + 1;
    this.favoriteRequestId = requestId;
    const local = storage.getFavorites();
    const initial = local.map((item) => this.prepareFavorite(item, null));
    this.setData({ favorites: initial, evaluating: Boolean(local.length) }, () => this.refreshVisible());
    const stored = await storage.syncFavorites();
    if (this.destroyed || requestId !== this.favoriteRequestId) return;
    const localIds = local.map((item) => String(item.id)).join(",");
    const storedIds = stored.map((item) => String(item.id)).join(",");
    if (storedIds !== localIds) {
      const synced = stored.map((item) => this.prepareFavorite(item, null));
      this.setData({ favorites: synced, evaluating: Boolean(stored.length) }, () => this.refreshVisible());
    }
    if (!stored.length) return;

    try {
      const response = await api.post("/api/v1/favorites/evaluate", {
        groups: stored.map((item) => ({
          id: item.id,
          generatedAfterIssue: item.generatedAfterIssue,
          tickets: item.tickets,
        })),
      });
      if (this.destroyed || requestId !== this.favoriteRequestId) return;
      const evaluations = new Map(response.evaluations.map((item) => [String(item.id), item]));
      const favorites = stored.map((item) => this.prepareFavorite(item, evaluations.get(String(item.id))));
      this.setData({ favorites, evaluating: false }, () => this.refreshVisible());
    } catch (error) {
      if (!this.destroyed && requestId === this.favoriteRequestId) this.setData({ evaluating: false });
    }
  },

  prepareFavorite(item, evaluation) {
    const status = evaluation ? evaluation.status : "loading";
    const evaluated = status === "evaluated";
    const target = evaluated ? evaluation.target : null;
    const resultTickets = evaluated ? evaluation.tickets : [];
    const tickets = item.tickets.map((ticket, index) => {
      const result = resultTickets[index] || {};
      const frontMatches = result.front_matches || [];
      const backMatches = result.back_matches || [];
      return {
        ...ticket,
        frontBalls: ticket.front.map((number) => ({ text: this.padNumber(number), hit: frontMatches.includes(number) })),
        backBalls: ticket.back.map((number) => ({ text: this.padNumber(number), hit: backMatches.includes(number) })),
        frontHits: result.front_hits || 0,
        backHits: result.back_hits || 0,
        prizeLabel: result.prize_level ? result.prize_label : "",
      };
    });
    return {
      ...item,
      tickets,
      expanded: false,
      detailsOpen: false,
      isModel: item.source === "quick_model" || item.source === "custom_model",
      sourceClass: item.source || "random",
      status,
      statusLabel: this.statusLabel(status, evaluation),
      evaluated,
      targetIssue: target ? target.issue : "",
      targetDate: target ? target.date : "",
      targetFront: target ? target.front.map((number) => this.padNumber(number)) : [],
      targetBack: target ? target.back.map((number) => this.padNumber(number)) : [],
      bestLabel: evaluated ? evaluation.best_label : "",
      hasPrize: evaluated && Boolean(evaluation.best_prize),
      detailsRows: this.detailRows(item.details),
    };
  },

  statusLabel(status, evaluation) {
    if (status === "loading") return "正在检查开奖";
    if (status === "pending") return "等待下一期开奖";
    if (status === "untracked") return "旧收藏，待手动核对";
    if (evaluation && evaluation.best_prize) return evaluation.best_label;
    return "未中奖";
  },

  detailRows(details) {
    if (!details) return [];
    const rows = [
      { label: "基础模型", value: details.modelLabel || details.model || "-" },
      { label: "学习范围", value: details.trainWindow ? `近${details.trainWindow}期` : "全部历史" },
      { label: "学习特征", value: details.featureLabel || "全部标准特征" },
      { label: "模型复杂度", value: details.presetLabel || "标准" },
      { label: "生成目标", value: details.objectiveLabel || "综合均衡" },
    ];
    if (details.candidates) rows.push({ label: "候选规模", value: `${details.candidates}组` });
    if (details.structureWeight !== null && details.structureWeight !== undefined) {
      rows.push({ label: "结构约束", value: `${Math.round(details.structureWeight * 100)}%` });
    }
    if (details.temperature !== null && details.temperature !== undefined) {
      rows.push({ label: "探索范围", value: `${Math.round(details.temperature * 100)}%` });
    }
    if (details.diversityWeight !== null && details.diversityWeight !== undefined) {
      rows.push({ label: "多注差异", value: `${Math.round(details.diversityWeight * 100)}%` });
    }
    return rows;
  },

  refreshVisible() {
    this.setData({ visibleFavorites: this.data.favorites.slice(0, this.data.displayLimit) });
  },

  toggleFavorite(event) {
    const id = event.currentTarget.dataset.id;
    const favorites = this.data.favorites.map((item) => ({
      ...item,
      expanded: item.id === id ? !item.expanded : false,
      detailsOpen: item.id === id ? item.detailsOpen : false,
    }));
    this.setData({ favorites }, () => this.refreshVisible());
  },

  toggleDetails(event) {
    const id = event.currentTarget.dataset.id;
    const favorites = this.data.favorites.map((item) => (
      item.id === id ? { ...item, detailsOpen: !item.detailsOpen } : item
    ));
    this.setData({ favorites }, () => this.refreshVisible());
  },

  showMore() {
    this.setData({ displayLimit: this.data.displayLimit + 8 }, () => this.refreshVisible());
  },

  async loadAbout() {
    try {
      this.setData({ about: await api.get("/api/v1/about") });
    } catch (error) {
      // Static service-boundary copy remains visible.
    }
  },

  removeFavorite(event) {
    const id = event.currentTarget.dataset.id;
    wx.showModal({
      title: "删除收藏",
      content: "确认删除这组号码及其开奖对比吗？",
      success: (result) => {
        if (result.confirm) this.loadStoredAfterRemove(id);
      },
    });
  },

  loadStoredAfterRemove(id) {
    storage.removeFavorite(id);
    this.loadFavorites();
  },

  padNumber(value) {
    return String(value).padStart(2, "0");
  },
});
