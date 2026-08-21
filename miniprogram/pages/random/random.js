const api = require("../../utils/api");
const storage = require("../../utils/storage");

const COUNT_OPTIONS = {
  frontOddCount: ["any", "0", "1", "2", "3", "4", "5"],
  frontBigCount: ["any", "0", "1", "2", "3", "4", "5"],
  backOddCount: ["any", "0", "1", "2"],
};

Page({
  data: {
    count: 5,
    tickets: [],
    statement: "",
    loading: false,
    latestIssue: null,
    favoriteSaved: false,
    mode: "quick",
    selectionAction: "required",
    customSection: "numbers",
    frontRequired: [],
    frontExcluded: [],
    backRequired: [],
    backExcluded: [],
    frontNumbers: [],
    backNumbers: [],
    countOptions: COUNT_OPTIONS,
    constraints: {
      frontOddCount: "any",
      frontBigCount: "any",
      backOddCount: "any",
      frontSumMin: "15",
      frontSumMax: "165",
      consecutive: "any",
      uniqueTails: false,
      zoneMode: "any",
    },
    candidateText: "",
    firstResultReady: false,
    placeholderRows: [1, 2, 3, 4, 5],
  },

  onLoad() {
    this.refreshNumberGrids();
  },

  onReady() {
    wx.pageScrollTo({ scrollTop: 0, duration: 0 });
    wx.nextTick(() => this.generate());
  },

  setMode(event) {
    const mode = event.currentTarget.dataset.mode;
    if (mode === this.data.mode) return;
    const applyMode = () => this.setData({ mode, candidateText: "" });
    if (mode === "quick" && this.data.mode === "custom") {
      wx.pageScrollTo({ scrollTop: 0, duration: 0, complete: applyMode });
      return;
    }
    applyMode();
  },

  decrease() {
    this.setData({ count: Math.max(1, this.data.count - 1) });
  },

  increase() {
    this.setData({ count: Math.min(20, this.data.count + 1) });
  },

  setSelectionAction(event) {
    this.setData({ selectionAction: event.currentTarget.dataset.action });
  },

  setCustomSection(event) {
    this.setData({ customSection: event.currentTarget.dataset.section });
  },

  tapNumber(event) {
    const pool = event.currentTarget.dataset.pool;
    const number = Number(event.currentTarget.dataset.number);
    const requiredKey = `${pool}Required`;
    const excludedKey = `${pool}Excluded`;
    const required = [...this.data[requiredKey]];
    const excluded = [...this.data[excludedKey]];
    const action = this.data.selectionAction;
    const target = action === "required" ? required : excluded;
    const other = action === "required" ? excluded : required;
    const existingIndex = target.indexOf(number);

    if (existingIndex >= 0) {
      target.splice(existingIndex, 1);
    } else {
      const maximum = action === "required" ? (pool === "front" ? 5 : 2) : (pool === "front" ? 30 : 10);
      if (target.length >= maximum) {
        wx.showToast({ title: action === "required" ? "必选号码已达上限" : "排除号码过多", icon: "none" });
        return;
      }
      const otherIndex = other.indexOf(number);
      if (otherIndex >= 0) other.splice(otherIndex, 1);
      target.push(number);
      target.sort((a, b) => a - b);
    }

    this.setData({
      [requiredKey]: required,
      [excludedKey]: excluded,
      candidateText: "",
    }, () => this.refreshNumberGrids());
  },

  refreshNumberGrids() {
    this.setData({
      frontNumbers: this.numberGrid(35, this.data.frontRequired, this.data.frontExcluded),
      backNumbers: this.numberGrid(12, this.data.backRequired, this.data.backExcluded),
    });
  },

  numberGrid(maximum, required, excluded) {
    return Array.from({ length: maximum }, (_, index) => {
      const value = index + 1;
      return {
        value,
        text: String(value).padStart(2, "0"),
        state: required.includes(value) ? "required" : (excluded.includes(value) ? "excluded" : ""),
      };
    });
  },

  setRule(event) {
    const field = event.currentTarget.dataset.field;
    const value = event.currentTarget.dataset.value;
    this.setData({ [`constraints.${field}`]: value, candidateText: "" });
  },

  inputSum(event) {
    const field = event.currentTarget.dataset.field;
    this.setData({ [`constraints.${field}`]: event.detail.value, candidateText: "" });
  },

  toggleUniqueTails(event) {
    this.setData({ "constraints.uniqueTails": event.detail.value, candidateText: "" });
  },

  resetConstraints() {
    this.setData({
      frontRequired: [],
      frontExcluded: [],
      backRequired: [],
      backExcluded: [],
      constraints: {
        frontOddCount: "any",
        frontBigCount: "any",
        backOddCount: "any",
        frontSumMin: "15",
        frontSumMax: "165",
        consecutive: "any",
        uniqueTails: false,
        zoneMode: "any",
      },
      candidateText: "",
    }, () => this.refreshNumberGrids());
  },

  randomConstraints() {
    if (this.data.mode !== "custom") return {};
    return {
      frontRequired: this.data.frontRequired,
      frontExcluded: this.data.frontExcluded,
      backRequired: this.data.backRequired,
      backExcluded: this.data.backExcluded,
      ...this.data.constraints,
    };
  },

  async generate() {
    if (this.data.loading) return;
    this.setData({ loading: true });
    try {
      const data = await api.post("/api/v1/random", {
        count: this.data.count,
        constraints: this.randomConstraints(),
      });
      const cachedHome = getApp().globalData.home;
      const counts = data.candidate_counts || {};
      this.setData({
        tickets: data.tickets,
        statement: data.statement,
        latestIssue: data.latest_issue || (cachedHome && cachedHome.latest ? cachedHome.latest.issue : null),
        favoriteSaved: false,
        firstResultReady: true,
        candidateText: this.data.mode === "custom"
          ? `当前条件有 ${counts.front || 0} 种前区、${counts.back || 0} 种后区组合`
          : "",
      });
    } catch (error) {
      wx.showToast({ title: error.message, icon: "none", duration: 2800 });
    } finally {
      this.setData({ loading: false });
    }
  },

  save() {
    if (!this.data.tickets.length) return;
    const customized = this.data.mode === "custom";
    storage.saveFavorite({
      source: customized ? "custom_random" : "random",
      sourceLabel: customized ? "自定义随机" : "随机生成",
      generatedAfterIssue: this.data.latestIssue,
      tickets: this.data.tickets,
      details: customized ? { type: "custom_random", constraints: this.randomConstraints() } : null,
    });
    this.setData({ favoriteSaved: true });
    wx.showToast({ title: "已收藏", icon: "success" });
  },
});
