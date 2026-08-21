const api = require("../../utils/api");
const storage = require("../../utils/storage");

Page({
  data: {
    loading: true,
    overview: null,
    models: [],
    modelIndex: 0,
    windows: [],
    windowIndex: 1,
    ticketCount: 5,
    job: null,
    result: null,
    favoriteSaved: false,
  },

  onLoad() {
    this.destroyed = false;
    this.loadOptions();
  },

  onUnload() {
    this.destroyed = true;
    if (this.pollTimer) clearTimeout(this.pollTimer);
  },

  async loadOptions() {
    try {
      const overview = await api.get("/api/v1/lab/overview");
      const models = overview.models.filter((item) => item.available);
      const modelIndex = Math.max(0, models.findIndex((item) => item.id === overview.recommendation.model));
      const targetWindow = overview.recommendation.train_window || 0;
      const windowIndex = Math.max(0, overview.windows.findIndex((item) => item.value === targetWindow));
      const windows = overview.windows.map((item, index) => ({ ...item, selected: index === windowIndex }));
      this.setData({ overview, models, modelIndex, windows, windowIndex, loading: false });
    } catch (error) {
      this.setData({ loading: false });
      wx.showToast({ title: error.message, icon: "none" });
    }
  },

  selectModel(event) {
    this.setData({ modelIndex: Number(event.detail.value) });
  },

  selectWindow(event) {
    const windowIndex = Number(event.currentTarget.dataset.index);
    const windows = this.data.windows.map((item, index) => ({ ...item, selected: index === windowIndex }));
    this.setData({ windowIndex, windows });
  },

  decreaseTickets() {
    this.setData({ ticketCount: Math.max(1, this.data.ticketCount - 1) });
  },

  increaseTickets() {
    this.setData({ ticketCount: Math.min(10, this.data.ticketCount + 1) });
  },

  async generate() {
    if (this.submitting || (this.data.job && ["queued", "running"].includes(this.data.job.status))) return;
    this.submitting = true;
    const model = this.data.models[this.data.modelIndex];
    const window = this.data.windows[this.data.windowIndex];
    try {
      const response = await api.post("/api/v1/lab/jobs/generate", {
        model: model.id,
        sourceMode: "quick_model",
        trainWindow: window.value,
        tickets: this.data.ticketCount,
        candidates: 3000,
        featureGroups: this.data.overview.feature_groups.map((item) => item.id),
        modelPreset: "standard",
        objective: "balanced",
        save: true,
      });
      this.setData({
        result: null,
        favoriteSaved: false,
        job: { id: response.job_id, status: "queued", progress: 0, message: "准备训练" },
      });
      this.poll(response.job_id);
    } catch (error) {
      wx.showToast({ title: error.message, icon: "none" });
    } finally {
      this.submitting = false;
    }
  },

  async poll(jobId) {
    if (this.destroyed) return;
    try {
      const job = await api.get(`/api/v1/lab/jobs/${jobId}`);
      this.pollErrorShown = false;
      if (job.status === "completed") {
        this.setData({ job, result: this.prepareResult(job.result) });
        return;
      }
      if (job.status === "failed") {
        this.setData({ job });
        wx.showToast({ title: job.error || "生成失败", icon: "none" });
        return;
      }
      this.setData({ job });
      this.pollTimer = setTimeout(() => this.poll(jobId), 1000);
    } catch (error) {
      if (!this.pollErrorShown) {
        this.pollErrorShown = true;
        wx.showToast({ title: "网络波动，正在重试", icon: "none" });
      }
      this.pollTimer = setTimeout(() => this.poll(jobId), 2000);
    }
  },

  prepareResult(result) {
    return {
      ...result,
      trainingModeLabel: result.training_mode === "cached" ? "读取已训练模型" : "本次重新训练",
      elapsedText: result.elapsed_seconds !== undefined ? `${Number(result.elapsed_seconds).toFixed(2)}秒` : "",
      tickets: result.tickets.map((ticket) => ({
        ...ticket,
        frontNumbers: ticket.front.split(" "),
        backNumbers: ticket.back.split(" "),
      })),
    };
  },

  saveFavorite() {
    if (!this.data.result || this.data.favoriteSaved) return;
    const result = this.data.result;
    storage.saveFavorite({
      source: "quick_model",
      sourceLabel: "模型生成",
      generatedAfterIssue: result.trained_until_issue,
      tickets: result.tickets,
      details: {
        model: result.model,
        modelLabel: result.model_label,
        trainWindow: result.train_window,
        featureLabel: "全部标准特征",
        presetLabel: "标准",
        objectiveLabel: "综合均衡",
        candidates: 3000,
      },
    });
    this.setData({ favoriteSaved: true });
    wx.showToast({ title: "已收藏", icon: "success" });
  },

  showInfo() {
    wx.showModal({
      title: "快速模型生成",
      content: "该页面固定使用全部标准特征、标准复杂度和综合均衡策略。需要编辑特征、复杂度、候选规模和权重时，请前往“实验”。模型结果是历史模式相对评分，不是客观中奖概率。",
      showCancel: false,
      confirmText: "知道了",
    });
  },
});
