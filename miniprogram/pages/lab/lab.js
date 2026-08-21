const api = require("../../utils/api");
const storage = require("../../utils/storage");

Page({
  data: {
    loading: true,
    labView: "setup",
    overview: null,
    availableModels: [],
    beginnerModels: [],
    modelIndex: 0,
    customWindow: 300,
    useAllHistory: false,
    objectiveId: "balanced",
    selectedFeatures: ["trend"],
    ticketCount: 5,
    candidateOptions: [1000, 3000, 10000, 30000],
    candidateIndex: 1,
    presetIndex: 1,
    structureWeight: 18,
    exploration: 100,
    diversityWeight: 15,
    testDrawOptions: [10, 20, 40, 60],
    testDrawIndex: 1,
    advancedOpen: false,
    configSummary: "",
    scoreZone: "front",
    generateJob: null,
    generateResult: null,
    favoriteSaved: false,
    backtestJob: null,
    backtestResult: null,
    generationHistory: [],
    visibleGenerationHistory: [],
    historyLimit: 10,
  },

  onLoad() {
    this.destroyed = false;
    this.hidden = false;
    this.pollTimers = {};
    this.loadOverview(true);
  },

  onShow() {
    this.hidden = false;
    if (!this.hasShown) {
      this.hasShown = true;
      return;
    }
    for (const type of ["generate", "backtest"]) {
      const job = this.data[`${type}Job`];
      if (job && ["queued", "running"].includes(job.status)) this.pollJob(job.id, type);
    }
  },

  onHide() {
    this.hidden = true;
    this.clearPollTimers();
  },

  onPullDownRefresh() {
    this.loadOverview(false).finally(wx.stopPullDownRefresh);
  },

  onUnload() {
    this.destroyed = true;
    this.clearPollTimers();
  },

  clearPollTimers() {
    Object.values(this.pollTimers || {}).forEach((timer) => clearTimeout(timer));
    this.pollTimers = {};
  },

  async loadOverview(initialize) {
    try {
      const overview = await api.get("/api/v1/lab/overview");
      const availableModels = overview.models.filter((item) => item.available);
      const generationHistory = this.prepareGenerationHistory(overview.generation_history || [], overview);
      const changes = { overview, availableModels, generationHistory, loading: false };
      if (initialize) {
        const recommendedModel = overview.recommendation.model;
        changes.modelIndex = Math.max(0, availableModels.findIndex((item) => item.id === recommendedModel));
        const targetWindow = overview.recommendation.train_window || 0;
        changes.customWindow = targetWindow || Math.min(300, overview.total_draws || 5000);
        changes.useAllHistory = !targetWindow;
        const latestGenerate = overview.jobs.find((item) => item.job_type === "generate");
        const latestBacktest = overview.jobs.find((item) => item.job_type === "backtest");
        if (latestGenerate) {
          changes.generateJob = latestGenerate;
          if (latestGenerate.status === "completed" && latestGenerate.result) {
            changes.generateResult = this.prepareGenerateResult(latestGenerate.result);
          }
        }
        if (latestBacktest) {
          changes.backtestJob = latestBacktest;
          if (latestBacktest.status === "completed" && latestBacktest.result) {
            changes.backtestResult = this.prepareBacktestResult(latestBacktest.result);
          }
        }
      }
      this.setData(changes, () => {
        this.refreshControlViews();
        this.refreshGenerationHistory();
      });
      if (initialize) {
        for (const type of ["generate", "backtest"]) {
          const job = changes[`${type}Job`];
          if (job && ["queued", "running"].includes(job.status)) this.pollJob(job.id, type);
        }
      }
    } catch (error) {
      this.setData({ loading: false });
      wx.showToast({ title: error.message, icon: "none" });
    }
  },

  refreshControlViews() {
    const { overview, availableModels, modelIndex, selectedFeatures, objectiveId } = this.data;
    if (!overview || !availableModels.length) return;
    const activeModel = availableModels[modelIndex] || availableModels[0];
    const preferredIds = [activeModel.id, "logistic", "hist_gradient_boosting", "extra_trees"];
    const beginnerModels = [];
    for (const id of preferredIds) {
      const model = availableModels.find((item) => item.id === id);
      if (model && !beginnerModels.some((item) => item.id === id) && beginnerModels.length < 3) {
        beginnerModels.push({ ...model, selected: id === activeModel.id });
      }
    }
    const featureGroups = overview.feature_groups.map((item) => ({
      ...item,
      selected: selectedFeatures.includes(item.id),
    }));
    const objectives = overview.objectives.map((item) => ({ ...item, selected: item.id === objectiveId }));
    const objective = overview.objectives.find((item) => item.id === objectiveId);
    const windowLabel = this.data.useAllHistory ? "全部历史" : `近${this.data.customWindow}期`;
    const configSummary = `${activeModel.label} · ${windowLabel} · ${selectedFeatures.length}类特征 · ${objective.label}`;
    this.setData({
      beginnerModels,
      "overview.feature_groups": featureGroups,
      "overview.objectives": objectives,
      configSummary,
    });
  },

  toggleAdvanced() {
    this.setData({ advancedOpen: !this.data.advancedOpen });
  },

  selectLabView(event) {
    this.setData({ labView: event.currentTarget.dataset.view });
  },

  selectQuickModel(event) {
    const id = event.currentTarget.dataset.id;
    const modelIndex = this.data.availableModels.findIndex((item) => item.id === id);
    this.setData({ modelIndex }, () => this.refreshControlViews());
  },

  selectModel(event) {
    this.setData({ modelIndex: Number(event.detail.value) }, () => this.refreshControlViews());
  },

  editCustomWindow(event) {
    this.setData({ customWindow: event.detail.value, useAllHistory: false });
  },

  commitCustomWindow() {
    const value = this.clampWindow(this.data.customWindow);
    this.setData({ customWindow: value, useAllHistory: false }, () => this.refreshControlViews());
  },

  decreaseWindow() {
    const value = this.clampWindow(Number(this.data.customWindow || 50) - 50);
    this.setData({ customWindow: value, useAllHistory: false }, () => this.refreshControlViews());
  },

  increaseWindow() {
    const value = this.clampWindow(Number(this.data.customWindow || 50) + 50);
    this.setData({ customWindow: value, useAllHistory: false }, () => this.refreshControlViews());
  },

  toggleAllHistory() {
    this.setData({ useAllHistory: !this.data.useAllHistory }, () => this.refreshControlViews());
  },

  clampWindow(value) {
    const maximum = this.data.overview ? (this.data.overview.total_draws || 5000) : 5000;
    return Math.min(Math.max(Number.parseInt(value, 10) || 50, 50), maximum);
  },

  toggleFeature(event) {
    const id = event.currentTarget.dataset.id;
    const selected = [...this.data.selectedFeatures];
    const index = selected.indexOf(id);
    if (index >= 0) {
      if (selected.length === 1) {
        wx.showToast({ title: "至少保留一类特征", icon: "none" });
        return;
      }
      selected.splice(index, 1);
    } else {
      selected.push(id);
    }
    this.setData({ selectedFeatures: selected }, () => this.refreshControlViews());
  },

  selectObjective(event) {
    this.setData({ objectiveId: event.currentTarget.dataset.id }, () => this.refreshControlViews());
  },

  selectCandidates(event) {
    this.setData({ candidateIndex: Number(event.detail.value) });
  },

  selectPreset(event) {
    this.setData({ presetIndex: Number(event.detail.value) });
  },

  selectTestDraws(event) {
    this.setData({ testDrawIndex: Number(event.detail.value) });
  },

  changeStructureWeight(event) {
    this.setData({ structureWeight: Number(event.detail.value) });
  },

  changeExploration(event) {
    this.setData({ exploration: Number(event.detail.value) });
  },

  changeDiversityWeight(event) {
    this.setData({ diversityWeight: Number(event.detail.value) });
  },

  decreaseTickets() {
    this.setData({ ticketCount: Math.max(1, this.data.ticketCount - 1) });
  },

  increaseTickets() {
    this.setData({ ticketCount: Math.min(10, this.data.ticketCount + 1) });
  },

  selectScoreZone(event) {
    this.setData({ scoreZone: event.currentTarget.dataset.zone });
  },

  experimentPayload(forBacktest = false) {
    const model = this.data.availableModels[this.data.modelIndex];
    return {
      model: model.id,
      sourceMode: "custom_model",
      trainWindow: this.data.useAllHistory ? 0 : this.clampWindow(this.data.customWindow),
      featureGroups: this.data.selectedFeatures,
      objective: this.data.objectiveId,
      modelPreset: this.data.overview.presets[this.data.presetIndex].id,
      structureWeight: this.data.advancedOpen ? this.data.structureWeight / 100 : null,
      temperature: this.data.advancedOpen ? this.data.exploration / 100 : null,
      diversityWeight: this.data.advancedOpen ? this.data.diversityWeight / 100 : null,
      tickets: forBacktest ? 3 : this.data.ticketCount,
      candidates: forBacktest ? Math.min(this.data.candidateOptions[this.data.candidateIndex], 1000) : this.data.candidateOptions[this.data.candidateIndex],
      testDraws: this.data.testDrawOptions[this.data.testDrawIndex],
      save: !forBacktest,
    };
  },

  async startGenerate() {
    if (this.generateSubmitting || (this.data.generateJob && ["queued", "running"].includes(this.data.generateJob.status))) return;
    this.generateSubmitting = true;
    try {
      const response = await api.post("/api/v1/lab/jobs/generate", this.experimentPayload(false));
      this.setData({
        generateResult: null,
        favoriteSaved: false,
        generateJob: { id: response.job_id, status: "queued", progress: 0, message: "任务已提交" },
      });
      this.pollJob(response.job_id, "generate");
    } catch (error) {
      wx.showToast({ title: error.message, icon: "none" });
    } finally {
      this.generateSubmitting = false;
    }
  },

  async startBacktest() {
    if (this.backtestSubmitting || (this.data.backtestJob && ["queued", "running"].includes(this.data.backtestJob.status))) return;
    this.backtestSubmitting = true;
    try {
      const response = await api.post("/api/v1/lab/jobs/backtest", this.experimentPayload(true));
      this.setData({
        backtestResult: null,
        backtestJob: { id: response.job_id, status: "queued", progress: 0, message: "能力评估已提交" },
      });
      this.pollJob(response.job_id, "backtest");
    } catch (error) {
      wx.showToast({ title: error.message, icon: "none" });
    } finally {
      this.backtestSubmitting = false;
    }
  },

  async pollJob(jobId, type) {
    if (this.destroyed || this.hidden) return;
    if (this.pollTimers[type]) clearTimeout(this.pollTimers[type]);
    try {
      const job = await api.get(`/api/v1/lab/jobs/${jobId}`);
      this.pollErrorShown = false;
      const update = {};
      update[`${type}Job`] = job;
      if (job.status === "completed") {
        if (type === "generate") update.generateResult = this.prepareGenerateResult(job.result);
        if (type === "backtest") update.backtestResult = this.prepareBacktestResult(job.result);
        update.labView = "results";
        this.setData(update);
        await this.loadOverview(false);
        return;
      }
      if (job.status === "failed") {
        this.setData(update);
        wx.showToast({ title: job.error || "任务失败", icon: "none" });
        return;
      }
      this.setData(update);
      this.pollTimers[type] = setTimeout(() => this.pollJob(jobId, type), 1200);
    } catch (error) {
      if (!this.pollErrorShown) {
        this.pollErrorShown = true;
        wx.showToast({ title: "网络波动，任务仍在后台运行", icon: "none" });
      }
      this.pollTimers[type] = setTimeout(() => this.pollJob(jobId, type), 2500);
    }
  },

  prepareGenerateResult(result) {
    const diagnostics = result.diagnostics || {};
    const frontMetrics = diagnostics.front_metrics || {};
    const backMetrics = diagnostics.back_metrics || {};
    const generationSettings = result.generation_settings || {};
    return {
      ...result,
      tickets: result.tickets.map((ticket) => ({
        ...ticket,
        front_numbers: ticket.front.split(" "),
        back_numbers: ticket.back.split(" "),
      })),
      front_scores: result.scores.front.slice(0, 10),
      back_scores: result.scores.back.slice(0, 6),
      diagnosticsView: [
        { label: "使用特征", value: diagnostics.feature_count ? `${diagnostics.feature_count}项` : "全部" },
        { label: "计算方式", value: result.training_mode === "cached" ? "读取缓存" : "本次重训" },
        { label: "前区 ROC AUC", value: Number(frontMetrics.roc_auc || 0).toFixed(3) },
        { label: "后区 ROC AUC", value: Number(backMetrics.roc_auc || 0).toFixed(3) },
        { label: "总耗时", value: result.elapsed_seconds !== undefined ? `${Number(result.elapsed_seconds).toFixed(2)}秒` : "-" },
      ],
    };
  },

  prepareGenerationHistory(rows, overview) {
    return rows.map((row) => {
      const preset = overview.presets.find((item) => item.id === row.model_preset);
      const objective = overview.objectives.find((item) => item.id === row.objective);
      return {
        ...row,
        expanded: false,
        createdText: String(row.created_at || "").replace("T", " ").slice(0, 16),
        sourceLabel: row.source_mode === "custom_model" ? "自定义模型" : row.source_mode === "quick_model" ? "快速模型" : "模型生成",
        windowLabel: row.train_window ? `近${row.train_window}期` : "全部历史",
        featureCount: (row.feature_groups || []).length,
        presetLabel: preset ? preset.label : row.model_preset,
        objectiveLabel: objective ? objective.label : row.objective,
        trainingModeLabel: row.training_mode === "cached" ? "读取缓存" : row.training_mode === "trained" ? "本次重训" : "历史记录",
        elapsedText: row.elapsed_seconds !== null && row.elapsed_seconds !== undefined ? `${Number(row.elapsed_seconds).toFixed(2)}秒` : "-",
        tickets: (row.tickets || []).map((ticket) => ({
          ...ticket,
          frontNumbers: String(ticket.front || "").split(" ").filter(Boolean),
          backNumbers: String(ticket.back || "").split(" ").filter(Boolean),
        })),
      };
    });
  },

  refreshGenerationHistory() {
    this.setData({ visibleGenerationHistory: this.data.generationHistory.slice(0, this.data.historyLimit) });
  },

  toggleGenerationHistory(event) {
    const id = event.currentTarget.dataset.id;
    const generationHistory = this.data.generationHistory.map((item) => ({
      ...item,
      expanded: item.id === id ? !item.expanded : false,
    }));
    this.setData({ generationHistory }, () => this.refreshGenerationHistory());
  },

  showMoreHistory() {
    this.setData({ historyLimit: this.data.historyLimit + 10 }, () => this.refreshGenerationHistory());
  },

  deleteGenerationHistory(event) {
    const id = event.currentTarget.dataset.id;
    wx.showModal({
      title: "删除生成历史",
      content: "确认删除这次模型配置和生成结果吗？已收藏的号码不会受影响。",
      success: async (result) => {
        if (!result.confirm) return;
        try {
          await api.post(`/api/v1/lab/history/${id}/delete`, {});
          const generationHistory = this.data.generationHistory.filter((item) => item.id !== id);
          this.setData({ generationHistory }, () => this.refreshGenerationHistory());
          wx.showToast({ title: "已删除", icon: "success" });
        } catch (error) {
          wx.showToast({ title: error.message, icon: "none" });
        }
      },
    });
  },

  saveCustomFavorite() {
    if (!this.data.generateResult || this.data.favoriteSaved) return;
    const result = this.data.generateResult;
    const selectedGroups = result.feature_groups || this.data.overview.feature_groups.map((item) => item.id);
    const featureLabels = this.data.overview.feature_groups
      .filter((item) => selectedGroups.includes(item.id))
      .map((item) => item.label)
      .join("、");
    const preset = this.data.overview.presets.find((item) => item.id === result.model_preset);
    const objective = this.data.overview.objectives.find((item) => item.id === result.objective);
    storage.saveFavorite({
      source: "custom_model",
      sourceLabel: "自定义模型",
      generatedAfterIssue: result.trained_until_issue,
      tickets: result.tickets,
      details: {
        model: result.model,
        modelLabel: result.model_label,
        trainWindow: result.train_window,
        featureLabel: featureLabels || "全部特征",
        presetLabel: preset ? preset.label : result.model_preset,
        objectiveLabel: objective ? objective.label : result.objective,
        candidates: result.generation_settings ? result.generation_settings.candidates : null,
        structureWeight: result.generation_settings ? result.generation_settings.structure_weight : null,
        temperature: result.generation_settings ? result.generation_settings.temperature : null,
        diversityWeight: result.generation_settings ? result.generation_settings.diversity_weight : null,
      },
    });
    this.setData({ favoriteSaved: true });
    wx.showToast({ title: "已收藏", icon: "success" });
  },

  prepareBacktestResult(result) {
    const model = result.summary.find((item) => item.strategy !== "random");
    const random = result.summary.find((item) => item.strategy === "random");
    const frontLift = Number(model.front_lift_vs_random || 0);
    const backLift = Number(model.back_lift_vs_random || 0);
    const prizeLift = Number(model.prize_rate_lift_vs_random || 0);
    return {
      ...result,
      model,
      random,
      frontLift,
      backLift,
      prizeLift,
      modelFrontText: Number(model.avg_front_hits).toFixed(3),
      randomFrontText: Number(random.avg_front_hits).toFixed(3),
      modelBackText: Number(model.avg_back_hits).toFixed(3),
      randomBackText: Number(random.avg_back_hits).toFixed(3),
      prizeRateText: `${(Number(model.any_prize_rate) * 100).toFixed(2)}%`,
      frontLiftText: `${frontLift > 0 ? "+" : ""}${frontLift.toFixed(3)}`,
      backLiftText: `${backLift > 0 ? "+" : ""}${backLift.toFixed(3)}`,
      prizeLiftText: `${prizeLift > 0 ? "+" : ""}${(prizeLift * 100).toFixed(2)}%`,
    };
  },

  showLabInfo() {
    wx.showModal({
      title: "自主模型实验",
      content: "你选择模型、学习范围、特征和组合目标，系统按这些设置重新训练。输出是历史模式下的相对评分，不是客观中奖概率；模型能力评估会用未参与学习的历史开奖与随机结果比较。",
      showCancel: false,
      confirmText: "知道了",
    });
  },
});
