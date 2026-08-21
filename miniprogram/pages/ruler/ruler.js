const api = require("../../utils/api");

const LINE_COLORS = ["#1769aa", "#c13f36", "#17845f", "#b5650b", "#6d55a3"];

Page({
  data: {
    loading: true,
    period: 10,
    periodOptions: [10, 20, 30, 50],
    rows: [],
    activePointCount: 0,
    lineCount: 0,
    activeColor: LINE_COLORS[0],
  },

  onLoad() {
    this.completedPaths = [];
    this.activePath = [];
    this.loadRows();
  },

  onShow() {
    if (!this.hasShown) {
      this.hasShown = true;
      return;
    }
    wx.nextTick(() => this.drawLines());
  },

  onPullDownRefresh() {
    api.clearCache();
    this.loadRows().finally(wx.stopPullDownRefresh);
  },

  onUnload() {
    this.destroyed = true;
  },

  async loadRows() {
    const requestId = (this.requestId || 0) + 1;
    this.requestId = requestId;
    this.setData({ loading: true });
    try {
      const data = await api.get(`/api/v1/draws?limit=${this.data.period}&offset=0&q=`);
      if (this.destroyed || requestId !== this.requestId) return;
      this.baseRows = data.rows.map((row, rowIndex) => ({
        ...row,
        rowIndex,
        shortDate: String(row.date || "").slice(5),
        frontCells: row.front.map((number, index) => ({
          key: `r${rowIndex}f${index}`,
          text: String(number).padStart(2, "0"),
        })),
        backCells: row.back.map((number, index) => ({
          key: `r${rowIndex}b${index}`,
          text: String(number).padStart(2, "0"),
        })),
      }));
      this.completedPaths = [];
      this.activePath = [];
      this.refreshRows();
      this.setData({ loading: false }, () => wx.nextTick(() => this.drawLines()));
    } catch (error) {
      if (requestId !== this.requestId) return;
      this.setData({ loading: false });
      wx.showToast({ title: error.message, icon: "none" });
    }
  },

  selectPeriod(event) {
    const period = Number(event.currentTarget.dataset.period);
    if (period === this.data.period) return;
    this.setData({ period }, () => this.loadRows());
  },

  selectPoint(event) {
    const point = {
      key: event.currentTarget.dataset.key,
      row: Number(event.currentTarget.dataset.row),
      pool: event.currentTarget.dataset.pool,
      index: Number(event.currentTarget.dataset.index),
    };
    const used = this.completedPaths.some((path) => path.points.some((item) => item.key === point.key))
      || this.activePath.some((item) => item.key === point.key);
    if (used) {
      wx.showToast({ title: "这个节点已经在线路中", icon: "none" });
      return;
    }
    this.activePath.push(point);
    this.refreshRows();
  },

  undoPoint() {
    if (!this.activePath.length) return;
    this.activePath.pop();
    this.refreshRows();
  },

  finishLine() {
    if (this.activePath.length < 2) return;
    this.completedPaths.push({
      color: LINE_COLORS[this.completedPaths.length % LINE_COLORS.length],
      points: this.activePath.slice(),
    });
    this.activePath = [];
    this.refreshRows();
  },

  clearLines() {
    wx.showModal({
      title: "清除全部线路",
      content: "已绘制的连线和节点都会被清除。",
      confirmText: "清除",
      confirmColor: "#c13f36",
      success: (result) => {
        if (!result.confirm) return;
        this.completedPaths = [];
        this.activePath = [];
        this.refreshRows();
      },
    });
  },

  refreshRows() {
    const selected = {};
    this.completedPaths.forEach((path) => {
      path.points.forEach((point) => {
        selected[point.key] = path.color;
      });
    });
    const activeColor = LINE_COLORS[this.completedPaths.length % LINE_COLORS.length];
    this.activePath.forEach((point) => {
      selected[point.key] = activeColor;
    });
    const rows = (this.baseRows || []).map((row) => ({
      ...row,
      frontCells: row.frontCells.map((cell) => ({
        ...cell,
        selected: Boolean(selected[cell.key]),
        color: selected[cell.key] || "",
      })),
      backCells: row.backCells.map((cell) => ({
        ...cell,
        selected: Boolean(selected[cell.key]),
        color: selected[cell.key] || "",
      })),
    }));
    this.setData({
      rows,
      activePointCount: this.activePath.length,
      lineCount: this.completedPaths.length,
      activeColor,
    }, () => wx.nextTick(() => this.drawLines()));
  },

  drawLines() {
    if (!this.data.rows.length) return;
    const paths = this.completedPaths.slice();
    if (this.activePath.length) {
      paths.push({
        color: LINE_COLORS[this.completedPaths.length % LINE_COLORS.length],
        points: this.activePath.slice(),
      });
    }
    const points = paths.flatMap((path) => path.points);
    const query = wx.createSelectorQuery().in(this);
    query.select("#ruleCanvas").fields({ node: true, size: true, rect: true });
    points.forEach((point) => query.select(`#${point.key}`).boundingClientRect());
    query.exec((results) => {
      const canvasResult = results[0];
      if (!canvasResult || !canvasResult.node) return;
      const canvas = canvasResult.node;
      const deviceDpr = wx.getWindowInfo ? wx.getWindowInfo().pixelRatio : 2;
      const dpr = Math.min(
        deviceDpr,
        2,
        4096 / Math.max(canvasResult.width, 1),
        4096 / Math.max(canvasResult.height, 1),
      );
      canvas.width = Math.max(1, Math.floor(canvasResult.width * dpr));
      canvas.height = Math.max(1, Math.floor(canvasResult.height * dpr));
      const context = canvas.getContext("2d");
      context.scale(dpr, dpr);
      context.clearRect(0, 0, canvasResult.width, canvasResult.height);
      context.lineCap = "round";
      context.lineJoin = "round";
      context.lineWidth = 2.5;

      let resultIndex = 1;
      paths.forEach((path) => {
        const rects = path.points.map(() => results[resultIndex++]).filter(Boolean);
        if (rects.length < 2) return;
        context.beginPath();
        context.strokeStyle = path.color;
        context.moveTo(
          rects[0].left - canvasResult.left + rects[0].width / 2,
          rects[0].top - canvasResult.top + rects[0].height / 2,
        );
        rects.slice(1).forEach((rect) => {
          context.lineTo(
            rect.left - canvasResult.left + rect.width / 2,
            rect.top - canvasResult.top + rect.height / 2,
          );
        });
        context.stroke();
      });
    });
  },

  showInfo() {
    wx.showModal({
      title: "画规是什么",
      content: "画规把每期开奖号码按位置排成矩阵。依次点击号码可以连接历史节点，用来观察位置变化和形态；线路只是可视化标记，不表示未来号码会沿线路出现。",
      showCancel: false,
      confirmText: "知道了",
    });
  },
});
