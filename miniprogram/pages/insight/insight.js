const api = require("../../utils/api");

Page({
  data: {
    window: 100,
    windows: [30, 50, 100, 300],
    viewMode: "heat",
    pool: "front",
    loading: true,
    numbers: [],
    distributions: null,
    sort: "hot",
    trendMetric: "sum",
    trendLabel: "前区和值",
    trendSeries: [],
    selectedTrend: null,
    trendFirstIssue: "",
    trendLastIssue: "",
    histogramPool: "front",
    histogramRows: [],
    selectedHistogram: null,
    selectedNumber: null,
    matrixPool: "front",
    matrixRows: [],
    matrixHeaders: [],
    matrixWidth: 1760,
    ratioMetric: "big_small",
    ratioLabel: "前区大小比",
    ratioRows: [],
    tailPool: "front",
    tailRows: [],
  },

  onLoad() {
    this.loadData();
  },

  onShow() {
    if (!this.hasShown) {
      this.hasShown = true;
      return;
    }
    this.loadData();
  },

  selectWindow(event) {
    const window = Number(event.currentTarget.dataset.window);
    if (window === this.data.window) return;
    this.setData({ window });
    this.loadData();
  },

  selectView(event) {
    this.setData({ viewMode: event.currentTarget.dataset.view });
  },

  selectPool(event) {
    this.setData({ pool: event.currentTarget.dataset.pool });
    this.applySort();
  },

  selectSort(event) {
    this.setData({ sort: event.currentTarget.dataset.sort });
    this.applySort();
  },

  async loadData() {
    const requestId = (this.dataRequestId || 0) + 1;
    this.dataRequestId = requestId;
    this.setData({ loading: true });
    try {
      const [numbers, distributions] = await Promise.all([
        api.get(`/api/v1/statistics/numbers?window=${this.data.window}`),
        api.get(`/api/v1/statistics/distributions?window=${this.data.window}`),
      ]);
      if (requestId !== this.dataRequestId) return;
      this.rawNumbers = numbers;
      const maxOdd = Math.max(...distributions.odd_distribution.map((item) => item.count), 1);
      const maxZone = Math.max(...distributions.zone_distribution.map((item) => item.count), 1);
      distributions.odd_distribution = distributions.odd_distribution.map((item) => ({
        ...item,
        percent: Math.round((item.count / maxOdd) * 100),
      }));
      distributions.zone_distribution = distributions.zone_distribution.map((item) => ({
        ...item,
        percent: Math.round((item.count / maxZone) * 100),
      }));
      distributions.five_zone_distribution = this.normalizeBars(distributions.five_zone_distribution);
      distributions.top_pairs = this.normalizeBars(distributions.top_pairs);
      distributions.position_statistics = distributions.position_statistics.map((item) => ({
        ...item,
        percent: Math.round((item.average / 35) * 100),
      }));
      const maxGap = Math.max(...distributions.gap_statistics.map((item) => item.average), 1);
      distributions.gap_statistics = distributions.gap_statistics.map((item) => ({
        ...item,
        percent: Math.round((item.average / maxGap) * 100),
      }));
      this.distributionsRaw = distributions;
      this.setData({ distributions, windows: numbers.available_windows, loading: false });
      this.applySort();
      this.prepareTrend();
      this.prepareHistogram();
      this.prepareMatrix();
      this.prepareRatios();
      this.prepareTails();
    } catch (error) {
      if (requestId !== this.dataRequestId) return;
      this.setData({ loading: false });
      wx.showToast({ title: error.message, icon: "none" });
    }
  },

  applySort() {
    if (!this.rawNumbers) return;
    const rows = [...this.rawNumbers[this.data.pool]];
    if (this.data.sort === "hot") {
      rows.sort((a, b) => b.heat_score - a.heat_score || a.omission - b.omission || a.number - b.number);
    } else if (this.data.sort === "cold") {
      rows.sort((a, b) => a.heat_score - b.heat_score || b.omission - a.omission || a.number - b.number);
    } else if (this.data.sort === "omission") {
      rows.sort((a, b) => b.omission - a.omission || a.number - b.number);
    } else {
      rows.sort((a, b) => a.number - b.number);
    }
    this.setData({
      numbers: rows.map((item) => ({ ...item, rateText: `${(Number(item.rate) * 100).toFixed(1)}%` })),
      selectedNumber: null,
    });
  },

  toggleNumberDetail(event) {
    const number = Number(event.currentTarget.dataset.number);
    this.setData({ selectedNumber: this.data.selectedNumber === number ? null : number });
  },

  selectTrendMetric(event) {
    this.setData({ trendMetric: event.currentTarget.dataset.metric });
    this.prepareTrend();
  },

  prepareTrend() {
    if (!this.distributionsRaw) return;
    const metric = this.data.trendMetric;
    const labels = { sum: "前区和值", back_sum: "后区和值", span: "前区跨度" };
    const values = this.distributionsRaw.recent_series.map((item) => Number(item[metric]));
    const minimum = Math.min(...values);
    const maximum = Math.max(...values);
    const range = Math.max(maximum - minimum, 1);
    const trendSeries = this.distributionsRaw.recent_series.map((item) => ({
      ...item,
      value: Number(item[metric]),
      height: Math.round(18 + ((Number(item[metric]) - minimum) / range) * 82),
    }));
    this.setData({
      trendLabel: labels[metric],
      trendSeries,
      selectedTrend: trendSeries[trendSeries.length - 1],
      trendFirstIssue: trendSeries[0].issue,
      trendLastIssue: trendSeries[trendSeries.length - 1].issue,
    });
  },

  selectTrendPoint(event) {
    this.setData({ selectedTrend: this.data.trendSeries[Number(event.currentTarget.dataset.index)] });
  },

  selectHistogramPool(event) {
    this.setData({ histogramPool: event.currentTarget.dataset.pool });
    this.prepareHistogram();
  },

  prepareHistogram() {
    if (!this.distributionsRaw) return;
    const source = this.data.histogramPool === "front"
      ? this.distributionsRaw.front_sum_histogram
      : this.distributionsRaw.back_sum_histogram;
    const maximum = Math.max(...source.map((item) => item.count), 1);
    const histogramRows = source.map((item) => ({
      ...item,
      percent: Math.max(4, Math.round((item.count / maximum) * 100)),
      rate_text: `${(Number(item.rate) * 100).toFixed(1)}%`,
    }));
    this.setData({
      histogramRows,
      selectedHistogram: histogramRows.reduce((best, item) => item.count > best.count ? item : best, histogramRows[0]),
    });
  },

  selectHistogram(event) {
    this.setData({ selectedHistogram: this.data.histogramRows[Number(event.currentTarget.dataset.index)] });
  },

  selectMatrixPool(event) {
    this.setData({ matrixPool: event.currentTarget.dataset.pool });
    this.prepareMatrix();
  },

  prepareMatrix() {
    if (!this.distributionsRaw) return;
    const maximum = this.data.matrixPool === "front" ? 35 : 12;
    this.setData({
      matrixRows: this.distributionsRaw.omission_matrix[this.data.matrixPool],
      matrixHeaders: Array.from({ length: maximum }, (_, index) => String(index + 1).padStart(2, "0")),
      matrixWidth: 112 + maximum * 48,
    });
  },

  selectRatioMetric(event) {
    this.setData({ ratioMetric: event.currentTarget.dataset.metric });
    this.prepareRatios();
  },

  prepareRatios() {
    if (!this.distributionsRaw) return;
    const labels = {
      big_small: "前区大小比",
      prime_composite: "前区质合比",
      back_odd_even: "后区奇偶比",
      back_big_small: "后区大小比",
      route_012: "前区012路",
    };
    this.setData({
      ratioLabel: labels[this.data.ratioMetric],
      ratioRows: this.normalizeBars(this.distributionsRaw.ratio_distributions[this.data.ratioMetric]),
    });
  },

  selectTailPool(event) {
    this.setData({ tailPool: event.currentTarget.dataset.pool });
    this.prepareTails();
  },

  prepareTails() {
    if (!this.distributionsRaw) return;
    this.setData({ tailRows: this.distributionsRaw.tail_frequency[this.data.tailPool] });
  },

  normalizeBars(rows) {
    const maximum = Math.max(...rows.map((item) => item.count), 1);
    return rows.map((item) => ({
      ...item,
      percent: Math.max(4, Math.round((item.count / maximum) * 100)),
    }));
  },

  showInfo(event) {
    const window = this.data.window;
    const explanations = {
      frequency: {
        title: "频次、遗漏与冷热",
        content: `“出现”是号码在最近 ${window} 期内出现的次数；“遗漏”是距离最近一次出现已经过去的期数；冷热根据同一区域号码在该窗口内的相对频次划分，只描述历史。`,
      },
      sum: {
        title: "前区平均和值",
        content: `先把每期前区的5个号码相加。例如03、04、14、28、31的和值是3+4+14+28+31=80。这里的大数字是最近 ${window} 期平均值，并与全部历史期数的长期均值比较。蓝色表示偏低、绿色表示接近、红色表示偏高。下方同时显示历史基准和当前窗口范围。它用于观察整体大小，不预测下一期。`,
      },
      backSum: {
        title: "后区平均和值",
        content: `先把每期后区的2个号码相加。例如05、07的和值是5+7=12。这里的大数字是最近 ${window} 期平均值，并与全部历史期数的长期均值比较。蓝色表示偏低、绿色表示接近、红色表示偏高。下方同时显示历史基准和当前窗口范围。它用于观察整体大小，不预测下一期。`,
      },
      span: {
        title: "平均跨度",
        content: `每期前区最大号码减去最小号码得到跨度。这里显示最近 ${window} 期跨度的平均数，下方显示中位数。`,
      },
      odd: {
        title: "奇偶分布",
        content: `统计最近 ${window} 期前区5个号码中，0奇5偶、1奇4偶等六种结构分别出现了多少次。`,
      },
      zone: {
        title: "三区结构",
        content: `前区分为一区01–12、二区13–24、三区25–35。“2:2:1”表示当期三个区域分别开出2、2、1个号码。图中展示最近 ${window} 期最常见的8种结构。`,
      },
      trend: {
        title: "近20期走势",
        content: "展示最近20期前区和值、后区和值或前区跨度的相对高低。点击任意柱子可查看对应期号和准确数值。走势只用于回顾波动，不代表变化会延续。",
      },
      histogram: {
        title: "和值区间分布",
        content: `把最近 ${window} 期的和值按区间分组，柱子越长表示落入该区间的期数越多。点击任一区间可查看期数和占比。`,
      },
      patterns: {
        title: "结构特征",
        content: `基于最近 ${window} 期前区号码计算：连号指同一期出现相邻号码；重号指与上一期有重复号码；大号指18–35；尾数种类按号码个位数去重。它们用于描述历史结构，不提高预测能力。`,
      },
      matrix: {
        title: "逐期号码与遗漏",
        content: "横向是号码，纵向是最近15期开奖。彩色圆点表示当期出现；未出现位置的数字表示截至该期已经连续遗漏多少期。可切换前区和后区。",
      },
      ratios: {
        title: "比例结构",
        content: `统计最近 ${window} 期大小、质合、奇偶和012路结构出现的次数。012路是号码除以3后的余数分类，只用于整理历史结构。`,
      },
      tails: {
        title: "尾数分布",
        content: `统计最近 ${window} 期各个位尾数出现的总次数。例如03、13、23都属于3尾。`,
      },
      positions: {
        title: "位置与间距",
        content: "前区号码从小到大排列后，第1至第5位分别计算平均值和范围；相邻两位相减得到4个间距，用于观察号码在数轴上的疏密。",
      },
      pairs: {
        title: "高频号码对",
        content: `统计最近 ${window} 期中同一期共同出现次数最多的前区两数组合。共同出现较多只描述历史，不代表未来会继续同时出现。`,
      },
    };
    const info = explanations[event.currentTarget.dataset.info];
    if (info) wx.showModal({ ...info, showCancel: false, confirmText: "知道了" });
  },
});
