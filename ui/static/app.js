(function () {
  const indicatorRows = document.getElementById("indicator-rows");
  const addIndicatorBtn = document.getElementById("add-indicator");
  const strategyToggle = document.getElementById("use-strategy");
  const strategyFields = document.getElementById("strategy-fields");
  const strategyForm = document.getElementById("strategy-form");
  const ruleModeToggle = document.getElementById("rule-mode-toggle");
  const ruleModeValue = document.getElementById("rule-mode-value");
  const ruleSimplePanel = document.getElementById("rule-simple-panel");
  const ruleAdvancedPanel = document.getElementById("rule-advanced-panel");
  const entryRuleSimple = document.querySelector('input[name="entry_rule_simple"]');
  const exitRuleSimple = document.querySelector('input[name="exit_rule_simple"]');
  const entryRuleAdvanced = document.querySelector('textarea[name="entry_rule_advanced"]');
  const exitRuleAdvanced = document.querySelector('textarea[name="exit_rule_advanced"]');
  const entryRuleCanonical = document.getElementById("entry-rule-canonical");
  const exitRuleCanonical = document.getElementById("exit-rule-canonical");

  function indicatorRowTemplate(values) {
    const alias = values?.alias || "";
    const kind = values?.kind || "";
    const length = values?.length || "";
    const source = values?.source || "close";
    const multiplier = values?.multiplier || "";

    return `
      <td><input type="text" name="indicator_alias" value="${alias}" placeholder="ema21" /></td>
      <td>
        <select name="indicator_kind">
          <option value="" ${kind === "" ? "selected" : ""}>Select</option>
          <option value="ema" ${kind === "ema" ? "selected" : ""}>ema</option>
          <option value="rsi" ${kind === "rsi" ? "selected" : ""}>rsi</option>
          <option value="macd" ${kind === "macd" ? "selected" : ""}>macd</option>
          <option value="adx" ${kind === "adx" ? "selected" : ""}>adx</option>
          <option value="supertrend" ${kind === "supertrend" ? "selected" : ""}>supertrend</option>
        </select>
      </td>
      <td><input type="number" name="indicator_length" min="1" value="${length}" /></td>
      <td><input type="text" name="indicator_source" value="${source}" placeholder="close" /></td>
      <td><input type="number" name="indicator_multiplier" step="0.1" value="${multiplier}" /></td>
      <td><button type="button" class="btn-danger remove-indicator">Remove</button></td>
    `;
  }

  if (addIndicatorBtn && indicatorRows) {
    addIndicatorBtn.addEventListener("click", function () {
      const row = document.createElement("tr");
      row.innerHTML = indicatorRowTemplate({});
      indicatorRows.appendChild(row);
    });

    indicatorRows.addEventListener("click", function (event) {
      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }
      if (!target.classList.contains("remove-indicator")) {
        return;
      }

      const row = target.closest("tr");
      if (!row) {
        return;
      }

      const allRows = indicatorRows.querySelectorAll("tr");
      if (allRows.length <= 1) {
        row.innerHTML = indicatorRowTemplate({});
        return;
      }

      row.remove();
    });
  }

  function syncStrategyFields() {
    if (!strategyToggle || !strategyFields) {
      return;
    }
    if (strategyToggle.checked) {
      strategyFields.classList.remove("is-hidden");
    } else {
      strategyFields.classList.add("is-hidden");
    }
  }

  if (strategyToggle) {
    strategyToggle.addEventListener("change", syncStrategyFields);
    syncStrategyFields();
  }

  function looksStructuredRuleText(text) {
    const raw = (text || "").trim().toLowerCase();
    if (!raw) {
      return false;
    }
    return raw.includes("\n") || raw.startsWith("all:") || raw.startsWith("any:") || raw.startsWith("entry:");
  }

  function syncRuleModeUI() {
    if (!ruleModeToggle || !ruleModeValue || !ruleSimplePanel || !ruleAdvancedPanel) {
      return;
    }

    const advanced = ruleModeToggle.checked;
    ruleModeValue.value = advanced ? "advanced" : "simple";

    if (advanced) {
      if (entryRuleAdvanced && !entryRuleAdvanced.value.trim() && entryRuleSimple && entryRuleSimple.value.trim()) {
        entryRuleAdvanced.value = entryRuleSimple.value.trim();
      }
      if (exitRuleAdvanced && !exitRuleAdvanced.value.trim() && exitRuleSimple && exitRuleSimple.value.trim()) {
        exitRuleAdvanced.value = exitRuleSimple.value.trim();
      }
      ruleSimplePanel.classList.add("is-hidden");
      ruleAdvancedPanel.classList.remove("is-hidden");
      return;
    }

    if (entryRuleSimple && !entryRuleSimple.value.trim() && entryRuleAdvanced) {
      const candidate = entryRuleAdvanced.value.trim();
      if (candidate && !looksStructuredRuleText(candidate)) {
        entryRuleSimple.value = candidate;
      }
    }
    if (exitRuleSimple && !exitRuleSimple.value.trim() && exitRuleAdvanced) {
      const candidate = exitRuleAdvanced.value.trim();
      if (candidate && !looksStructuredRuleText(candidate)) {
        exitRuleSimple.value = candidate;
      }
    }

    ruleAdvancedPanel.classList.add("is-hidden");
    ruleSimplePanel.classList.remove("is-hidden");
  }

  function syncCanonicalRules() {
    if (!entryRuleCanonical || !exitRuleCanonical || !ruleModeToggle) {
      return;
    }

    if (ruleModeToggle.checked) {
      entryRuleCanonical.value = entryRuleAdvanced ? entryRuleAdvanced.value : "";
      exitRuleCanonical.value = exitRuleAdvanced ? exitRuleAdvanced.value : "";
    } else {
      entryRuleCanonical.value = entryRuleSimple ? entryRuleSimple.value : "";
      exitRuleCanonical.value = exitRuleSimple ? exitRuleSimple.value : "";
    }
  }

  if (ruleModeToggle) {
    ruleModeToggle.addEventListener("change", syncRuleModeUI);
    syncRuleModeUI();
  }

  if (strategyForm) {
    strategyForm.addEventListener("submit", function () {
      syncCanonicalRules();
    });
  }

  function formatFixed(value, digits) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "n/a";
    }
    return Number(value).toFixed(digits);
  }

  function extractNumericValue(raw) {
    if (Array.isArray(raw)) {
      if (raw.length === 0) {
        return null;
      }
      const maybePrice = raw[raw.length - 1];
      const num = Number(maybePrice);
      return Number.isNaN(num) ? null : num;
    }
    const num = Number(raw);
    return Number.isNaN(num) ? null : num;
  }

  function renderPriceChart(payload, container) {
    const hasOsc = Array.isArray(payload.oscillators) && payload.oscillators.length > 0;
    const grid = hasOsc
      ? [
          { left: 56, right: 20, top: 48, height: "58%" },
          { left: 56, right: 20, top: "73%", height: "18%" },
        ]
      : [{ left: 56, right: 20, top: 48, bottom: 44 }];

    const xAxis = hasOsc
      ? [
          {
            type: "category",
            data: payload.dates || [],
            boundaryGap: false,
            axisLine: { lineStyle: { color: "#334155" } },
            axisLabel: { show: false },
          },
          {
            type: "category",
            gridIndex: 1,
            data: payload.dates || [],
            boundaryGap: false,
            axisLine: { lineStyle: { color: "#334155" } },
            axisLabel: { color: "#94a3b8" },
          },
        ]
      : [
          {
            type: "category",
            data: payload.dates || [],
            boundaryGap: false,
            axisLine: { lineStyle: { color: "#334155" } },
            axisLabel: { color: "#94a3b8" },
          },
        ];

    const yAxis = hasOsc
      ? [
          {
            scale: true,
            axisLine: { lineStyle: { color: "#334155" } },
            axisLabel: {
              color: "#94a3b8",
              formatter: function (value) {
                return formatFixed(value, 2);
              },
            },
            splitLine: { lineStyle: { color: "#253248" } },
          },
          {
            scale: true,
            gridIndex: 1,
            axisLine: { lineStyle: { color: "#334155" } },
            axisLabel: {
              color: "#94a3b8",
              formatter: function (value) {
                return formatFixed(value, 2);
              },
            },
            splitLine: { lineStyle: { color: "#253248" } },
          },
        ]
      : [
          {
            scale: true,
            axisLine: { lineStyle: { color: "#334155" } },
            axisLabel: {
              color: "#94a3b8",
              formatter: function (value) {
                return formatFixed(value, 2);
              },
            },
            splitLine: { lineStyle: { color: "#253248" } },
          },
        ];

    const series = [
      {
        name: "OHLC",
        type: "candlestick",
        data: payload.candles || [],
        itemStyle: {
          color: "#22c55e",
          color0: "#ef4444",
          borderColor: "#22c55e",
          borderColor0: "#ef4444",
        },
      },
    ];

    (payload.overlays || []).forEach(function (overlay) {
      series.push({
        name: overlay.name,
        type: "line",
        data: overlay.data || [],
        showSymbol: false,
        smooth: false,
        lineStyle: { width: 1.5 },
      });
    });

    if (Array.isArray(payload.entries) && payload.entries.length > 0) {
      series.push({
        name: "Entry",
        type: "scatter",
        symbol: "triangle",
        symbolSize: 12,
        itemStyle: { color: "#22c55e" },
        data: payload.entries.map(function (item) {
          return [item.date, item.price];
        }),
      });
    }

    if (Array.isArray(payload.exits) && payload.exits.length > 0) {
      series.push({
        name: "Exit",
        type: "scatter",
        symbol: "triangle",
        symbolRotate: 180,
        symbolSize: 12,
        itemStyle: { color: "#f97316" },
        data: payload.exits.map(function (item) {
          return [item.date, item.price];
        }),
      });
    }

    (payload.oscillators || []).forEach(function (osc) {
      series.push({
        name: osc.name,
        type: "line",
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: osc.data || [],
        showSymbol: false,
        smooth: false,
        lineStyle: { width: 1.3 },
      });
    });

    const option = {
      animation: false,
      backgroundColor: "transparent",
      title: {
        text: payload.title || "Price",
        left: 10,
        top: 6,
        textStyle: { color: "#e2e8f0", fontSize: 14, fontWeight: 600 },
      },
      legend: {
        top: 28,
        left: 10,
        textStyle: { color: "#cbd5e1" },
        itemWidth: 14,
        itemHeight: 10,
      },
      grid: grid,
      xAxis: xAxis,
      yAxis: yAxis,
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross" },
        backgroundColor: "rgba(2, 6, 23, 0.94)",
        borderColor: "#334155",
        textStyle: { color: "#e2e8f0" },
        formatter: function (params) {
          const rows = [];
          const items = Array.isArray(params) ? params : [params];
          if (items.length > 0) {
            rows.push(items[0].axisValueLabel || "");
          }

          items.forEach(function (item) {
            if (!item || !item.seriesName) {
              return;
            }

            if (item.seriesType === "candlestick" && Array.isArray(item.data)) {
              const o = formatFixed(item.data[0], 2);
              const c = formatFixed(item.data[1], 2);
              const l = formatFixed(item.data[2], 2);
              const h = formatFixed(item.data[3], 2);
              rows.push(item.marker + " " + item.seriesName + ": O " + o + "  H " + h + "  L " + l + "  C " + c);
              return;
            }

            const numericValue = extractNumericValue(item.value);
            rows.push(item.marker + " " + item.seriesName + ": " + formatFixed(numericValue, 2));
          });

          return rows.join("<br/>");
        },
      },
      series: series,
    };

    const chart = echarts.init(container, null, { renderer: "canvas" });
    chart.setOption(option);
    window.addEventListener("resize", function () {
      chart.resize();
    });
  }

  function renderEquityChart(payload, container) {
    const series = [
      {
        name: "Strategy NAV",
        type: "line",
        data: payload.values || [],
        showSymbol: false,
        smooth: false,
        lineStyle: { width: 2, color: "#22c55e" },
        areaStyle: { color: "rgba(34, 197, 94, 0.15)" },
      },
    ];

    if (Array.isArray(payload.benchmark_values) && payload.benchmark_values.length > 0) {
      series.push({
        name: payload.benchmark_label || "Buy & Hold",
        type: "line",
        data: payload.benchmark_values,
        showSymbol: false,
        smooth: false,
        lineStyle: { width: 2, color: "#60a5fa", type: "dashed" },
      });
    }

    const option = {
      animation: false,
      backgroundColor: "transparent",
      title: {
        text: payload.title || "Equity Curve",
        left: 10,
        top: 6,
        textStyle: { color: "#e2e8f0", fontSize: 14, fontWeight: 600 },
      },
      legend: {
        top: 28,
        left: 10,
        textStyle: { color: "#cbd5e1" },
        itemWidth: 14,
        itemHeight: 10,
      },
      grid: { left: 56, right: 20, top: 48, bottom: 44 },
      xAxis: {
        type: "category",
        data: payload.dates || [],
        boundaryGap: false,
        axisLine: { lineStyle: { color: "#334155" } },
        axisLabel: { color: "#94a3b8" },
      },
      yAxis: {
        scale: true,
        axisLine: { lineStyle: { color: "#334155" } },
        axisLabel: {
          color: "#94a3b8",
          formatter: function (value) {
            return formatFixed(value, 2);
          },
        },
        splitLine: { lineStyle: { color: "#253248" } },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross" },
        backgroundColor: "rgba(2, 6, 23, 0.94)",
        borderColor: "#334155",
        textStyle: { color: "#e2e8f0" },
        valueFormatter: function (value) {
          return formatFixed(value, 2);
        },
      },
      series: series,
    };

    const chart = echarts.init(container, null, { renderer: "canvas" });
    chart.setOption(option);
    window.addEventListener("resize", function () {
      chart.resize();
    });
  }

  function renderDrawdownChart(payload, container) {
    const option = {
      animation: false,
      backgroundColor: "transparent",
      title: {
        text: payload.title || "Drawdown",
        left: 10,
        top: 6,
        textStyle: { color: "#e2e8f0", fontSize: 14, fontWeight: 600 },
      },
      grid: { left: 56, right: 20, top: 48, bottom: 44 },
      xAxis: {
        type: "category",
        data: payload.dates || [],
        boundaryGap: false,
        axisLine: { lineStyle: { color: "#334155" } },
        axisLabel: { color: "#94a3b8" },
      },
      yAxis: {
        min: function (value) {
          return Math.min(value.min, -0.01);
        },
        max: 0,
        axisLine: { lineStyle: { color: "#334155" } },
        axisLabel: {
          color: "#94a3b8",
          formatter: function (value) {
            return formatFixed(value * 100, 2) + "%";
          },
        },
        splitLine: { lineStyle: { color: "#253248" } },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross" },
        backgroundColor: "rgba(2, 6, 23, 0.94)",
        borderColor: "#334155",
        textStyle: { color: "#e2e8f0" },
        valueFormatter: function (value) {
          if (value === null || value === undefined) {
            return "n/a";
          }
          return (value * 100).toFixed(2) + "%";
        },
      },
      series: [
        {
          name: "Drawdown",
          type: "line",
          data: payload.values || [],
          showSymbol: false,
          smooth: false,
          lineStyle: { width: 2, color: "#f97316" },
          areaStyle: { color: "rgba(249, 115, 22, 0.2)" },
          markLine: {
            symbol: "none",
            lineStyle: { color: "#475569", type: "dashed" },
            data: [{ yAxis: 0 }],
          },
        },
      ],
    };

    const chart = echarts.init(container, null, { renderer: "canvas" });
    chart.setOption(option);
    window.addEventListener("resize", function () {
      chart.resize();
    });
  }

  function renderChartFromScript(scriptId, containerId, renderer) {
    const dataScript = document.getElementById(scriptId);
    const container = document.getElementById(containerId);
    if (!dataScript || !container) {
      return;
    }

    try {
      const payload = JSON.parse(dataScript.textContent || "{}");
      renderer(payload, container);
    } catch (error) {
      console.error("Failed to render chart", error);
    }
  }

  renderChartFromScript("price-chart-json", "price-chart", renderPriceChart);
  renderChartFromScript("equity-chart-json", "equity-chart", renderEquityChart);
  renderChartFromScript("drawdown-chart-json", "drawdown-chart", renderDrawdownChart);
})();
