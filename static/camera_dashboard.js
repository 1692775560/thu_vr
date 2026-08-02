const cameraKeys = ["head", "base"];

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

function renderCodes(key, labels) {
  const strip = document.getElementById(`${key}-codes`);
  if (!strip) return;
  strip.replaceChildren();
  if (!labels || labels.length === 0) {
    const empty = document.createElement("span");
    empty.className = "empty-code";
    empty.textContent = "暂未定位到标签";
    strip.appendChild(empty);
    return;
  }
  for (const item of labels) {
    const badge = document.createElement("span");
    const state = !item.resolved ? "unresolved" : (item.inferred ? "inferred" : "read");
    badge.className = `code-badge ${state}`;
    const suffix = state === "inferred" ? " · 推断" : (state === "unresolved" ? " · 未读全" : ` · ${Number(item.score || 0).toFixed(2)}`);
    badge.textContent = `${item.label}${suffix}`;
    const raw = (item.raw_texts || []).map((value) => value.text).join(" / ");
    badge.title = raw ? `OCR 原文：${raw}` : "OCR 未读出文字";
    strip.appendChild(badge);
  }
}

function updateCamera(key, data) {
  const online = document.getElementById(`${key}-online`);
  online.textContent = data.online ? "在线" : "离线";
  online.className = `state-pill ${data.online ? "online" : "offline"}`;
  setText(`${key}-fps`, `${Number(data.fps || 0).toFixed(1)} FPS`);
  const resolution = data.resolution && data.resolution[0]
    ? `${data.resolution[0]}×${data.resolution[1]}` : "—";
  setText(`${key}-resolution`, resolution);
  setText(`${key}-age`, data.frame_age_seconds == null ? "等待首帧" : `${data.frame_age_seconds.toFixed(1)}s 前`);

  const annotation = data.annotation || {};
  if (annotation.error) {
    setText(`${key}-ocr-count`, "识别失败");
    setText(`${key}-ocr-time`, annotation.error);
  } else if (annotation.updated_at) {
    const candidateCount = Number(annotation.candidate_count || 0);
    const resolvedCount = Number(annotation.resolved_count || 0);
    const inferredCount = Number(annotation.inferred_count || 0);
    const unresolvedCount = Number(annotation.unresolved_count || 0);
    setText(`${key}-ocr-count`, candidateCount
      ? `${candidateCount} 张标签 · ${resolvedCount} 完整（${inferredCount} 推断 / ${unresolvedCount} 未读全）`
      : `${annotation.detection_count || 0} 个文字区域`);
    const version = annotation.algorithm_version ? ` · ${annotation.algorithm_version}` : "";
    setText(`${key}-ocr-time`, `${Number(annotation.processing_seconds || 0).toFixed(1)}s / 帧${version}`);
  } else {
    setText(`${key}-ocr-count`, "等待识别");
    setText(`${key}-ocr-time`, "—");
  }
  renderCodes(key, annotation.labels || annotation.codes || []);
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const status = await response.json();
    let onlineCount = 0;
    for (const key of cameraKeys) {
      updateCamera(key, status[key]);
      if (status[key].online) onlineCount += 1;
    }
    const system = document.querySelector(".system-status");
    system.classList.toggle("ready", onlineCount === cameraKeys.length);
    setText("system-copy", onlineCount === 2 ? "双相机在线" : `${onlineCount}/2 相机在线`);
  } catch (error) {
    document.querySelector(".system-status")?.classList.remove("ready");
    setText("system-copy", "网页服务连接中断");
  }
}

refreshStatus();
setInterval(refreshStatus, 1000);

window.addEventListener("load", () => {
  // Four never-ending MJPEG responses can exhaust a browser's per-host
  // connection pool, leaving the later (base-camera) images unrequested.
  // Poll finite in-memory snapshots instead so all four views stay visible.
  for (const image of document.querySelectorAll("img[data-stream]")) {
    const snapshotUrl = image.dataset.stream.replace("/stream/", "/snapshot/") + ".jpg";
    const interval = image.dataset.stream.endsWith("/raw") ? 500 : 1500;
    const refresh = () => {
      image.onload = () => window.setTimeout(refresh, interval);
      image.onerror = () => window.setTimeout(refresh, 1200);
      image.src = `${snapshotUrl}?t=${Date.now()}`;
    };
    refresh();
  }
  loadExperimentConfig();
});

const variantNames = {
  full_frame_raw: "可部署 · 整帧原图基线",
  candidate_color: "可部署 · 候选标签彩色放大",
  candidate_gray: "可部署 · 候选标签灰度",
  candidate_clahe: "可部署 · 候选标签 CLAHE",
  candidate_clahe_unsharp: "可部署 · CLAHE + 锐化",
  candidate_adaptive: "可部署 · 自适应二值化",
  candidate_color_dynamic_row_consensus: "可部署 · 动态行前缀共识",
  candidate_color_row_visual_stack: "可部署 · 行前缀像素叠加",
  candidate_color_temporal_vote: "可部署 · 纯 OCR 跨帧投票",
  candidate_color_dynamic_row_temporal_vote: "可部署 · 动态前缀 + 跨帧投票",
  candidate_color_spatial_temporal_fusion: "可部署 · 空间跟踪 + 字符级时序融合",
  candidate_color_known_row: "实验上限 · 人工已知行前缀（不可部署）",
  candidate_color_sequence_fusion: "实验上限 · 人工行/固定编号（旧，不可部署）",
  candidate_color_spatial_sequence_fusion: "实验上限 · 人工行/固定编号（不可部署）",
};

const variantDescriptions = {
  full_frame_raw: "直接对整张原图运行 OCR，不做标签候选裁剪。",
  candidate_color: "先定位白色标签，放大彩色裁剪后 OCR；前缀和编号都必须从像素读出。",
  candidate_gray: "标签裁剪转灰度后 OCR。",
  candidate_clahe: "对标签裁剪做局部对比度增强后 OCR。",
  candidate_clahe_unsharp: "CLAHE 后再锐化，增强小字边缘。",
  candidate_adaptive: "标签裁剪做自适应二值化后 OCR。",
  candidate_color_dynamic_row_consensus: "行前缀在当前图像中至少重复读对两次后，才用于同一行；不依赖预设 A01/A05。",
  candidate_color_row_visual_stack: "将同一行多个标签的上半部对齐叠加，动态读出重复前缀；不使用已知前缀或编号范围。",
  candidate_color_temporal_vote: "同一标签跨多帧多数投票，不使用人工行前缀。",
  candidate_color_dynamic_row_temporal_vote: "动态行前缀共识后再做跨帧多数投票，是面向现场的稳定方案。",
  candidate_color_spatial_temporal_fusion: "按标签位置建立轨迹，分别融合行前缀和四位编号；可合并不同帧中的局部读数。",
  candidate_color_known_row: "使用采样时人工勾选的行前缀，只用于测量理论上限。",
  candidate_color_sequence_fusion: "使用人工行前缀与固定编号顺序，只用于旧实验对照。",
  candidate_color_spatial_sequence_fusion: "使用人工行前缀与固定编号顺序，只用于实验上限对照。",
};

const analysisStateNames = {
  not_started: "等待分析",
  queued: "排队中",
  running: "分析中",
  complete: "已完成",
  error: "分析失败",
};

const auditSelections = {};

function percent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function element(tag, className, textContent) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (textContent != null) node.textContent = textContent;
  return node;
}

function metricCell(className, metrics) {
  const cell = element("span", className, percent(metrics?.success_rate));
  if (metrics) cell.title = `命中 ${metrics.true_positive_total}/${metrics.expected_total}`;
  return cell;
}

function renderVariantResults(container, summary, targetRows) {
  if (summary.length === 0) {
    container.appendChild(element("div", "variant-empty", "首帧完成后将在这里逐项显示各方案结果。"));
    return;
  }
  const columnCount = targetRows.length + 5;
  const configureColumns = (row) => {
    row.style.gridTemplateColumns = `minmax(190px, 1.7fr) repeat(${columnCount}, minmax(76px, .62fr))`;
    row.style.minWidth = `${300 + columnCount * 82}px`;
  };
  const header = element("div", "variant-row variant-header");
  configureColumns(header);
  for (const label of ["方案", "总体", ...targetRows, "整帧全对", "OCR 置信度", "精确率", "耗时/帧"]) {
    header.appendChild(element("span", "", label));
  }
  container.appendChild(header);
  for (const item of summary) {
    const usesPrior = item.variant.includes("prior") || item.variant.includes("fusion") || item.variant.includes("known_row");
    const row = element("div", `variant-row${usesPrior ? " prior" : ""}`);
    configureColumns(row);
    const name = element("strong", "variant-name", variantNames[item.variant] || item.variant);
    name.title = [variantDescriptions[item.variant], item.evaluation_basis].filter(Boolean).join("；");
    row.appendChild(name);
    row.appendChild(metricCell("metric success", item));
    for (const labelRow of targetRows) {
      row.appendChild(metricCell("metric row-rate", item.rows?.[labelRow]));
    }
    row.appendChild(element("span", "metric", percent(item.exact_frame_rate)));
    row.appendChild(element("span", "metric confidence", percent(item.mean_ocr_confidence)));
    row.appendChild(element("span", "metric", percent(item.precision)));
    row.appendChild(element("span", "metric", item.variant.includes("temporal_vote") ? "后处理" : `${Number(item.mean_runtime_seconds || 0).toFixed(2)}s`));
    container.appendChild(row);
  }
}

function renderAuditMatrix(container, variant, targetRows) {
  container.replaceChildren();
  const audit = variant?.label_audit;
  if (!audit) {
    container.appendChild(element("div", "variant-empty", "该方案尚无逐标签审计数据。"));
    return;
  }
  const byLabel = new Map((audit.labels || []).map((item) => [item.label, item]));
  const grid = element("div", "label-audit-grid");
  for (const rowName of targetRows) {
    const row = element("div", "label-audit-row");
    row.appendChild(element("strong", "label-audit-row-name", rowName));
    for (let number = 10; number <= 20; number += 1) {
      const label = `${rowName}-${String(number).padStart(4, "0")}`;
      const item = byLabel.get(label) || { total_frames: 0, correct_frames: 0, wrong_frames: 0, missed_frames: 0, wrong_predictions: [] };
      let state = "missed";
      if (item.total_frames > 0 && item.correct_frames === item.total_frames) state = "correct";
      else if (item.correct_frames > 0) state = "mixed";
      else if (item.wrong_frames > 0) state = "wrong";
      const cell = element("div", `label-audit-cell ${state}`);
      cell.appendChild(element("strong", "", String(number).padStart(4, "0")));
      cell.appendChild(element("span", "", `${item.correct_frames}/${item.total_frames}`));
      const wrong = (item.wrong_predictions || []).map((value) => `${value.label}×${value.frames}`).join("、");
      cell.title = `${label}\n正确 ${item.correct_frames} 帧；误读 ${item.wrong_frames} 帧；漏检 ${item.missed_frames} 帧${wrong ? `\n误读为：${wrong}` : ""}`;
      row.appendChild(cell);
    }
    grid.appendChild(row);
  }
  container.appendChild(grid);

  const details = element("div", "label-audit-details");
  const wrongRegions = (audit.wrong_regions || []).slice(0, 12).map((region) => {
    const center = region.center ? ` @(${Math.round(region.center[0])},${Math.round(region.center[1])})` : "";
    return `${region.ground_truth_label} → ${region.predicted_label}${center}`;
  });
  const falsePredictions = (audit.false_predictions || []).slice(0, 12).map((item) => `${item.label}×${item.frames}`);
  if (wrongRegions.length) details.appendChild(element("span", "wrong", `误读区域：${wrongRegions.join("；")}`));
  if (falsePredictions.length) details.appendChild(element("span", "false", `多余/错误预测：${falsePredictions.join("；")}`));
  if (!wrongRegions.length && !falsePredictions.length) details.appendChild(element("span", "clean", "没有定位到误读区域或多余预测。"));
  container.appendChild(details);
}

function renderLabelAudit(section, trial, camera, summary, targetRows) {
  const available = summary.filter((item) => item.label_audit);
  if (!available.length) return;
  const key = `${trial.trial_id}:${camera}`;
  const preferred = available.find((item) => item.variant === "candidate_color_spatial_temporal_fusion")
    || available.find((item) => item.variant === "candidate_color_dynamic_row_temporal_vote")
    || available.find((item) => item.variant === "candidate_color")
    || available[0];
  if (!available.some((item) => item.variant === auditSelections[key])) auditSelections[key] = preferred.variant;

  const panel = element("div", "label-audit-panel");
  const heading = element("div", "label-audit-heading");
  heading.appendChild(element("strong", "", "逐标签审计"));
  const select = document.createElement("select");
  select.className = "label-audit-select";
  for (const item of available) {
    const option = document.createElement("option");
    option.value = item.variant;
    option.textContent = variantNames[item.variant] || item.variant;
    option.selected = item.variant === auditSelections[key];
    select.appendChild(option);
  }
  heading.appendChild(select);
  panel.appendChild(heading);
  const legend = element("div", "label-audit-legend");
  panel.appendChild(legend);
  const matrix = element("div", "label-audit-matrix");
  const renderSelected = () => {
    const selected = available.find((item) => item.variant === auditSelections[key]) || preferred;
    legend.textContent = `真值来源：${selected.label_audit?.truth_source || "实验记录"}。绿色：每帧正确 · 橙色：部分帧正确 · 红色：定位到但误读 · 灰色：漏检；若采样时勾选行错误，审计也会随之错误。`;
    renderAuditMatrix(matrix, selected, targetRows);
  };
  select.addEventListener("change", () => {
    auditSelections[key] = select.value;
    renderSelected();
  });
  renderSelected();
  panel.appendChild(matrix);
  section.appendChild(panel);
}

function renderCameraAblation(body, trial, camera) {
  const cameraResult = trial.camera_results?.[camera];
  const cameraLabel = camera === "head" ? "头部相机" : "底部相机";
  if (!cameraResult) {
    if (camera === "base") {
      body.appendChild(element("div", "legacy-camera-note", "该历史记录没有保存底部相机原图，无法补算。"));
    }
    return;
  }
  const section = element("section", "camera-ablation");
  const heading = element("div", "camera-ablation-heading");
  heading.appendChild(element("strong", "", cameraLabel));
  heading.appendChild(element("span", "", `${trial.analyzed_frames_by_camera?.[camera] || cameraResult.frame_count || 0}/${trial.frame_count_by_camera?.[camera] || 0} 帧 · ${(cameraResult.target_rows || []).join("/")}`));
  section.appendChild(heading);
  const check = cameraResult.ground_truth_check;
  if (check?.state === "warning") {
    section.appendChild(element("div", "ground-truth-warning", `真值行可能有误：${check.warning}`));
  }
  const variants = element("div", "variant-results");
  renderVariantResults(variants, cameraResult.summary || [], cameraResult.target_rows || []);
  section.appendChild(variants);
  renderLabelAudit(section, trial, camera, cameraResult.summary || [], cameraResult.target_rows || []);
  body.appendChild(section);
}

function renderRecentTrials(trials) {
  const container = document.getElementById("recent-trials");
  if (!container) return;
  container.replaceChildren();
  if (!trials || trials.length === 0) {
    const empty = document.createElement("span");
    empty.className = "empty-code";
    empty.textContent = "暂无采样";
    container.appendChild(empty);
    return;
  }
  for (const trial of trials) {
    const item = element("article", "trial-item");
    const visual = element("div", "trial-visual");
    const thumbnailUrls = trial.thumbnail_urls || { head: trial.thumbnail_url };
    for (const camera of cameraKeys) {
      if (!thumbnailUrls[camera]) continue;
      const figure = element("figure", "trial-camera-thumb");
      const thumbnail = document.createElement("img");
      thumbnail.src = `${thumbnailUrls[camera]}?v=${encodeURIComponent(trial.trial_id)}`;
      thumbnail.alt = `${trial.trial_id} ${camera === "head" ? "头部" : "底部"}相机原图`;
      thumbnail.loading = "lazy";
      figure.append(thumbnail, element("figcaption", "", camera === "head" ? "头部原图" : "底部原图"));
      visual.appendChild(figure);
    }

    const body = element("div", "trial-body");
    const heading = element("div", "trial-heading");
    const titleBlock = element("div", "trial-title");
    titleBlock.appendChild(element("strong", "", trial.trial_id));
    const capturedAt = trial.captured_at ? new Date(trial.captured_at).toLocaleString("zh-CN", { hour12: false }) : "时间未知";
    titleBlock.appendChild(element("span", "", capturedAt));
    const analysis = trial.analysis || { state: "not_started", progress: 0 };
    const state = element("span", `analysis-pill ${analysis.state || "not_started"}`, analysisStateNames[analysis.state] || analysis.state);
    heading.append(titleBlock, state);

    const detail = element("div", "trial-meta");
    const pitch = trial.head_up_deg == null ? "角度未知" : `抬头 ${Number(trial.head_up_deg).toFixed(2)}°`;
    for (const value of [
      `${Number(trial.distance_m).toFixed(2)} m`,
      pitch,
      `头 ${trial.frame_count_by_camera?.head || 0} 帧 / 底 ${trial.frame_count_by_camera?.base || 0} 帧`,
      `${trial.analyzed_frames || 0}/${trial.frame_count} 帧已汇总`,
    ]) {
      detail.appendChild(element("span", "", value));
    }
    if (trial.notes) detail.appendChild(element("span", "note", trial.notes));

    if (analysis.state === "queued" || analysis.state === "running") {
      const progressWrap = element("div", "analysis-progress");
      const progressBar = element("span", "");
      progressBar.style.width = percent(analysis.progress);
      progressWrap.appendChild(progressBar);
      const progressCopy = element("div", "analysis-progress-copy");
      const variant = analysis.current_variant ? (variantNames[analysis.current_variant] || analysis.current_variant) : "等待计算资源";
      progressCopy.textContent = `${percent(analysis.progress)} · ${variant}`;
      body.append(heading, detail, progressWrap, progressCopy);
    } else {
      body.append(heading, detail);
    }
    if (analysis.state === "error") {
      body.appendChild(element("div", "analysis-error", analysis.error || "分析进程异常退出"));
    }
    renderCameraAblation(body, trial, "head");
    renderCameraAblation(body, trial, "base");
    item.append(visual, body);
    container.appendChild(item);
  }
}

let experimentConfigLoading = false;
async function loadExperimentConfig() {
  if (experimentConfigLoading) return;
  experimentConfigLoading = true;
  try {
    const response = await fetch("/api/experiment/config", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const trials = data.recent_trials || [];
    renderRecentTrials(trials);
    const running = trials.filter((trial) => ["queued", "running"].includes(trial.analysis?.state)).length;
    const state = document.getElementById("experiment-state");
    if (running > 0) {
      state.textContent = `${running} 条分析中`;
      state.className = "state-pill analysis-running";
    } else if (trials.length > 0) {
      state.textContent = `${trials.length} 条记录`;
      state.className = "state-pill online";
    }
  } catch (error) {
    setText("experiment-result", "实验配置读取失败，请刷新页面重试。");
  } finally {
    experimentConfigLoading = false;
  }
}

setInterval(loadExperimentConfig, 2000);

document.getElementById("experiment-form")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = document.getElementById("experiment-capture");
  const state = document.getElementById("experiment-state");
  const result = document.getElementById("experiment-result");
  const parseTargetRows = (camera) => {
    const value = document.getElementById(`${camera}-target-rows`)?.value || "";
    return [...new Set(value.toUpperCase().split(/[,，\s/]+/).filter(Boolean))];
  };
  const targetRowsByCamera = Object.fromEntries(
    cameraKeys.map((camera) => [camera, parseTargetRows(camera)])
  );
  const missingCamera = cameraKeys.find((camera) => targetRowsByCamera[camera].length === 0);
  if (missingCamera) {
    result.className = "experiment-result error";
    result.textContent = `请填写${missingCamera === "head" ? "头部" : "底部"}相机画面实际可见的标签行。`;
    return;
  }
  const invalid = cameraKeys.flatMap((camera) =>
    targetRowsByCamera[camera]
      .filter((row) => !/^[A-Z][0-9]{2}$/.test(row))
      .map((row) => `${camera === "head" ? "头部" : "底部"}:${row}`)
  );
  if (invalid.length) {
    result.className = "experiment-result error";
    result.textContent = `行前缀格式错误：${invalid.join("、")}；应为一个字母加两位数字，例如 B04、N03。`;
    return;
  }
  button.disabled = true;
  state.textContent = "采样中";
  result.className = "experiment-result";
  const frameCount = Number(document.getElementById("experiment-frame-count").value);
  result.textContent = `正在为两个相机各保存 ${frameCount} 张独立原始帧并读取机器人状态…`;
  try {
    const response = await fetch("/api/experiment/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        distance_m: Number(document.getElementById("experiment-distance").value),
        target_rows_by_camera: targetRowsByCamera,
        frame_count: frameCount,
        notes: document.getElementById("experiment-notes").value,
      }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || `HTTP ${response.status}`);
    result.className = "experiment-result success";
    result.textContent = `已保存 ${data.trial_id}：头部/底部各 ${data.frame_count_by_camera.head}/${data.frame_count_by_camera.base} 帧，距离 ${data.distance_m.toFixed(2)} m，实际抬头 ${data.head_up_deg == null ? "未知" : Number(data.head_up_deg).toFixed(2) + "°"}；全部方案已进入自动分析队列。`;
    state.textContent = "分析已入队";
    for (const input of document.querySelectorAll('input[name$="_target_rows"]')) input.checked = false;
    await loadExperimentConfig();
  } catch (error) {
    result.className = "experiment-result error";
    result.textContent = `采样失败：${error.message}`;
    state.textContent = "采样失败";
  } finally {
    button.disabled = false;
  }
});
