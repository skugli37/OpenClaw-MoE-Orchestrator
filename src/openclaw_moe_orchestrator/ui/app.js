const dashboardUrl = "/api/dashboard";

const ROLE_ORDER = ["reasoning", "coding", "general", "vision", "embedding", "safety"];
const ROLE_LABELS = {
  reasoning: "Reasoning",
  coding: "Coding",
  general: "General",
  vision: "Vision",
  embedding: "Embedding",
  safety: "Safety",
};

const els = {
  heroStatus: document.getElementById("hero-status"),
  metricGrid: document.getElementById("metric-grid"),
  profileCard: document.getElementById("profile-card"),
  jobFeed: document.getElementById("job-feed"),
  runsFeed: document.getElementById("runs-feed"),
  refresh: document.getElementById("refresh-dashboard"),
  buttons: Array.from(document.querySelectorAll("[data-workflow]")),
  resetModels: document.getElementById("reset-models"),
  applyModels: document.getElementById("apply-models"),
  roleGrid: document.getElementById("role-grid"),
  catalogStatus: document.getElementById("catalog-status"),
  liveInventory: document.getElementById("live-inventory"),
};

let pollHandle = null;
let latestSnapshot = null;
let roleSelections = {};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function getRoleDefaults(catalog, role) {
  return [...(catalog?.defaults?.[role] || [])];
}

function getRoleSpecs(catalog, role) {
  return [...(catalog?.roles?.[role] || [])];
}

function getLiveEntries(catalog) {
  return [...(catalog?.live_entries || [])];
}

function initRoleSelections(catalog) {
  ROLE_ORDER.forEach((role) => {
    const existing = roleSelections[role];
    if (existing?.length) {
      return;
    }
    roleSelections[role] = getRoleDefaults(catalog, role);
  });
}

function buildInstallPayload() {
  return {
    workflow: "install-openclaw-cloud",
    reasoning_models: roleSelections.reasoning || [],
    coding_models: roleSelections.coding || [],
    general_models: roleSelections.general || [],
    vision_models: roleSelections.vision || [],
    embedding_models: roleSelections.embedding || [],
    safety_models: roleSelections.safety || [],
  };
}

async function fetchDashboard() {
  const response = await fetch(dashboardUrl);
  if (!response.ok) {
    throw new Error(`Dashboard request failed: ${response.status}`);
  }
  return response.json();
}

async function submitWorkflow(workflow, payload = {}) {
  const response = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ workflow, ...payload }),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "request failed" }));
    throw new Error(error.error || "request failed");
  }
  return response.json();
}

function renderHero(snapshot) {
  const active = snapshot.active_profile || {};
  const catalog = snapshot.model_catalog || {};
  const models = catalog.live_models || [];
  els.heroStatus.innerHTML = `
    <div class="status-pill">${active.primary ? "Live" : "Needs Setup"}</div>
    <p><strong>${models.length}</strong> models visible through Ollama.</p>
    <p class="mono">${escapeHtml(active.primary || "No active primary configured")}</p>
  `;
}

function renderMetrics(snapshot) {
  const environment = snapshot.environment || {};
  const catalog = snapshot.model_catalog || {};
  const metrics = [
    ["Python", environment.python || "n/a"],
    ["GPU", environment.gpu_name || "CPU"],
    ["Ollama Models", String((catalog.live_models || []).length)],
    ["OpenClaw Binary", environment.openclaw_binary ? "Detected" : "Missing"],
  ];
  els.metricGrid.innerHTML = metrics
    .map(
      ([label, value]) => `
        <div class="metric-card">
          <div class="metric-label">${label}</div>
          <div class="metric-value">${escapeHtml(value)}</div>
        </div>
      `,
    )
    .join("");
}

function renderProfile(snapshot) {
  const profile = snapshot.active_profile || {};
  if (!profile.exists) {
    els.profileCard.innerHTML = `<div class="profile-lines"><p>No active OpenClaw config found.</p></div>`;
    return;
  }
  const fallbackText = (profile.fallbacks || []).join(", ") || "None";
  els.profileCard.innerHTML = `
    <div class="profile-lines">
      <div class="pair"><span>Provider</span><strong>${escapeHtml(profile.api || "ollama")}</strong></div>
      <div class="pair"><span>Primary</span><strong class="mono">${escapeHtml(profile.primary)}</strong></div>
      <div class="pair"><span>Fallbacks</span><strong class="mono">${escapeHtml(fallbackText)}</strong></div>
      <div class="pair"><span>Endpoint</span><strong class="mono">${escapeHtml(profile.base_url || "n/a")}</strong></div>
      <div class="pair"><span>Catalog</span><strong>${(profile.provider_models || []).length} models in active config</strong></div>
    </div>
  `;
}

function renderJobs(snapshot) {
  const jobs = snapshot.jobs || [];
  if (!jobs.length) {
    els.jobFeed.innerHTML = `<div class="job-card"><p>No jobs yet.</p></div>`;
    return;
  }
  els.jobFeed.innerHTML = jobs
    .map(
      (job) => `
        <div class="job-card">
          <div class="badge ${job.status}">${escapeHtml(job.status)}</div>
          <p class="job-title">${escapeHtml(job.workflow)}</p>
          <div class="job-meta">
            <div class="pair"><span>Job ID</span><strong class="mono">${escapeHtml(job.job_id)}</strong></div>
            <div class="pair"><span>Created</span><strong>${new Date(job.created_at).toLocaleString()}</strong></div>
            ${
              job.error
                ? `<div class="pair"><span>Error</span><strong class="mono">${escapeHtml(job.error)}</strong></div>`
                : ""
            }
          </div>
        </div>
      `,
    )
    .join("");
}

function renderRuns(snapshot) {
  const runs = snapshot.recent_runs || [];
  if (!runs.length) {
    els.runsFeed.innerHTML = `<div class="run-card"><p>No run bundles found.</p></div>`;
    return;
  }
  els.runsFeed.innerHTML = runs
    .map((run) => {
      const outputs = (run.outputs || [])
        .map(
          (output) =>
            `<a href="/repo/${encodeURI(output.path)}" target="_blank" rel="noreferrer">${escapeHtml(output.name)}</a>`,
        )
        .join("");
      return `
        <div class="run-card">
          <p class="run-title">${escapeHtml(run.workflow || run.run_id)}</p>
          <div class="run-meta">
            <div class="pair"><span>Run ID</span><strong class="mono">${escapeHtml(run.run_id)}</strong></div>
            <div class="pair"><span>Created</span><strong>${new Date(run.created_at).toLocaleString()}</strong></div>
            <div class="pair"><span>Bundle</span><strong class="mono">${escapeHtml(run.bundle_dir)}</strong></div>
          </div>
          <div class="run-outputs">${outputs || "<span class=\"panel-note\">No outputs yet.</span>"}</div>
        </div>
      `;
    })
    .join("");
}

function renderCatalog(snapshot) {
  const catalog = snapshot.model_catalog || {};
  initRoleSelections(catalog);

  const liveCount = (catalog.live_models || []).length;
  const warnings = [];
  if (catalog.live_error) {
    warnings.push(`Catalog refresh degraded: ${catalog.live_error}`);
  }
  if (!liveCount) {
    warnings.push("No live Ollama models reported.");
  }

  els.catalogStatus.innerHTML = `
    <div class="catalog-banner">
      <div>
        <strong>${liveCount}</strong> live models on <span class="mono">${escapeHtml(catalog.base_url || "n/a")}</span>
      </div>
      <div class="catalog-banner-copy">${warnings.join(" ") || "Role chains are editable. Apply writes a fresh OpenClaw config."}</div>
    </div>
  `;

  els.roleGrid.innerHTML = ROLE_ORDER.map((role) => renderRoleCard(role, catalog)).join("");
  bindRoleControls(catalog);
}

function renderRoleCard(role, catalog) {
  const specs = getRoleSpecs(catalog, role);
  const selected = roleSelections[role] || [];
  const currentValue = selected[0] || "";
  const optionModels = new Map();
  specs.forEach((spec) => optionModels.set(spec.model, spec));
  getLiveEntries(catalog).forEach((entry) => {
    if (!optionModels.has(entry.model)) {
      optionModels.set(entry.model, {
        model: entry.model,
        source: "live",
        available: true,
      });
    }
  });
  const options = [...optionModels.values()]
    .map((spec) => {
      const availability = spec.source === "live" ? "live" : spec.available ? "manifest+live" : "manifest";
      return `
        <option value="${escapeHtml(spec.model)}" ${spec.model === currentValue ? "selected" : ""}>
          ${escapeHtml(spec.model)} · ${availability}
        </option>
      `;
    })
    .join("");
  const selectedChips = selected.length
    ? selected
        .map(
          (model, index) => `
            <div class="model-chip">
              <button class="chip-move" data-role="${role}" data-direction="up" data-index="${index}" ${index === 0 ? "disabled" : ""}>↑</button>
              <span class="mono">${escapeHtml(model)}</span>
              <button class="chip-move" data-role="${role}" data-direction="down" data-index="${index}" ${index === selected.length - 1 ? "disabled" : ""}>↓</button>
              <button class="chip-remove" data-role="${role}" data-index="${index}">Remove</button>
            </div>
          `,
        )
        .join("")
    : `<p class="panel-note">No override selected. Manifest default will be used.</p>`;

  const manifestList = specs
    .map(
      (spec) => `
        <div class="catalog-row">
          <div>
            <strong class="mono">${escapeHtml(spec.model)}</strong>
            <div class="catalog-meta">${escapeHtml((spec.capabilities || []).join(" · ") || "no capabilities")}</div>
          </div>
          <div class="catalog-flags">
            <span class="mini-badge ${spec.available ? "live" : "ghost"}">${spec.available ? "live" : "manifest-only"}</span>
            ${spec.active ? '<span class="mini-badge active">active</span>' : ""}
          </div>
        </div>
      `,
    )
    .join("");

  return `
    <section class="role-card">
      <div class="role-head">
        <div>
          <h4>${ROLE_LABELS[role]}</h4>
          <p class="panel-note">Ordered failover chain for ${ROLE_LABELS[role].toLowerCase()} work.</p>
        </div>
      </div>
      <div class="role-editor">
        <label class="field-label" for="role-select-${role}">Add model</label>
        <div class="role-picker">
          <select id="role-select-${role}" data-role-select="${role}">
            ${options}
          </select>
          <button class="ghost accent role-add" type="button" data-role="${role}">Add</button>
        </div>
        <div class="selected-stack">${selectedChips}</div>
      </div>
      <div class="catalog-list">${manifestList}</div>
    </section>
  `;
}

function renderLiveInventory(snapshot) {
  const entries = getLiveEntries(snapshot.model_catalog || {});
  if (!entries.length) {
    els.liveInventory.innerHTML = `<div class="inventory-card"><p class="panel-note">No live Ollama models detected.</p></div>`;
    return;
  }
  els.liveInventory.innerHTML = `
    <div class="inventory-card">
      <div class="panel-subhead compact">
        <h3>Live Inventory</h3>
        <p class="panel-note">All models currently visible from Ollama.</p>
      </div>
      <div class="inventory-list">
        ${entries
          .map(
            (entry) => `
              <div class="inventory-row">
                <strong class="mono">${escapeHtml(entry.model)}</strong>
                <span class="inventory-meta">${escapeHtml(
                  [entry.family, entry.parameter_size, entry.quantization_level].filter(Boolean).join(" · ") || "live model",
                )}</span>
              </div>
            `,
          )
          .join("")}
      </div>
    </div>
  `;
}

function bindRoleControls(catalog) {
  document.querySelectorAll(".role-add").forEach((button) => {
    button.addEventListener("click", () => {
      const role = button.dataset.role;
      const select = document.querySelector(`[data-role-select="${role}"]`);
      const value = select?.value;
      if (!role || !value) {
        return;
      }
      const current = roleSelections[role] || [];
      if (!current.includes(value)) {
        roleSelections[role] = [...current, value];
        renderCatalog(latestSnapshot);
      }
    });
  });

  document.querySelectorAll(".chip-remove").forEach((button) => {
    button.addEventListener("click", () => {
      const role = button.dataset.role;
      const index = Number(button.dataset.index);
      const current = [...(roleSelections[role] || [])];
      current.splice(index, 1);
      roleSelections[role] = current;
      renderCatalog(latestSnapshot);
    });
  });

  document.querySelectorAll(".chip-move").forEach((button) => {
    button.addEventListener("click", () => {
      const role = button.dataset.role;
      const direction = button.dataset.direction;
      const index = Number(button.dataset.index);
      const current = [...(roleSelections[role] || [])];
      const swapIndex = direction === "up" ? index - 1 : index + 1;
      if (swapIndex < 0 || swapIndex >= current.length) {
        return;
      }
      [current[index], current[swapIndex]] = [current[swapIndex], current[index]];
      roleSelections[role] = current;
      renderCatalog(latestSnapshot);
    });
  });
}

async function refreshDashboard() {
  try {
    latestSnapshot = await fetchDashboard();
    renderHero(latestSnapshot);
    renderMetrics(latestSnapshot);
    renderProfile(latestSnapshot);
    renderCatalog(latestSnapshot);
    renderLiveInventory(latestSnapshot);
    renderJobs(latestSnapshot);
    renderRuns(latestSnapshot);
  } catch (error) {
    els.heroStatus.innerHTML = `
      <div class="status-pill">Error</div>
      <p class="mono">${escapeHtml(error.message)}</p>
    `;
  }
}

async function onActionClick(event) {
  const workflow = event.currentTarget.dataset.workflow;
  event.currentTarget.disabled = true;
  try {
    const payload = workflow === "install-openclaw-cloud" ? buildInstallPayload() : {};
    await submitWorkflow(workflow, payload);
    await refreshDashboard();
  } catch (error) {
    window.alert(error.message);
  } finally {
    event.currentTarget.disabled = false;
  }
}

async function applyModelSelections() {
  els.applyModels.disabled = true;
  try {
    await submitWorkflow("install-openclaw-cloud", buildInstallPayload());
    await refreshDashboard();
  } catch (error) {
    window.alert(error.message);
  } finally {
    els.applyModels.disabled = false;
  }
}

function resetModelSelections() {
  if (!latestSnapshot?.model_catalog) {
    return;
  }
  roleSelections = {};
  initRoleSelections(latestSnapshot.model_catalog);
  renderCatalog(latestSnapshot);
}

els.refresh.addEventListener("click", refreshDashboard);
els.buttons.forEach((button) => button.addEventListener("click", onActionClick));
els.applyModels.addEventListener("click", applyModelSelections);
els.resetModels.addEventListener("click", resetModelSelections);

refreshDashboard();
pollHandle = window.setInterval(refreshDashboard, 8000);

window.addEventListener("beforeunload", () => {
  if (pollHandle) {
    window.clearInterval(pollHandle);
  }
});
