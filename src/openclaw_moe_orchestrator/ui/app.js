const dashboardUrl = "/api/dashboard";

const els = {
  heroStatus: document.getElementById("hero-status"),
  metricGrid: document.getElementById("metric-grid"),
  profileCard: document.getElementById("profile-card"),
  jobFeed: document.getElementById("job-feed"),
  runsFeed: document.getElementById("runs-feed"),
  refresh: document.getElementById("refresh-dashboard"),
  buttons: Array.from(document.querySelectorAll("[data-workflow]")),
  reasoningModels: document.getElementById("reasoning-models"),
  codingModels: document.getElementById("coding-models"),
  generalModels: document.getElementById("general-models"),
};

let pollHandle = null;

function linesToList(value) {
  return value
    .split("\n")
    .map((item) => item.trim())
    .filter(Boolean);
}

async function fetchDashboard() {
  const response = await fetch(dashboardUrl);
  if (!response.ok) {
    throw new Error(`Dashboard request failed: ${response.status}`);
  }
  return response.json();
}

async function submitWorkflow(workflow) {
  const payload = { workflow };
  if (workflow === "install-openclaw-cloud") {
    payload.reasoning_models = linesToList(els.reasoningModels.value);
    payload.coding_models = linesToList(els.codingModels.value);
    payload.general_models = linesToList(els.generalModels.value);
  }
  const response = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "request failed" }));
    throw new Error(error.error || "request failed");
  }
  return response.json();
}

function renderHero(snapshot) {
  const active = snapshot.active_profile || {};
  const models = snapshot.environment?.ollama_models || [];
  els.heroStatus.innerHTML = `
    <div class="status-pill">${active.primary ? "Live" : "Needs Setup"}</div>
    <p><strong>${models.length}</strong> models visible through Ollama.</p>
    <p class="mono">${active.primary || "No active primary configured"}</p>
  `;
}

function renderMetrics(snapshot) {
  const environment = snapshot.environment || {};
  const metrics = [
    ["Python", environment.python || "n/a"],
    ["GPU", environment.gpu_name || "CPU"],
    ["Ollama Models", String((environment.ollama_models || []).length)],
    ["OpenClaw Binary", environment.openclaw_binary ? "Detected" : "Missing"],
  ];
  els.metricGrid.innerHTML = metrics
    .map(
      ([label, value]) => `
        <div class="metric-card">
          <div class="metric-label">${label}</div>
          <div class="metric-value">${value}</div>
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
      <div class="pair"><span>Provider</span><strong>${profile.api || "ollama"}</strong></div>
      <div class="pair"><span>Primary</span><strong class="mono">${profile.primary}</strong></div>
      <div class="pair"><span>Fallbacks</span><strong class="mono">${fallbackText}</strong></div>
      <div class="pair"><span>Endpoint</span><strong class="mono">${profile.base_url || "n/a"}</strong></div>
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
          <div class="badge ${job.status}">${job.status}</div>
          <p class="job-title">${job.workflow}</p>
          <div class="job-meta">
            <div class="pair"><span>Job ID</span><strong class="mono">${job.job_id}</strong></div>
            <div class="pair"><span>Created</span><strong>${new Date(job.created_at).toLocaleString()}</strong></div>
            ${
              job.error
                ? `<div class="pair"><span>Error</span><strong class="mono">${job.error}</strong></div>`
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
            `<a href="/repo/${output.path}" target="_blank" rel="noreferrer">${output.name}</a>`,
        )
        .join("");
      return `
        <div class="run-card">
          <p class="run-title">${run.workflow || run.run_id}</p>
          <div class="run-meta">
            <div class="pair"><span>Run ID</span><strong class="mono">${run.run_id}</strong></div>
            <div class="pair"><span>Created</span><strong>${new Date(run.created_at).toLocaleString()}</strong></div>
            <div class="pair"><span>Bundle</span><strong class="mono">${run.bundle_dir}</strong></div>
          </div>
          <div class="run-outputs">${outputs || "<span class=\"panel-note\">No outputs yet.</span>"}</div>
        </div>
      `;
    })
    .join("");
}

async function refreshDashboard() {
  try {
    const snapshot = await fetchDashboard();
    renderHero(snapshot);
    renderMetrics(snapshot);
    renderProfile(snapshot);
    renderJobs(snapshot);
    renderRuns(snapshot);
  } catch (error) {
    els.heroStatus.innerHTML = `
      <div class="status-pill">Error</div>
      <p class="mono">${error.message}</p>
    `;
  }
}

async function onActionClick(event) {
  const workflow = event.currentTarget.dataset.workflow;
  event.currentTarget.disabled = true;
  try {
    await submitWorkflow(workflow);
    await refreshDashboard();
  } catch (error) {
    window.alert(error.message);
  } finally {
    event.currentTarget.disabled = false;
  }
}

els.refresh.addEventListener("click", refreshDashboard);
els.buttons.forEach((button) => button.addEventListener("click", onActionClick));

refreshDashboard();
pollHandle = window.setInterval(refreshDashboard, 8000);

window.addEventListener("beforeunload", () => {
  if (pollHandle) {
    window.clearInterval(pollHandle);
  }
});
