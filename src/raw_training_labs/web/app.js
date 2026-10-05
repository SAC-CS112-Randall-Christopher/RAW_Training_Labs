"use strict";
let token = "",
  page = "projects",
  selected = "",
  state = null,
  models = [],
  projects = [],
  editing = null,
  loadTicket = 0,
  renderTicket = 0;
const $ = (selector) => document.querySelector(selector);
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const pretty = (v) => esc(JSON.stringify(v, null, 2));
const gib = (v) => `${(v / 1024 ** 3).toFixed(2)} GiB`;
const pct = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const show = (message, error = false) => {
  $("#message").hidden = false;
  $("#message").textContent = message;
  $("#message").className = error ? "error" : "";
};
async function api(path, payload) {
  const response = await fetch(`/api${path}`, {
    method: payload === undefined ? "GET" : "POST",
    headers:
      payload === undefined
        ? {}
        : { "Content-Type": "application/json", "X-RAW-CSRF": token },
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok)
    throw new Error(
      result.error || JSON.stringify(result.detail) || "Request failed",
    );
  return result;
}
function bind(selector, event, fn) {
  const el = $(selector);
  if (el)
    el.addEventListener(event, async (e) => {
      const button = el.tagName === "BUTTON" ? el : e.submitter,
        originalText = button?.textContent,
        wasDisabled = button?.disabled;
      if (button) {
        button.disabled = true;
        button.setAttribute("aria-busy", "true");
        button.textContent = `${originalText} …`;
      }
      try {
        await fn(e);
      } catch (error) {
        show(error.message, true);
      } finally {
        if (button?.isConnected) {
          button.disabled = wasDisabled;
          button.removeAttribute("aria-busy");
          button.textContent = originalText;
        }
      }
    });
}
async function load() {
  const owner = selected,
    ticket = ++loadTicket;
  const incomingProjects = await api("/projects");
  if (owner !== selected || ticket !== loadTicket) return;
  projects = incomingProjects;
  $("#project").innerHTML =
    '<option value="">Choose a project</option>' +
    projects
      .map(
        (p) =>
          `<option value="${p.id}" ${p.id === selected ? "selected" : ""}>${esc(p.name)}</option>`,
      )
      .join("");
  const incomingState = owner ? await api(`/projects/${owner}`) : null;
  if (owner !== selected || ticket !== loadTicket) return;
  state = incomingState;
  await render();
}
function context() {
  if (!state)
    return '<div class="card empty">Choose a project or create one in Projects to continue.</div>';
  return `<div class="card"><h2>${esc(state.project.name)}</h2><p class="muted">${esc(state.project.customer)} · ${esc(state.project.task)}</p><p><strong>Next action:</strong> ${esc(state.next_action)}</p><span class="mono">${selected}</span></div>`;
}
function selectJob(kind, status = "completed") {
  return state?.jobs.find((j) => j.kind === kind && j.status === status);
}
function corpus() {
  return state?.artifacts.find((a) => a.kind === "corpus");
}
function tag(status) {
  return `<span class="tag ${status === "completed" ? "good" : ["failed", "interrupted"].includes(status) ? "bad" : "warning"}">${esc(status)}</span>`;
}
function jobCards(kinds = null) {
  const visibleJobs = kinds
    ? state.jobs.filter((j) => kinds.includes(j.kind))
    : state.jobs;
  if (!visibleJobs.length) return '<p class="muted">No model jobs yet.</p>';
  return visibleJobs
    .map(
      (
        j,
      ) => `<div class="card"><div class="actions"><h3>${esc(j.kind)}</h3>${tag(j.status)}</div><p class="mono">${j.id}</p>
    <p>${esc(j.reason || j.progress?.phase || "Job recorded")}</p>${j.training ? `<p>Successful steps: <strong>${j.training.step ?? 0} / 6</strong> · selected step ${j.training.best_step ?? "pending"}</p>` : ""}
    ${j.progress?.cases ? `<p>Evaluation progress: ${j.progress.cases.length} cases retained</p>` : ""}
    <p class="muted">${esc(j.output_location)}</p><div class="actions">${["running", "starting", "cancelling"].includes(j.status) ? `<button class="secondary control" data-job="${j.id}" data-action="cancel">Cancel job</button>${j.kind === "train" ? `<button class="secondary control" data-job="${j.id}" data-action="yield">Save checkpoint &amp; pause</button>` : ""}` : ""}
    <button class="secondary log" data-job="${j.id}">Inspect log</button></div><details><summary>Recorded progress and configuration</summary><pre>${pretty({ request: j.request, progress: j.progress, training: j.training })}</pre></details></div>`,
    )
    .join("");
}
async function render() {
  const owner = selected,
    snapshot = state,
    renderedPage = page,
    ticket = ++renderTicket;
  const current = () =>
    owner === selected &&
    snapshot === state &&
    page === renderedPage &&
    ticket === renderTicket;
  const title = {
    projects: "Projects",
    data: "Data & examples",
    training: "Training",
    compare: "Compare & test",
    delivery: "Delivery",
    models: "Model coverage",
  }[page];
  $("#page-title").textContent = title;
  document
    .querySelectorAll("nav button")
    .forEach((b) => b.classList.toggle("selected", b.dataset.page === page));
  const content = $("#content");
  if (page === "projects") {
    content.innerHTML = `<div class="card"><h2>Start with the customer task</h2><p class="muted">A configured base model, retrieval, tools or training may be appropriate. Measure the baseline before committing to new weights.</p>
      <div class="actions"><button id="demo-create">Create synthetic service-note demo</button></div></div>
      <div class="card"><h2>New project</h2><form id="project-form"><div class="grid"><label>Project name<input name="name" required maxlength="160"></label><label>Customer / owner<input name="customer" required maxlength="160"></label></div>
      <label>Task<textarea name="task" required minlength="10" rows="2"></textarea></label><div class="grid"><label>Initial approach<select name="approach"><option value="baseline">Configured baseline</option><option value="retrieval">Customer documents / retrieval</option><option value="tools">Business systems / tools</option><option value="fine_tuning">Fine-tuning hypothesis</option><option value="combination">Combination</option></select></label>
      <label>Scoring task<select name="scoring"><option value="work_request">Structured work request</option><option value="decision">Single-choice decision</option><option value="exact_text">Exact text</option></select></label>
      <label>Required fields<input name="fields" value="site,trade,urgency,summary" required></label><label>Minimum task success<input name="success" type="number" min="0" max="1" step="0.05" value="0.8" required></label></div>
      <button type="submit">Create project</button></form></div>${projects.map((p) => `<div class="card"><h3>${esc(p.name)}</h3><p>${esc(p.task)}</p><p class="muted">${esc(p.customer)}</p><button class="secondary open-project" data-id="${p.id}">Open project</button></div>`).join("")}`;
    bind("#demo-create", "click", async () => {
      const r = await api("/demo", { create: true });
      selected = r.project.id;
      page = "data";
      show(
        "Synthetic originals created. Review is required; no training was started.",
      );
      await load();
    });
    bind("#project-form", "submit", async (e) => {
      e.preventDefault();
      const f = new FormData(e.target);
      const r = await api("/projects", {
        name: f.get("name"),
        customer: f.get("customer"),
        task: f.get("task"),
        approach: f.get("approach"),
        criteria: {
          task: f.get("scoring"),
          fields: String(f.get("fields"))
            .split(",")
            .map((s) => s.trim()),
          minimum_success: Number(f.get("success")),
        },
      });
      selected = r.id;
      page = "data";
      show("Project created.");
      await load();
    });
    document.querySelectorAll(".open-project").forEach(
      (b) =>
        (b.onclick = async () => {
          selected = b.dataset.id;
          page = "data";
          await load();
        }),
    );
    return;
  }
  if (page === "models") {
    content.innerHTML =
      `<div class="card"><h2>Explicit capability profiles</h2><p>Fast decisions and deliberate reasoning describe intended behavior. Thinking mode, task format and training compatibility are recorded separately.</p><p class="muted">Initial methods: compatible supervised LoRA, with QLoRA gated by model and hardware support. GRPO and other reinforcement learning methods are a later milestone. Non-generative classifier/reranker training is unsupported.</p></div>` +
      models
        .map(
          (m) =>
            `<div class="card"><h2>${esc(m.label)}</h2>${tag(m.status)}<p>${esc(m.model)} · ${esc(m.capability)} · ${esc(m.mode)}</p><p class="muted">${esc(m.runtime)}<br>${esc(m.license)}</p><p class="mono">${esc(m.revision)}</p><ul>${m.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>${m.source ? `<a href="${esc(m.source)}" target="_blank" rel="noreferrer">Upstream model card</a>` : ""}</div>`,
        )
        .join("");
    return;
  }
  if (!state) {
    content.innerHTML = context();
    return;
  }
  if (page === "data") {
    const [examples, validation] = await Promise.all([
      api(`/projects/${owner}/examples`),
      api(`/projects/${owner}/validation`),
    ]);
    if (!current()) return;
    content.innerHTML =
      context() +
      `<div class="card"><h2>Review before preparation</h2><div class="grid">${Object.entries(
        validation.counts,
      )
        .map(
          ([s, n]) =>
            `<div class="metric"><strong>${n}</strong><span>${esc(s)}</span></div>`,
        )
        .join("")}</div>
      <p>${validation.eligible ? "Data checks passed." : `${validation.problems.length} review or data issues require attention.`}</p><div class="actions"><button id="author-review" class="secondary" ${corpus() ? "disabled" : ""}>Review synthetic examples (Codex author)</button><button id="prepare" ${!validation.eligible || corpus() ? "disabled" : ""}>Freeze reviewed data</button></div><details><summary>Validation issues and limits</summary><pre>${pretty(validation)}</pre></details></div>
      <div class="card"><h2>Import neutral examples</h2><p class="muted">JSON object with an examples array, or JSONL. Include prompt, completion, group, split, availability, source and explicit use permissions. Initial limit: 64 cases. Imported examples remain pending.</p><textarea id="import-text" rows="5" placeholder='{"examples": [...]}'></textarea><div class="actions"><button id="import">Import examples</button></div><details><summary>Neutral format example</summary><pre>${pretty({ examples: [{ id: "case-001", prompt: [{ role: "user", content: "Site: Unit 12. Trade: plumbing. Issue: tap drips." }], completion: '{"site":"Unit 12","trade":"plumbing","urgency":"routine","summary":"tap drips"}', group: "request-001", split: "train", available_at: 100, source_kind: "customer", rights: { training: true, evaluation: true, external_processing: false, sharing: false, export: false, retention: true, basis: "Record the actual permission basis here" } }] })}</pre></details></div>
      <div class="card"><h2>Originals &amp; teaching targets</h2><table><thead><tr><th>Case / split</th><th>Original input</th><th>Review</th><th></th></tr></thead><tbody>${examples.map((e) => `<tr><td class="mono">${esc(e.id)}<br>${esc(e.original.split)}</td><td>${esc(e.original.prompt.at(-1).content)}<details><summary>Targets and permissions</summary><pre>${pretty({ original: e.original.completion, reviewed: e.review?.completion, rights: e.review?.rights || e.original.rights })}</pre></details></td><td>${tag(e.review?.approved ? "approved" : "pending")}<br>Revision ${e.revision || 0}<br>${esc(e.review?.reviewer || "")}</td><td><button class="secondary review" data-id="${esc(e.id)}" ${corpus() ? "disabled" : ""}>Review</button></td></tr>`).join("")}</tbody></table></div>`;
    bind("#author-review", "click", async () => {
      await api(`/projects/${selected}/demo-review`, { review: true });
      show(
        "Synthetic author review recorded as delegated instructional review; no human review is implied.",
      );
      await load();
    });
    bind("#prepare", "click", async () => {
      const r = await api(`/projects/${selected}/prepare`, { freeze: true });
      show(`Frozen corpus ${r.id}. Final targets are now sealed.`);
      await load();
    });
    bind("#import", "click", async () => {
      const text = $("#import-text").value.trim();
      let body;
      try {
        body = JSON.parse(text);
      } catch {
        body = { examples: text.split("\n").map((line) => JSON.parse(line)) };
      }
      if (Array.isArray(body)) body = { examples: body };
      await api(`/projects/${selected}/examples`, body);
      show("Originals imported; review required.");
      await load();
    });
    document.querySelectorAll(".review").forEach(
      (b) =>
        (b.onclick = () => {
          if (!current()) return;
          editing = {
            ...examples.find((e) => e.id === b.dataset.id),
            project_id: owner,
          };
          $("#review-id").textContent = editing.id;
          $("#review-form").elements.completion.value =
            editing.review?.completion || editing.original.completion;
          $("#review-form").elements.reason.value = "";
          const rights = editing.review?.rights || editing.original.rights;
          $("#rights-controls").innerHTML =
            Object.entries(rights)
              .filter(([k]) => k !== "basis")
              .map(
                ([k, v]) =>
                  `<label class="check"><input type="checkbox" name="${k}" ${v ? "checked" : ""}>Permission: ${esc(k.replaceAll("_", " "))}</label>`,
              )
              .join("") +
            `<label>Permission basis<textarea name="basis" rows="2" required>${esc(rights.basis)}</textarea></label>`;
          $("#review-dialog").showModal();
        }),
    );
    return;
  }
  if (page === "training") {
    const base = selectJob("baseline"),
      candidate = selectJob("train"),
      paused = state.jobs.find((j) => j.kind === "train" && j.resume_supported),
      active = state.jobs.some((j) =>
        ["starting", "running", "cancelling"].includes(j.status),
      );
    content.innerHTML =
      context() +
      `<div class="card"><h2>Supported execution profile</h2><label>Model / intended behavior<select id="profile">${models.map((m) => `<option value="${m.id}" ${m.status !== "supported" ? "disabled" : ""}>${esc(m.label)}${m.status !== "supported" ? " · unavailable for training" : ""}</option>`).join("")}</select></label>
      <p>CPU LoRA · 2 threads · 8 GiB memory ceiling · 6 successful steps · completion-only labels · no silent truncation.</p><p class="muted">Model files require explicit acquisition. Browser closure does not stop an admitted job. Interrupted jobs require a verified clean checkpoint; an abrupt stop may require a new candidate.</p><div class="actions"><button id="acquire" class="secondary">Acquire pinned demo model</button><button id="preflight" class="secondary" ${!corpus() ? "disabled" : ""}>Check data &amp; hardware</button></div><div id="preflight-result"></div>
      <label class="check"><input type="checkbox" id="authorize">I authorize this CPU job within the displayed resource limits</label><label class="check"><input type="checkbox" id="pause-three">Pause training after step 3 to check recovery</label>
      <div class="actions"><button id="baseline" ${!corpus() || active ? "disabled" : ""}>Evaluate baseline</button><button id="train" ${!base || active ? "disabled" : ""}>Train candidate</button><button id="resume" class="secondary" ${!paused || active ? "disabled" : ""}>Resume clean checkpoint</button></div><p class="muted">${candidate ? "Candidate completed; use Compare & test to judge task quality." : "A completed configured baseline is required before training."}</p></div><div id="job-list">${jobCards()}</div>`;
    bind("#acquire", "click", async (e) => {
      e.target.disabled = true;
      try {
        const r = await api(`/projects/${owner}/acquire`, {
          authorized: true,
          profile_id: $("#profile").value,
          idempotency_key: crypto.randomUUID(),
        });
        show(`Acquisition job ${r.id} recorded. No training is started.`);
        if (owner === selected) await load();
      } finally {
        e.target.disabled = false;
      }
    });
    bind("#preflight", "click", async () => {
      if (!current()) return;
      const profileId = $("#profile").value,
        corpusId = corpus().id;
      const r = await api(
        `/projects/${owner}/preflight/${corpusId}/${profileId}`,
      );
      if (
        !current() ||
        $("#profile").value !== profileId ||
        corpus()?.id !== corpusId
      )
        return;
      $("#preflight-result").innerHTML =
        `<p>${esc(r.reason)}</p><details open><summary>Limits, labels and environment</summary><pre>${pretty(r)}</pre></details>`;
    });
    for (const [button, kind, resume] of [
      ["baseline", "baseline", null],
      ["train", "train", null],
      ["resume", "train", paused?.id],
    ])
      bind(`#${button}`, "click", async () => {
        if (!current()) return;
        if (!$("#authorize").checked)
          throw new Error(
            "Authorize the displayed CPU resource envelope first.",
          );
        const body = {
          kind,
          corpus_id: corpus().id,
          profile_id: $("#profile").value,
          authorized: true,
          idempotency_key: crypto.randomUUID(),
          baseline_id:
            kind === "train"
              ? resume
                ? paused.request.baseline_id
                : base?.id
              : null,
          resume_id: resume,
          candidate_id: null,
          yield_after_step:
            kind === "train" && !resume && $("#pause-three").checked ? 3 : null,
        };
        const r = await api(`/projects/${owner}/jobs`, body);
        show(`Job ${r.id} recorded. You may close and reopen the workbench.`);
        if (owner === selected) await load();
      });
    wireControls();
    return;
  }
  if (page === "compare") {
    const final = selectJob("compare"),
      candidate = selectJob("train"),
      baseline = candidate
        ? state.jobs.find((j) => j.id === candidate.request.baseline_id)
        : selectJob("baseline"),
      comparison = final?.result?.comparison,
      active = state.jobs.some((j) =>
        ["starting", "running", "cancelling"].includes(j.status),
      );
    content.innerHTML =
      context() +
      `<div class="card"><h2>Matched task comparison</h2><p>Same frozen inputs, model profile, thinking mode, generation settings and per-case seeds. Every invalid or failed call stays in the denominator.</p><label class="check"><input type="checkbox" id="authorize-compare">Authorize final held-out comparison; results may inform review, not a new test-selected checkpoint</label><button id="compare" ${!baseline || !candidate || active ? "disabled" : ""}>Compare baseline &amp; candidate</button></div><div id="job-list">${jobCards(["compare"])}</div>`;
    if (comparison) {
      const b = comparison.baseline,
        c = comparison.candidate;
      content.innerHTML += `<div class="card"><h2>${comparison.criteria_passed ? "Declared criteria passed" : "Candidate remains in development"}</h2><table><thead><tr><th>Measure</th><th>Baseline</th><th>Candidate</th></tr></thead><tbody>${[
        ["Task success", pct(b.success_rate), pct(c.success_rate)],
        ["Field accuracy", pct(b.field_accuracy), pct(c.field_accuracy)],
        ["Invalid outputs", b.invalid_outputs, c.invalid_outputs],
        [
          "Unsupported field claims",
          b.unsupported_claims,
          c.unsupported_claims,
        ],
        ["Call failures", b.call_failures, c.call_failures],
        [
          "p95 response time",
          `${b.p95_seconds.toFixed(2)} s`,
          `${c.p95_seconds.toFixed(2)} s`,
        ],
        [
          "Observed memory",
          gib(b.peak_observed_rss_bytes),
          gib(c.peak_observed_rss_bytes),
        ],
      ]
        .map((r) => `<tr>${r.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`)
        .join(
          "",
        )}</tbody></table><p>${comparison.regressions} regressions · ${comparison.improvements} improved cases</p><p class="muted">${esc(comparison.next_action)}. Training loss is reported separately.</p></div>`;
      content.innerHTML += comparison.pairs
        .map(
          (p) =>
            `<div class="card"><h3>${esc(p.id)} · ${esc(p.split)} ${p.regression ? tag("regression") : ""}</h3><p>${esc(p.candidate.input.at(-1).content)}</p><div class="split"><div><h3>Baseline</h3><pre>${esc(p.baseline.raw)}</pre><p>${pct(p.baseline.score.field_matches / p.baseline.score.field_total)} fields · ${p.baseline.latency_seconds.toFixed(2)} s</p></div><div><h3>Candidate</h3><pre>${esc(p.candidate.raw)}</pre><p>${pct(p.candidate.score.field_matches / p.candidate.score.field_total)} fields · ${p.candidate.latency_seconds.toFixed(2)} s</p></div></div><details><summary>Reviewed target and call evidence</summary><pre>${pretty({ target: p.candidate.expected, baseline: p.baseline.score, candidate: p.candidate.score, status: p.candidate.status })}</pre></details></div>`,
        )
        .join("");
      const assessments = await api(
        `/projects/${owner}/jobs/${final.id}/assessments`,
      );
      if (!current()) return;
      content.innerHTML += `<div class="card"><h2>Human review / disagreement</h2><form id="assessment-form"><label>Case<select name="case_id">${comparison.pairs.map((p) => `<option>${esc(p.id)}</option>`).join("")}</select></label><label>Reviewer<input name="reviewer" required></label><label>Assessment<select name="assessment"><option value="uncertain">Uncertain</option><option value="candidate_better">Candidate better</option><option value="baseline_better">Baseline better</option><option value="equivalent">Equivalent</option></select></label><label>Evidence and reason<textarea name="reason" minlength="10" required rows="2"></textarea></label><button>Record review</button></form><pre>${pretty(assessments)}</pre></div>`;
      bind("#assessment-form", "submit", async (e) => {
        e.preventDefault();
        await api(
          `/projects/${selected}/jobs/${final.id}/assessments`,
          Object.fromEntries(new FormData(e.target)),
        );
        show("Review retained independently of automatic scoring.");
        await load();
      });
    } else if (baseline?.result) {
      content.innerHTML += `<div class="card"><h2>Baseline before training</h2><pre>${pretty(baseline.result.baseline.metrics)}</pre><p class="muted">This configured baseline used validation and regression inputs. The paired comparison uses the held-out and regression inputs.</p></div>`;
    }
    bind("#compare", "click", async () => {
      if (!$("#authorize-compare").checked)
        throw new Error("Authorize final held-out comparison first.");
      const r = await api(`/projects/${selected}/jobs`, {
        kind: "compare",
        corpus_id: corpus().id,
        authorized: true,
        idempotency_key: crypto.randomUUID(),
        baseline_id: baseline.id,
        candidate_id: candidate.id,
      });
      show(`Comparison job ${r.id} recorded.`);
      await load();
    });
    wireControls();
    return;
  }
  if (page === "delivery") {
    const final = selectJob("compare"),
      baseline = state.jobs.find((j) => j.id === final?.request.baseline_id),
      candidate = state.jobs.find((j) => j.id === final?.request.candidate_id),
      active = state.jobs.some((j) =>
        ["starting", "running", "cancelling"].includes(j.status),
      ),
      delivery = state.jobs.find(
        (j) =>
          j.kind === "export" &&
          j.status === "completed" &&
          j.request.candidate_id === candidate?.id &&
          j.request.baseline_id === baseline?.id,
      );
    content.innerHTML =
      context() +
      `<div class="card"><h2>Package &amp; independently verify</h2><p>Export the selected PEFT adapter with exact base identity, tokenizer, template, recipe, evaluation, licenses, limitations and runtime requirements. A separate process loads the exported package and checks a retained candidate result.</p><p class="muted">Model packages may retain source information. Export permissions are checked. Runtime verification and task-quality acceptance are separate.</p><label class="check"><input id="authorize-export" type="checkbox">Authorize sensitive development-package export and local runtime verification</label><button id="export" ${!final || active ? "disabled" : ""}>Export &amp; verify package</button></div><div id="job-list">${jobCards(["export"])}</div>${delivery ? `<div class="card"><h2>${delivery.status === "completed" ? "Runtime verified" : "Export " + esc(delivery.status)}</h2>${tag(delivery.status)}<pre>${pretty(delivery.result || delivery.progress)}</pre><p class="muted">No installation or activation performed. Customer acceptance remains a separate decision.</p></div>` : ""}`;
    bind("#export", "click", async () => {
      if (!$("#authorize-export").checked)
        throw new Error(
          "Authorize package export and runtime verification first.",
        );
      const r = await api(`/projects/${selected}/jobs`, {
        kind: "export",
        corpus_id: corpus().id,
        authorized: true,
        idempotency_key: crypto.randomUUID(),
        baseline_id: baseline.id,
        candidate_id: candidate.id,
      });
      show(`Export job ${r.id} recorded.`);
      await load();
    });
    wireControls();
  }
}
function wireControls() {
  document.querySelectorAll(".control").forEach(
    (b) =>
      (b.onclick = async () => {
        try {
          await api(
            `/projects/${selected}/jobs/${b.dataset.job}/${b.dataset.action}`,
            { request: true },
          );
          show("Recovery action recorded.");
          await load();
        } catch (e) {
          show(e.message, true);
        }
      }),
  );
  document.querySelectorAll(".log").forEach(
    (b) =>
      (b.onclick = async () => {
        try {
          const r = await api(
            `/projects/${selected}/jobs/${b.dataset.job}/log`,
          );
          const pre = document.createElement("pre");
          pre.textContent = r.text;
          b.closest(".card").append(pre);
        } catch (e) {
          show(e.message, true);
        }
      }),
  );
}
bind("#review-form", "submit", async (e) => {
  e.preventDefault();
  if (!editing || editing.project_id !== selected)
    throw new Error("Project changed; reopen the review in its own project.");
  const reviewedExample = editing,
    owner = editing.project_id,
    exampleId = editing.id,
    f = new FormData(e.target),
    rights = { basis: f.get("basis") };
  for (const k of [
    "training",
    "evaluation",
    "external_processing",
    "sharing",
    "export",
    "retention",
  ])
    rights[k] = f.get(k) === "on";
  await api(
    `/projects/${owner}/examples/${encodeURIComponent(exampleId)}/review`,
    {
      reviewer: f.get("reviewer"),
      reviewer_kind: f.get("reviewer_kind"),
      reviewer_authored_material: f.get("reviewer_authored_material") === "on",
      approved: f.get("approved") === "on",
      completion: f.get("completion"),
      reason: f.get("reason"),
      rights,
    },
  );
  if (editing === reviewedExample && owner === selected) {
    $("#review-dialog").close();
    editing = null;
    show("Review revision preserved with the original.");
    await load();
  }
});
bind("#close-review", "click", () => {
  editing = null;
  $("#review-dialog").close();
});
bind("#project", "change", async (e) => {
  selected = e.target.value;
  renderTicket++;
  editing = null;
  $("#review-dialog").close();
  $("#content").replaceChildren();
  state = null;
  await load();
});
bind("#refresh", "click", load);
document.querySelectorAll("nav button").forEach(
  (b) =>
    (b.onclick = async () => {
      page = b.dataset.page;
      try {
        await render();
      } catch (e) {
        show(e.message, true);
      }
    }),
);
(async () => {
  try {
    token = (await api("/session")).token;
    models = await api("/profiles");
    await load();
  } catch (e) {
    show(e.message, true);
  }
})();
setInterval(async () => {
  if (
    !selected ||
    !state?.jobs.some((j) =>
      ["starting", "running", "cancelling"].includes(j.status),
    )
  )
    return;
  const owner = selected;
  try {
    const prior = state.jobs.map((j) => j.status).join(),
      incoming = await api(`/projects/${owner}`);
    if (owner !== selected) return;
    state = incoming;
    if (["training", "compare", "delivery"].includes(page) && $("#job-list")) {
      const kinds =
        page === "training"
          ? null
          : [page === "compare" ? "compare" : "export"];
      $("#job-list").innerHTML = jobCards(kinds);
      wireControls();
    }
    if (prior !== state.jobs.map((j) => j.status).join()) {
      await render();
    }
  } catch (e) {
    show(e.message, true);
  }
}, 3000);
