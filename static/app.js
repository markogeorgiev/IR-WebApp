async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = null;
  try {
    payload = JSON.parse(text);
  } catch (error) {
    payload = null;
  }
  if (!response.ok) {
    if (payload && payload.error) {
      throw new Error(payload.error);
    }
    throw new Error(text || "Request failed");
  }
  if (!payload) {
    throw new Error("Invalid JSON response from server.");
  }
  return payload;
}


const MODEL_GUIDANCE = {
  bm25: "Focus on exact term matches and include key phrases early in the text.",
  tfidf: "Use distinctive terms that differentiate the document from others.",
  sbert: "Improve semantic clarity; add paraphrases of the query intent.",
  "sbert-minilm-v6": "Add concise, semantically aligned sentences.",
  "e5-small-v2": "Include explicit query-style phrasing and relevant synonyms.",
  "ql-dirichlet": "Repeat important terms naturally to increase language model likelihood.",
};

function createDocBlock(index, models, queries) {
  const block = document.createElement("div");
  block.className = "doc-block";
  block.dataset.block = String(index);
  block.innerHTML = `
    <h2>Document ${index}</h2>
    <div class="doc-grid">
      <div>
        <label>Search document</label>
        <input class="doc-search" type="text" placeholder="Search by ID or text" />
        <button class="doc-search-btn">Search</button>
        <div class="doc-results">
          <select class="doc-select"></select>
        </div>
      </div>
      <div>
        <label>Model</label>
        <select class="doc-model"></select>
        <div class="guidance"></div>
      </div>
      <div>
        <label>Query</label>
        <select class="doc-query"></select>
      </div>
    </div>
    <div class="doc-grid">
      <div>
        <label>Title</label>
        <input class="doc-title" type="text" placeholder="Document title" />
      </div>
      <div>
        <label>Abstract</label>
        <textarea class="doc-abstract" placeholder="Document abstract"></textarea>
      </div>
    </div>
    <div class="actions">
      <button class="doc-load-btn">Load document</button>
      <button class="doc-rerank-btn">Save + rerank</button>
    </div>
    <p class="hint doc-status">Status: idle</p>
    <div class="doc-add-more hidden">
      <button class="doc-add-btn" title="Add another document block">+</button>
      <span>Add another document</span>
    </div>
    <div class="rank-summary">
      <div class="rank-metric">Old rank: <span class="old-rank">-</span></div>
      <div class="rank-metric">New rank: <span class="new-rank">-</span></div>
      <div class="rank-metric">Change: <span class="rank-change">-</span></div>
      <div class="rank-metric">Old score: <span class="old-score">-</span></div>
      <div class="rank-metric">New score: <span class="new-score">-</span></div>
    </div>
    <div class="plot-preview">
      <img class="plot-image" alt="KLRVF comparison plot" />
    </div>
  `;

  const modelSelect = block.querySelector(".doc-model");
  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.label;
    option.disabled = !model.available;
    modelSelect.appendChild(option);
  });

  const querySelect = block.querySelector(".doc-query");
  queries.forEach((query) => {
    const option = document.createElement("option");
    option.value = query.id;
    option.textContent = `${query.id}: ${query.text}`;
    querySelect.appendChild(option);
  });

  const guidance = block.querySelector(".guidance");
  const updateGuidance = () => {
    const modelName = modelSelect.value;
    guidance.textContent = MODEL_GUIDANCE[modelName] || "Choose a model to see guidance.";
  };
  updateGuidance();
  modelSelect.addEventListener("change", updateGuidance);

  const searchBtn = block.querySelector(".doc-search-btn");
  const searchInput = block.querySelector(".doc-search");
  const docSelect = block.querySelector(".doc-select");
  const loadBtn = block.querySelector(".doc-load-btn");
  const titleInput = block.querySelector(".doc-title");
  const abstractInput = block.querySelector(".doc-abstract");
  const statusEl = block.querySelector(".doc-status");

  searchBtn.addEventListener("click", async () => {
    const term = searchInput.value.trim();
    if (!term) {
      statusEl.textContent = "Status: enter a search term.";
      return;
    }
    statusEl.textContent = "Status: searching...";
    try {
      const payload = await fetchJson(`/api/docs/search?q=${encodeURIComponent(term)}`);
      docSelect.innerHTML = "";
      payload.results.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.doc_id;
        option.textContent = `${item.doc_id} | ${item.title}`;
        docSelect.appendChild(option);
      });
      statusEl.textContent = `Status: ${payload.results.length} result(s).`;
    } catch (error) {
      statusEl.textContent = `Status: ${error.message}`;
    }
  });

  loadBtn.addEventListener("click", async () => {
    const docId = docSelect.value;
    if (!docId) {
      statusEl.textContent = "Status: select a document.";
      return;
    }
    statusEl.textContent = "Status: loading document...";
    try {
      const payload = await fetchJson(`/api/docs/${encodeURIComponent(docId)}`);
      titleInput.value = payload.title || "";
      abstractInput.value = payload.abstract || "";
      statusEl.textContent = "Status: document loaded.";
    } catch (error) {
      statusEl.textContent = `Status: ${error.message}`;
    }
  });

  const rerankBtn = block.querySelector(".doc-rerank-btn");
  const oldRankEl = block.querySelector(".old-rank");
  const newRankEl = block.querySelector(".new-rank");
  const changeEl = block.querySelector(".rank-change");
  const oldScoreEl = block.querySelector(".old-score");
  const newScoreEl = block.querySelector(".new-score");
  const plotImg = block.querySelector(".plot-image");
  const addMore = block.querySelector(".doc-add-more");
  const addBtn = block.querySelector(".doc-add-btn");

  rerankBtn.addEventListener("click", async () => {
    const docId = docSelect.value;
    if (!docId) {
      statusEl.textContent = "Status: select a document.";
      return;
    }
    statusEl.textContent = "Status: reranking...";
    try {
      const payload = await fetchJson("/api/docs/rerank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          doc_id: docId,
          title: titleInput.value,
          abstract: abstractInput.value,
          model: modelSelect.value,
          query_id: querySelect.value,
        }),
      });
      oldRankEl.textContent = payload.old_rank ?? "-";
      newRankEl.textContent = payload.new_rank ?? "-";
      changeEl.textContent = payload.rank_change ?? "-";
      oldScoreEl.textContent =
        payload.old_score === null || payload.old_score === undefined
          ? "-"
          : payload.old_score.toFixed(4);
      newScoreEl.textContent =
        payload.new_score === null || payload.new_score === undefined
          ? "-"
          : payload.new_score.toFixed(4);
      plotImg.src = `${payload.plot_url}?t=${Date.now()}`;
      plotImg.classList.add("plot-image");
      if (payload.plot_error) {
        statusEl.textContent = `Status: rerank complete (plot error: ${payload.plot_error}).`;
      } else {
        statusEl.textContent = "Status: rerank complete.";
      }
      if (addMore) {
        addMore.classList.remove("hidden");
      }
      setupPlotModal();
    } catch (error) {
      statusEl.textContent = `Status: ${error.message}`;
    }
  });

  if (addBtn) {
    addBtn.addEventListener("click", () => {
      const event = new CustomEvent("doc:add");
      window.dispatchEvent(event);
    });
  }

  return block;
}

async function initDocEditorPage() {
  const container = document.getElementById("doc-blocks");
  const addBtn = document.getElementById("add-doc-block");
  if (!container || !addBtn) {
    return;
  }

  const [modelsPayload, queriesPayload] = await Promise.all([
    fetchJson("/api/models"),
    fetchJson("/api/queries"),
  ]);

  let index = 1;
  const addBlock = () => {
    const block = createDocBlock(index, modelsPayload.models, queriesPayload.queries);
    container.appendChild(block);
    index += 1;
  };

  addBtn.addEventListener("click", addBlock);
  window.addEventListener("doc:add", addBlock);
  addBlock();
}
function initIndexPage() {
  const form = document.getElementById("query-form");
  const status = document.getElementById("upload-status");
  const useDefaultBtn = document.getElementById("use-default");
  const sourceEl = document.getElementById("current-source");
  const countEl = document.getElementById("current-count");

  async function refreshSummary() {
    try {
      const payload = await fetchJson("/api/queries");
      if (sourceEl) {
        sourceEl.textContent = payload.source;
      }
      if (countEl) {
        countEl.textContent = payload.count;
      }
    } catch (error) {
      if (status) {
        status.textContent = error.message;
      }
    }
  }

  refreshSummary();

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    status.textContent = "Uploading...";
    const formData = new FormData(form);
    formData.append("source", "upload");
    try {
      const payload = await fetchJson("/api/queries", {
        method: "POST",
        body: formData,
      });
      status.textContent = `Uploaded ${payload.count} queries.`;
      refreshSummary();
    } catch (error) {
      status.textContent = error.message;
    }
  });

  useDefaultBtn.addEventListener("click", async () => {
    status.textContent = "Loading default queries...";
    const formData = new FormData();
    formData.append("source", "default");
    try {
      const payload = await fetchJson("/api/queries", {
        method: "POST",
        body: formData,
      });
      status.textContent = `Loaded ${payload.count} default queries.`;
      refreshSummary();
    } catch (error) {
      status.textContent = error.message;
    }
  });
}

function renderResults(results) {
  const container = document.getElementById("results");
  container.innerHTML = "";
  if (!results.length) {
    container.textContent = "No results.";
    return;
  }

  const table = document.createElement("table");
  table.className = "results-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Rank</th>
        <th>Score</th>
        <th>Text</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;
  const tbody = table.querySelector("tbody");

  results.forEach((item, idx) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td class="rank"><span class="rank-badge">${idx + 1}</span></td>
      <td class="score">${item.score.toFixed(4)}</td>
      <td>${item.text}</td>
    `;
    row.addEventListener("click", () => {
      openResultModal(item, idx + 1);
    });
    tbody.appendChild(row);
  });
  container.appendChild(table);
}

function logConsole(message) {
  const consoleEl = document.getElementById("model-console");
  if (!consoleEl) {
    return;
  }
  const line = document.createElement("div");
  line.className = "console-line";
  const now = new Date();
  const stamp = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  line.innerHTML = `<strong>[${stamp}]</strong> ${message}`;
  consoleEl.appendChild(line);
  consoleEl.scrollTop = consoleEl.scrollHeight;
}

function clearConsole() {
  const consoleEl = document.getElementById("model-console");
  if (consoleEl) {
    consoleEl.innerHTML = "";
  }
}

function logRankingConsole(message) {
  const consoleEl = document.getElementById("ranking-console");
  if (!consoleEl) {
    return;
  }
  const line = document.createElement("div");
  line.className = "console-line";
  const now = new Date();
  const stamp = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  line.innerHTML = `<strong>[${stamp}]</strong> ${message}`;
  consoleEl.appendChild(line);
  consoleEl.scrollTop = consoleEl.scrollHeight;
}

function clearRankingConsole() {
  const consoleEl = document.getElementById("ranking-console");
  if (consoleEl) {
    consoleEl.innerHTML = "";
  }
}

async function updateAnalysisStatus() {
  const statusEl = document.getElementById("analysis-status");
  if (!statusEl) {
    return;
  }
  try {
    const payload = await fetchJson("/api/analysis/status");
    const status = payload.status || "idle";
    let message = `Status: ${status}`;
    if (payload.error) {
      message += ` (${payload.error})`;
    }
    statusEl.textContent = message;
  } catch (error) {
    statusEl.textContent = `Status: ${error.message}`;
  }
}

async function initModelsPage() {
  const querySelect = document.getElementById("query-select");
  const modelSelect = document.getElementById("model-select");
  const querySource = document.getElementById("query-source");
  const modelStatus = document.getElementById("model-status");
  const rankBtn = document.getElementById("rank-btn");
  const topKInput = document.getElementById("top-k");
  const clearBtn = document.getElementById("clear-embedding-cache");
  const modal = document.getElementById("result-modal");
  const modalClose = document.getElementById("modal-close");
  const modalTitle = document.getElementById("modal-title");
  const modalMeta = document.getElementById("modal-meta");
  const modalText = document.getElementById("modal-text");
  const storageKey = "nfcorpus_model_selection";

  try {
    const queryPayload = await fetchJson("/api/queries");
    querySource.textContent = `Loaded ${queryPayload.count} queries from ${queryPayload.source}.`;
    querySelect.innerHTML = "";
    queryPayload.queries.forEach((query) => {
      const option = document.createElement("option");
      option.value = query.id;
      option.textContent = `${query.id}: ${query.text}`;
      querySelect.appendChild(option);
    });
    const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
    if (saved.queryId) {
      querySelect.value = saved.queryId;
    }
    logConsole(
      `Loaded <strong>${queryPayload.count}</strong> queries from <em>${queryPayload.source}</em>.`
    );
  } catch (error) {
    querySource.textContent = error.message;
    logConsole(`Query load failed: <em>${error.message}</em>`);
  }

  try {
    const modelPayload = await fetchJson("/api/models");
    modelSelect.innerHTML = "";
    modelPayload.models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = model.label;
      option.disabled = !model.available;
      modelSelect.appendChild(option);
    });
    const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
    if (saved.model) {
      modelSelect.value = saved.model;
    }
    modelStatus.textContent = "Disabled models need dependencies installed.";
    logConsole(
      `Models online: ${modelPayload.models
        .filter((model) => model.available)
        .map((model) => `<strong>${model.label}</strong>`)
        .join(", ")}.`
    );
  } catch (error) {
    modelStatus.textContent = error.message;
    logConsole(`Model load failed: <em>${error.message}</em>`);
  }

  if (topKInput) {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
    if (saved.topK) {
      topKInput.value = saved.topK;
    }
  }

  const persistSelection = () => {
    localStorage.setItem(
      storageKey,
      JSON.stringify({
        queryId: querySelect.value,
        model: modelSelect.value,
        topK: topKInput?.value || "10",
      })
    );
  };
  querySelect.addEventListener("change", persistSelection);
  modelSelect.addEventListener("change", persistSelection);
  if (topKInput) {
    topKInput.addEventListener("change", persistSelection);
  }

  rankBtn.addEventListener("click", async () => {
    const queryId = querySelect.value;
    const model = modelSelect.value;
    const topKRaw = topKInput?.value || "10";
    const topK = topKRaw === "all" ? "all" : Math.max(1, parseInt(topKRaw, 10));
    persistSelection();
    modelStatus.textContent = "Ranking...";
    const queryLabel =
      querySelect.options[querySelect.selectedIndex]?.textContent || queryId;
    clearConsole();
    logConsole("Starting ranking job...");
    logConsole(`Model selected: <strong>${model.toUpperCase()}</strong>`);
    logConsole(`Query selected: <em>${queryLabel}</em>`);
    logConsole(`Top-K requested: <strong>${topK}</strong>`);
    logConsole("Dispatching request to ranking service...");
    try {
      const payload = await fetchJson("/api/rank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, query_id: queryId, top_k: topK }),
      });
      modelStatus.textContent = `Results for: ${payload.query}`;
      renderResults(payload.results);
      logConsole("Ranking completed successfully.");
      logConsole(
        `Returned <strong>${payload.results.length}</strong> documents for display.`
      );
    } catch (error) {
      modelStatus.textContent = error.message;
      logConsole("Ranking failed.");
      logConsole(`Error: <em>${error.message}</em>`);
    }
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      const warning1 =
        "This will delete cached SBERT embeddings. Rebuilding can take a long time on CPU-only systems.";
      const warning2 =
        "Are you absolutely sure? This action cannot be undone.";
      if (!window.confirm(warning1)) {
        return;
      }
      if (!window.confirm(warning2)) {
        return;
      }
      modelStatus.textContent = "Clearing embedding cache...";
      logConsole("Clearing embedding cache...");
      try {
        const payload = await fetchJson("/api/cache/embeddings/clear", {
          method: "POST",
        });
        modelStatus.textContent = `Cleared embedding cache (${payload.removed_files} files).`;
        logConsole(
          `Embedding cache cleared. Removed <strong>${payload.removed_files}</strong> files.`
        );
      } catch (error) {
        modelStatus.textContent = error.message;
        logConsole(`Cache clear failed: <em>${error.message}</em>`);
      }
    });
  }

  if (modal && modalClose) {
    modalClose.addEventListener("click", () => {
      modal.classList.add("hidden");
    });
    modal.addEventListener("click", (event) => {
      if (event.target === modal) {
        modal.classList.add("hidden");
      }
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        modal.classList.add("hidden");
      }
    });
  }

  window.openResultModal = (item, rank) => {
    if (!modal) {
      return;
    }
    modalTitle.textContent = item.title || item.id;
    modalMeta.textContent = `Rank ${rank} | Score ${item.score.toFixed(4)} | ID ${item.id}`;
    modalText.textContent = item.text;
    modal.classList.remove("hidden");
  };
}

async function initAnalysisPage() {
  const runBtn = document.getElementById("run-analysis");
  await updateAnalysisStatus();
  if (!runBtn) {
    return;
  }
  runBtn.addEventListener("click", async () => {
    const statusEl = document.getElementById("analysis-status");
    statusEl.textContent = "Status: running";
    try {
      await fetchJson("/api/analysis/run", { method: "POST" });
    } catch (error) {
      statusEl.textContent = `Status: ${error.message}`;
      return;
    }
    const interval = setInterval(async () => {
      await updateAnalysisStatus();
      const payload = await fetchJson("/api/analysis/status");
      if (payload.status !== "running") {
        clearInterval(interval);
      }
    }, 3000);
  });
}

function setupPlotModal() {
  const modal = document.getElementById("plot-modal");
  const modalClose = document.getElementById("plot-modal-close");
  const modalImage = document.getElementById("plot-modal-image");
  const modalContent = modal?.querySelector(".modal-content");
  if (!modal || !modalClose || !modalImage) {
    return;
  }
  modalClose.addEventListener("click", () => {
    modal.classList.add("hidden");
  });
  modal.addEventListener("click", (event) => {
    if (event.target === modal) {
      modal.classList.add("hidden");
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      modal.classList.add("hidden");
    }
  });

  document.querySelectorAll(".plot-image").forEach((img) => {
    img.addEventListener("click", () => {
      modalImage.src = img.getAttribute("src");
      const zoom = img.getAttribute("data-zoom") || "1.1";
      if (modalContent) {
        modalContent.style.setProperty("--plot-zoom", zoom);
      }
      modal.classList.remove("hidden");
    });
    img.addEventListener("error", () => {
      img.classList.add("hidden");
    });
  });
}

function renderQueryList(queries) {
  const container = document.getElementById("query-list");
  if (!container) {
    return;
  }
  container.innerHTML = "";
  queries.forEach((query) => {
    const row = document.createElement("div");
    row.className = "query-row";
    row.innerHTML = `
      <div class="query-text">
        <label>Query ${query.id}</label>
        <textarea rows="2" data-id="${query.id}">${query.text}</textarea>
      </div>
      <div class="query-actions">
        <button data-action="save" data-id="${query.id}">Save</button>
        <button class="button-danger" data-action="delete" data-id="${query.id}">Delete</button>
      </div>
    `;
    container.appendChild(row);
  });
}

async function initQueriesPage() {
  const statusEl = document.getElementById("query-status");
  const addBtn = document.getElementById("add-query");
  const resetBtn = document.getElementById("reset-queries");
  const input = document.getElementById("new-query");

  async function refresh() {
    const payload = await fetchJson("/api/queries");
    renderQueryList(payload.queries);
  }

  try {
    await refresh();
  } catch (error) {
    statusEl.textContent = error.message;
  }

  document.addEventListener("click", async (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    const action = target.dataset.action;
    const id = target.dataset.id;
    if (!action || !id) {
      return;
    }
    if (action === "save") {
      const textarea = document.querySelector(`textarea[data-id="${id}"]`);
      const text = textarea?.value || "";
      try {
        await fetchJson(`/api/queries/${id}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        statusEl.textContent = `Saved query ${id}.`;
      } catch (error) {
        statusEl.textContent = error.message;
      }
    }
    if (action === "delete") {
      if (!window.confirm("Delete this query?")) {
        return;
      }
      try {
        await fetchJson(`/api/queries/${id}`, { method: "DELETE" });
        statusEl.textContent = `Deleted query ${id}.`;
        await refresh();
      } catch (error) {
        statusEl.textContent = error.message;
      }
    }
  });

  if (addBtn) {
    addBtn.addEventListener("click", async () => {
      const text = input.value.trim();
      if (!text) {
        statusEl.textContent = "Enter a query.";
        return;
      }
      try {
        await fetchJson("/api/queries/add", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        });
        input.value = "";
        statusEl.textContent = "Query added.";
        await refresh();
      } catch (error) {
        statusEl.textContent = error.message;
      }
    });
  }

  if (resetBtn) {
    resetBtn.addEventListener("click", async () => {
      if (!window.confirm("Reset queries to default?")) {
        return;
      }
      try {
        await fetchJson("/api/queries/reset", { method: "POST" });
        statusEl.textContent = "Queries reset to default.";
        await refresh();
      } catch (error) {
        statusEl.textContent = error.message;
      }
    });
  }
}

function renderCoverageTable(data) {
  const container = document.getElementById("coverage-table");
  if (!container) {
    return;
  }
  container.innerHTML = "";
  const table = document.createElement("table");
  table.className = "coverage-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Model</th>
        <th>Queries with rankings</th>
        <th>Coverage</th>
      </tr>
    </thead>
    <tbody></tbody>
  `;
  const tbody = table.querySelector("tbody");
  data.models.forEach((model) => {
    const covered = data.coverage[model.name] || [];
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${model.label}</td>
      <td>${covered.join(", ") || "—"}</td>
      <td><span class="coverage-pill">${covered.length} / ${data.queries.length}</span></td>
    `;
    tbody.appendChild(row);
  });
  container.appendChild(table);
}

function renderEmbeddingCache(data) {
  const container = document.getElementById("embedding-cache");
  if (!container) {
    return;
  }
  const cache = data.embedding_cache || { count: 0, files: [] };
  const list = cache.files.length
    ? cache.files.map((name) => `<li>${name}</li>`).join("")
    : "<li>None</li>";
  container.innerHTML = `
    <p><span class="coverage-pill">${cache.count} cached file(s)</span></p>
    <ul class="cache-list">${list}</ul>
  `;
}

async function initRankingsPage() {
  const statusEl = document.getElementById("rank-status");
  const modelSelect = document.getElementById("rank-model");
  const querySelect = document.getElementById("rank-query");
  const runBtn = document.getElementById("run-ranking");
  const runAllModelsBtn = document.getElementById("run-all-models");
  const runAllQueriesBtn = document.getElementById("run-all-queries");
  const runPlotsBtn = document.getElementById("run-plots");
  const storageKey = "nfcorpus_rankings_selection";
  let rankingsPoll = null;

  async function fetchRankingsStatus() {
    return await fetchJson("/api/rankings/status");
  }

  async function refresh() {
    const payload = await fetchRankingsStatus();
    renderCoverageTable(payload);
    renderEmbeddingCache(payload);
    modelSelect.innerHTML = "";
    querySelect.innerHTML = "";
    payload.models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.name;
      option.textContent = model.label;
      option.disabled = !model.available;
      modelSelect.appendChild(option);
    });
    payload.queries.forEach((query) => {
      const option = document.createElement("option");
      option.value = query.id;
      option.textContent = `${query.id}: ${query.text}`;
      querySelect.appendChild(option);
    });
    const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
    if (saved.model) {
      modelSelect.value = saved.model;
    }
    if (saved.queryId) {
      querySelect.value = saved.queryId;
    }
    logRankingConsole(
      `Loaded <strong>${payload.queries.length}</strong> queries and <strong>${payload.models.length}</strong> models.`
    );
    return payload;
  }

  function stopRankingsPoll() {
    if (rankingsPoll) {
      clearInterval(rankingsPoll);
      rankingsPoll = null;
    }
  }

  function updateRankingState(state, options = {}) {
    if (!state) {
      return;
    }
    const quiet = options.quiet || false;
    if (state.status === "running") {
      statusEl.textContent = "Running rankings...";
      if (!quiet) {
        logRankingConsole("Ranking job is running in the background.");
      }
      return;
    }
    if (state.status === "failed") {
      const message = state.error || "Ranking job failed.";
      statusEl.textContent = message;
      if (!quiet) {
        logRankingConsole(`Ranking failed: <em>${message}</em>`);
      }
      return;
    }
    if (state.status === "completed") {
      statusEl.textContent = `Wrote ${state.rows} ranking rows.`;
      if (!quiet) {
        logRankingConsole(`Wrote <strong>${state.rows}</strong> rows.`);
      }
    }
  }

  async function pollRankingsJob() {
    try {
      const payload = await fetchRankingsStatus();
      const state = payload.rankings_state;
      updateRankingState(state, { quiet: true });
      if (!state || state.status === "running") {
        return;
      }
      stopRankingsPoll();
      updateRankingState(state);
      await refresh();
    } catch (error) {
      stopRankingsPoll();
      statusEl.textContent = error.message;
      logRankingConsole(`Ranking status failed: <em>${error.message}</em>`);
    }
  }

  function startRankingsPoll() {
    if (rankingsPoll) {
      return;
    }
    rankingsPoll = setInterval(pollRankingsJob, 1500);
    pollRankingsJob();
  }

  try {
    const payload = await refresh();
    if (payload.rankings_state?.status === "running") {
      startRankingsPoll();
    }
  } catch (error) {
    statusEl.textContent = error.message;
    logRankingConsole(`Status load failed: <em>${error.message}</em>`);
  }

  const persistSelection = () => {
    localStorage.setItem(
      storageKey,
      JSON.stringify({
        model: modelSelect.value,
        queryId: querySelect.value,
      })
    );
  };
  modelSelect.addEventListener("change", persistSelection);
  querySelect.addEventListener("change", persistSelection);

  async function runRanking(model, queryId) {
    statusEl.textContent = "Running rankings...";
    clearRankingConsole();
    logRankingConsole("Starting ranking generation...");
    logRankingConsole(`Model scope: <strong>${model}</strong>`);
    logRankingConsole(`Query scope: <strong>${queryId}</strong>`);
    persistSelection();
    try {
      const payload = await fetchJson("/api/rankings/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, query_id: queryId }),
      });
      if (payload.status === "running") {
        logRankingConsole("Ranking job already running. Watching for completion...");
        startRankingsPoll();
        return;
      }
      if (payload.status === "started") {
        logRankingConsole("Ranking job started in the background.");
        startRankingsPoll();
        return;
      }
      logRankingConsole("Ranking job accepted.");
      startRankingsPoll();
    } catch (error) {
      statusEl.textContent = error.message;
      logRankingConsole(`Ranking failed: <em>${error.message}</em>`);
    }
  }

  if (runBtn) {
    runBtn.addEventListener("click", async () => {
      await runRanking(modelSelect.value, querySelect.value);
    });
  }
  if (runAllModelsBtn) {
    runAllModelsBtn.addEventListener("click", async () => {
      await runRanking("all", querySelect.value);
    });
  }
  if (runAllQueriesBtn) {
    runAllQueriesBtn.addEventListener("click", async () => {
      await runRanking(modelSelect.value, "all");
    });
  }
  if (runPlotsBtn) {
    runPlotsBtn.addEventListener("click", async () => {
      statusEl.textContent = "Generating plots...";
      clearRankingConsole();
      logRankingConsole("Starting plot generation from existing rankings...");
      try {
        await fetchJson("/api/analysis/plots", { method: "POST" });
        statusEl.textContent = "Plot generation started.";
        logRankingConsole("Plot generation started.");
        await updateAnalysisStatus();
      } catch (error) {
        statusEl.textContent = error.message;
        logRankingConsole(`Plot generation failed: <em>${error.message}</em>`);
      }
    });
  }
}

async function initDashboardPage() {
  await initRankingsPage();
  await updateAnalysisStatus();
  setupPlotModal();
}

const page = document.currentScript.dataset.page;
if (page === "index") {
  initIndexPage();
}
if (page === "models") {
  initModelsPage();
}
if (page === "analysis") {
  initAnalysisPage();
}
if (page === "queries") {
  initQueriesPage();
}
if (page === "rankings") {
  initRankingsPage();
}
if (page === "dashboard") {
  initDashboardPage();
}
if (page === "doc-editor") {
  initDocEditorPage();
}
