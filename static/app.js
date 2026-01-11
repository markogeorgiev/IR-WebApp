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
  "sbert-minilm-v6": "Add concise, semantically aligned sentences.",
  "e5-small-v2": "Include explicit query-style phrasing and relevant synonyms.",
};

function createModelSection(model, queries) {
  const section = document.createElement("section");
  section.className = "card model-section";
  section.dataset.model = model.name;
  const guidance = MODEL_GUIDANCE[model.name] || "Choose a model to see guidance.";

  section.innerHTML = `
    <div class="model-header">
      <div class="model-meta">
        <h2>${model.label}</h2>
        <p class="hint">${guidance}</p>
      </div>
      <div class="model-controls">
        <div class="model-control">
          <label for="query-${model.name}">Query</label>
          <select id="query-${model.name}" class="model-query"></select>
        </div>
        <button class="button-primary model-add-doc">+ Add document</button>
        <button class="model-rerank-all">Rerank all</button>
        <span class="hint model-status">Status: idle</span>
      </div>
    </div>
    <div class="table-wrap">
      <table class="doc-table">
        <thead>
          <tr>
            <th>Doc ID</th>
            <th>Title</th>
            <th>Abstract</th>
            <th>Old Rank</th>
            <th>New Rank</th>
            <th>Change</th>
            <th>Old Score</th>
            <th>New Score</th>
            <th>Plot</th>
            <th>Remove</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  `;

  const querySelect = section.querySelector(".model-query");
  queries.forEach((query) => {
    const option = document.createElement("option");
    option.value = query.id;
    option.textContent = `${query.id}: ${query.text}`;
    querySelect.appendChild(option);
  });

  const addBtn = section.querySelector(".model-add-doc");
  if (!model.available) {
    section.classList.add("model-disabled");
    addBtn.disabled = true;
    querySelect.disabled = true;
    const hint = document.createElement("p");
    hint.className = "hint model-unavailable";
    hint.textContent = "Model unavailable.";
    section.querySelector(".model-meta")?.appendChild(hint);
  }

  return { section, addBtn };
}

function addDocRow(section, doc) {
  const tbody = section.querySelector("tbody");
  if (!tbody) {
    return null;
  }
  const modelName = section.dataset.model || "unknown";
  const existing = tbody.querySelector(`tr[data-doc-id="${doc.docId}"]`);
  if (existing) {
    return null;
  }

  const row = document.createElement("tr");
  row.dataset.docId = doc.docId;
  row.dataset.title = doc.title;

  const cell = (className, text) => {
    const td = document.createElement("td");
    if (className) {
      td.className = className;
    }
    td.textContent = text;
    return td;
  };

  const abstractCell = document.createElement("td");
  abstractCell.className = "doc-abstract-cell";
  const abstractInput = document.createElement("textarea");
  abstractInput.className = "doc-abstract-input";
  abstractInput.value = doc.abstract || "";
  abstractCell.appendChild(abstractInput);

  const oldRankCell = cell("metric old-rank", "-");
  const newRankCell = cell("metric new-rank", "-");
  const changeCell = cell("metric rank-change", "-");
  const oldScoreCell = cell("metric old-score", "-");
  const newScoreCell = cell("metric new-score", "-");

  const plotCell = document.createElement("td");
  const plotBtn = document.createElement("button");
  plotBtn.className = "plot-open";
  plotBtn.textContent = "View";
  plotBtn.disabled = true;
  plotCell.appendChild(plotBtn);

  const removeCell = document.createElement("td");
  const removeBtn = document.createElement("button");
  removeBtn.className = "button-danger doc-remove";
  removeBtn.textContent = "Remove";
  removeCell.appendChild(removeBtn);

  const statusCell = cell("doc-status", "Status: idle");

  row.appendChild(cell("doc-id", doc.docId));
  row.appendChild(cell("doc-title", doc.title));
  row.appendChild(abstractCell);
  row.appendChild(oldRankCell);
  row.appendChild(newRankCell);
  row.appendChild(changeCell);
  row.appendChild(oldScoreCell);
  row.appendChild(newScoreCell);
  row.appendChild(plotCell);
  row.appendChild(removeCell);
  row.appendChild(statusCell);

  const persistEdits = () => {
    const storageKey = `nfcorpus_doc_edits_${modelName}`;
    const rows = Array.from(section.querySelectorAll("tbody tr")).map((entry) => {
      const abstract = entry.querySelector(".doc-abstract-input")?.value || "";
      return {
        doc_id: entry.dataset.docId,
        title: entry.dataset.title || "",
        abstract,
      };
    });
    localStorage.setItem(storageKey, JSON.stringify(rows));
  };

  const persistRerank = (payload) => {
    const storageKey = `nfcorpus_doc_rerank_${modelName}`;
    const current = JSON.parse(localStorage.getItem(storageKey) || "{}");
    current[doc.docId] = {
      old_rank: payload.old_rank ?? null,
      new_rank: payload.new_rank ?? null,
      rank_change: payload.rank_change ?? null,
      old_score: payload.old_score ?? null,
      new_score: payload.new_score ?? null,
      plot_url: payload.plot_url || null,
    };
    localStorage.setItem(storageKey, JSON.stringify(current));
  };

  const applyPersistedRerank = () => {
    const storageKey = `nfcorpus_doc_rerank_${modelName}`;
    const current = JSON.parse(localStorage.getItem(storageKey) || "{}");
    const cached = current[doc.docId];
    if (!cached) {
      return;
    }
    row.querySelector(".old-rank").textContent = cached.old_rank ?? "-";
    row.querySelector(".new-rank").textContent = cached.new_rank ?? "-";
    row.querySelector(".rank-change").textContent = cached.rank_change ?? "-";
    row.querySelector(".old-score").textContent =
      cached.old_score === null || cached.old_score === undefined
        ? "-"
        : Number(cached.old_score).toFixed(4);
    row.querySelector(".new-score").textContent =
      cached.new_score === null || cached.new_score === undefined
        ? "-"
        : Number(cached.new_score).toFixed(4);
    if (cached.plot_url) {
      row.dataset.plotUrl = `${cached.plot_url}?t=${Date.now()}`;
      plotBtn.disabled = false;
    }
  };

  abstractInput.addEventListener("input", () => {
    persistEdits();
  });

  plotBtn.addEventListener("click", () => {
    const plotUrl = row.dataset.plotUrl;
    if (!plotUrl) {
      return;
    }
    openPlotModal(plotUrl, "1.1");
  });

  removeBtn.addEventListener("click", () => {
    const storageKey = `nfcorpus_doc_edits_${modelName}`;
    const rerankKey = `nfcorpus_doc_rerank_${modelName}`;
    const rows = Array.from(section.querySelectorAll("tbody tr"))
      .filter((entry) => entry !== row)
      .map((entry) => {
        const abstract = entry.querySelector(".doc-abstract-input")?.value || "";
        return {
          doc_id: entry.dataset.docId,
          title: entry.dataset.title || "",
          abstract,
        };
      });
    localStorage.setItem(storageKey, JSON.stringify(rows));
    const cached = JSON.parse(localStorage.getItem(rerankKey) || "{}");
    delete cached[doc.docId];
    localStorage.setItem(rerankKey, JSON.stringify(cached));
    row.remove();
  });

  tbody.appendChild(row);
  applyPersistedRerank();
  persistEdits();
  return row;
}

async function rerankSection(section) {
  const statusEl = section.querySelector(".model-status");
  const querySelect = section.querySelector(".model-query");
  const rerankBtn = section.querySelector(".model-rerank-all");
  const rows = Array.from(section.querySelectorAll("tbody tr"));
  if (!querySelect?.value) {
    statusEl.textContent = "Status: select a query.";
    return;
  }
  if (!rows.length) {
    statusEl.textContent = "Status: add at least one document.";
    return;
  }

  const docs = rows.map((row) => {
    const abstractInput = row.querySelector(".doc-abstract-input");
    return {
      doc_id: row.dataset.docId,
      title: row.dataset.title || "",
      abstract: abstractInput?.value || "",
    };
  });

  statusEl.textContent = `Status: reranking ${docs.length} document(s)...`;
  rows.forEach((row) => {
    const statusCell = row.querySelector(".doc-status");
    if (statusCell) {
      statusCell.textContent = "Status: queued.";
    }
  });
  if (rerankBtn) {
    rerankBtn.disabled = true;
  }

  try {
    const payload = await fetchJson("/api/docs/rerank/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: section.dataset.model,
        query_id: querySelect.value,
        docs,
      }),
    });

    payload.results.forEach((result) => {
      const row = section.querySelector(`tr[data-doc-id="${result.doc_id}"]`);
      if (!row) {
        return;
      }
      const statusCell = row.querySelector(".doc-status");
      if (result.error) {
        if (statusCell) {
          statusCell.textContent = `Status: ${result.error}`;
        }
        return;
      }
      row.querySelector(".old-rank").textContent = result.old_rank ?? "-";
      row.querySelector(".new-rank").textContent = result.new_rank ?? "-";
      row.querySelector(".rank-change").textContent = result.rank_change ?? "-";
      row.querySelector(".old-score").textContent =
        result.old_score === null || result.old_score === undefined
          ? "-"
          : result.old_score.toFixed(4);
      row.querySelector(".new-score").textContent =
        result.new_score === null || result.new_score === undefined
          ? "-"
          : result.new_score.toFixed(4);
      const plotBtn = row.querySelector(".plot-open");
      row.dataset.plotUrl = `${result.plot_url}?t=${Date.now()}`;
      if (plotBtn) {
        plotBtn.disabled = false;
      }
      if (statusCell) {
        statusCell.textContent = "Status: rerank complete.";
      }
      const modelName = section.dataset.model || "unknown";
      const storageKey = `nfcorpus_doc_rerank_${modelName}`;
      const current = JSON.parse(localStorage.getItem(storageKey) || "{}");
      current[result.doc_id] = {
        old_rank: result.old_rank ?? null,
        new_rank: result.new_rank ?? null,
        rank_change: result.rank_change ?? null,
        old_score: result.old_score ?? null,
        new_score: result.new_score ?? null,
        plot_url: result.plot_url || null,
      };
      localStorage.setItem(storageKey, JSON.stringify(current));
    });

    if (payload.plot_error) {
      statusEl.textContent = `Status: rerank complete (plot error: ${payload.plot_error}).`;
    } else {
      statusEl.textContent = "Status: rerank complete.";
    }
  } catch (error) {
    statusEl.textContent = `Status: ${error.message}`;
  } finally {
    if (rerankBtn) {
      rerankBtn.disabled = false;
    }
  }
}

async function initDocEditorPage() {
  const container = document.getElementById("model-sections");
  const modal = document.getElementById("doc-modal");
  if (!container || !modal) {
    return;
  }

  const csvInput = document.getElementById("doc-csv");
  const csvBtn = document.getElementById("doc-csv-upload");
  const csvStatus = document.getElementById("doc-csv-status");

  const [modelsPayload, queriesPayload] = await Promise.all([
    fetchJson("/api/models"),
    fetchJson("/api/queries"),
  ]);

  const modalModelLabel = document.getElementById("doc-modal-model");
  const modalClose = document.getElementById("doc-modal-close");
  const modalCancel = document.getElementById("doc-cancel-btn");
  const searchInput = document.getElementById("doc-search-input");
  const searchBtn = document.getElementById("doc-search-btn");
  const docSelect = document.getElementById("doc-select");
  const loadBtn = document.getElementById("doc-load-btn");
  const titleInput = document.getElementById("doc-title-input");
  const abstractInput = document.getElementById("doc-abstract-input");
  const addBtn = document.getElementById("doc-add-btn");
  const statusEl = document.getElementById("doc-modal-status");
  let activeSection = null;

  const resetModal = () => {
    if (searchInput) {
      searchInput.value = "";
    }
    if (docSelect) {
      docSelect.innerHTML = "";
    }
    if (titleInput) {
      titleInput.value = "";
    }
    if (abstractInput) {
      abstractInput.value = "";
    }
    if (statusEl) {
      statusEl.textContent = "Ready.";
    }
  };

  const closeModal = () => {
    modal.classList.add("hidden");
    activeSection = null;
  };

  const openDocModal = (section, label) => {
    activeSection = section;
    if (modalModelLabel) {
      modalModelLabel.textContent = label || "Model";
    }
    resetModal();
    modal.classList.remove("hidden");
  };

  const modelSections = new Map();
  modelsPayload.models.forEach((model) => {
    const { section, addBtn } = createModelSection(
      model,
      queriesPayload.queries
    );
    container.appendChild(section);
    modelSections.set(model.name, section);
    addBtn.addEventListener("click", () => {
      openDocModal(section, model.label);
    });
    const rerankBtn = section.querySelector(".model-rerank-all");
    if (rerankBtn) {
      rerankBtn.addEventListener("click", async () => {
        await rerankSection(section);
      });
    }
    const storageKey = `nfcorpus_doc_edits_${model.name}`;
    const savedRows = JSON.parse(localStorage.getItem(storageKey) || "[]");
    if (Array.isArray(savedRows)) {
      savedRows.forEach((entry) => {
        if (!entry?.doc_id) {
          return;
        }
        addDocRow(section, {
          docId: entry.doc_id,
          title: entry.title || "",
          abstract: entry.abstract || "",
        });
      });
    }
  });

  modalClose?.addEventListener("click", closeModal);
  modalCancel?.addEventListener("click", closeModal);
  modal.addEventListener("click", (event) => {
    if (event.target === modal) {
      closeModal();
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeModal();
    }
  });

  searchBtn?.addEventListener("click", async () => {
    const term = searchInput?.value.trim() || "";
    if (!term) {
      statusEl.textContent = "Enter a search term.";
      return;
    }
    statusEl.textContent = "Searching...";
    try {
      const payload = await fetchJson(`/api/docs/search?q=${encodeURIComponent(term)}`);
      if (docSelect) {
        docSelect.innerHTML = "";
        payload.results.forEach((item) => {
          const option = document.createElement("option");
          option.value = item.doc_id;
          option.textContent = `${item.doc_id} | ${item.title}`;
          docSelect.appendChild(option);
        });
      }
      statusEl.textContent = `Found ${payload.results.length} result(s).`;
    } catch (error) {
      statusEl.textContent = error.message;
    }
  });

  loadBtn?.addEventListener("click", async () => {
    const docId = docSelect?.value;
    if (!docId) {
      statusEl.textContent = "Select a document.";
      return;
    }
    statusEl.textContent = "Loading document...";
    try {
      const payload = await fetchJson(`/api/docs/${encodeURIComponent(docId)}`);
      if (titleInput) {
        titleInput.value = payload.title || "";
      }
      if (abstractInput) {
        abstractInput.value = payload.abstract || "";
      }
      statusEl.textContent = "Document loaded.";
    } catch (error) {
      statusEl.textContent = error.message;
    }
  });

  addBtn?.addEventListener("click", () => {
    const docId = docSelect?.value;
    if (!docId) {
      statusEl.textContent = "Select a document first.";
      return;
    }
    if (!activeSection) {
      statusEl.textContent = "Select a model first.";
      return;
    }
    const title = titleInput?.value.trim() || "";
    const abstract = abstractInput?.value || "";
    if (!title) {
      statusEl.textContent = "Load the document details first.";
      return;
    }
    const row = addDocRow(activeSection, { docId, title, abstract });
    if (!row) {
      statusEl.textContent = "Document already added.";
      return;
    }
    closeModal();
  });

  const parseCsv = (text) => {
    const rows = [];
    let row = [];
    let current = "";
    let inQuotes = false;
    for (let i = 0; i < text.length; i += 1) {
      const char = text[i];
      const next = text[i + 1];
      if (char === "\"" && inQuotes && next === "\"") {
        current += "\"";
        i += 1;
        continue;
      }
      if (char === "\"") {
        inQuotes = !inQuotes;
        continue;
      }
      if (char === "," && !inQuotes) {
        row.push(current);
        current = "";
        continue;
      }
      if ((char === "\n" || char === "\r") && !inQuotes) {
        if (current || row.length) {
          row.push(current);
          rows.push(row);
          row = [];
          current = "";
        }
        continue;
      }
      current += char;
    }
    if (current || row.length) {
      row.push(current);
      rows.push(row);
    }
    return rows.filter((r) => r.some((cell) => cell.trim() !== ""));
  };

  const normalizeModel = (name) => {
    const key = (name || "").trim().toLowerCase();
    if (key === "sbert") {
      return "sbert-minilm-v6";
    }
    if (key === "e5") {
      return "e5-small-v2";
    }
    return key;
  };

  const loadCsvFile = async () => {
    if (!csvInput || !csvInput.files?.length) {
      if (csvStatus) {
        csvStatus.textContent = "Select a CSV file first.";
      }
      return;
    }
    const file = csvInput.files[0];
    const text = await file.text();
    const rows = parseCsv(text);
    if (!rows.length) {
      if (csvStatus) {
        csvStatus.textContent = "CSV is empty.";
      }
      return;
    }
    const header = rows[0].map((cell) => cell.trim().toLowerCase());
    const colIndex = (name) => header.indexOf(name);
    const modelIdx = colIndex("model");
    const docIdx = colIndex("doc_id");
    const titleIdx = colIndex("title");
    const abstractIdx = colIndex("abstract");
    if (modelIdx === -1 || docIdx === -1 || titleIdx === -1 || abstractIdx === -1) {
      if (csvStatus) {
        csvStatus.textContent =
          "CSV must include headers: model, doc_id, title, abstract.";
      }
      return;
    }
    let added = 0;
    let skipped = 0;
    rows.slice(1).forEach((row) => {
      const modelName = normalizeModel(row[modelIdx]);
      const docId = (row[docIdx] || "").trim();
      if (!modelName || !docId) {
        skipped += 1;
        return;
      }
      const section = modelSections.get(modelName);
      if (!section) {
        skipped += 1;
        return;
      }
      const title = row[titleIdx] || "";
      const abstract = row[abstractIdx] || "";
      const result = addDocRow(section, { docId, title, abstract });
      if (result) {
        added += 1;
      } else {
        skipped += 1;
      }
    });
    if (csvStatus) {
      csvStatus.textContent = `Loaded ${added} documents (${skipped} skipped).`;
    }
  };

  if (csvBtn) {
    csvBtn.addEventListener("click", async () => {
      await loadCsvFile();
    });
  }

  if (csvInput) {
    csvInput.addEventListener("change", async () => {
      if (csvInput.files?.length) {
        await loadCsvFile();
      }
    });
  }

  setupPlotModal();
}
function initIndexPage() {
  const form = document.getElementById("query-form");
  const status = document.getElementById("upload-status");
  const useDefaultBtn = document.getElementById("use-default");
  const sourceEl = document.getElementById("current-source");
  const countEl = document.getElementById("current-count");
  const fileInput = document.getElementById("query_file");
  const overlay = document.getElementById("drop-overlay");

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

  const uploadFormData = async (formData) => {
    status.textContent = "Uploading...";
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
  };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(form);
    formData.append("source", "upload");
    await uploadFormData(formData);
  });

  if (useDefaultBtn) {
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

  if (overlay) {
    let dragDepth = 0;
    const showOverlay = () => {
      overlay.classList.remove("hidden");
    };
    const hideOverlay = () => {
      overlay.classList.add("hidden");
    };

    const hasFiles = (event) =>
      Array.from(event.dataTransfer?.types || []).includes("Files");

    document.addEventListener("dragenter", (event) => {
      if (!hasFiles(event)) {
        return;
      }
      event.preventDefault();
      dragDepth += 1;
      showOverlay();
    });

    document.addEventListener("dragover", (event) => {
      if (!hasFiles(event)) {
        return;
      }
      event.preventDefault();
    });

    document.addEventListener("dragleave", (event) => {
      if (!hasFiles(event)) {
        return;
      }
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) {
        hideOverlay();
      }
    });

    document.addEventListener("drop", async (event) => {
      if (!hasFiles(event)) {
        return;
      }
      event.preventDefault();
      dragDepth = 0;
      hideOverlay();
      const file = event.dataTransfer?.files?.[0];
      if (!file) {
        return;
      }
      if (fileInput) {
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
      }
      const formData = new FormData();
      formData.append("source", "upload");
      formData.append("file", file);
      await uploadFormData(formData);
    });
  }
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
      <td class="result-text">
        <div class="result-id">ID: ${item.id}</div>
        ${item.text}
      </td>
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
        "This will delete cached dense embeddings (SBERT/E5). Rebuilding can take a long time on CPU-only systems.";
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

let plotModal = null;
let plotModalImage = null;
let plotModalContent = null;

function openPlotModal(src, zoom = "1.1") {
  if (!plotModal || !plotModalImage) {
    return;
  }
  plotModalImage.src = src;
  if (plotModalContent) {
    plotModalContent.style.setProperty("--plot-zoom", zoom);
  }
  plotModal.classList.remove("hidden");
}

function setupPlotModal() {
  plotModal = document.getElementById("plot-modal");
  const modalClose = document.getElementById("plot-modal-close");
  plotModalImage = document.getElementById("plot-modal-image");
  plotModalContent = plotModal?.querySelector(".modal-content");
  if (!plotModal || !modalClose || !plotModalImage) {
    return;
  }
  modalClose.addEventListener("click", () => {
    plotModal.classList.add("hidden");
  });
  plotModal.addEventListener("click", (event) => {
    if (event.target === plotModal) {
      plotModal.classList.add("hidden");
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      plotModal.classList.add("hidden");
    }
  });

  document.querySelectorAll(".plot-image").forEach((img) => {
    img.addEventListener("click", () => {
      const src = img.getAttribute("src");
      if (!src) {
        return;
      }
      const zoom = img.getAttribute("data-zoom") || "1.1";
      openPlotModal(src, zoom);
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

  const clearBtn = document.getElementById("clear-embedding-cache");
  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      const warning1 =
        "This will delete cached dense embeddings (SBERT/E5). Rebuilding can take a long time on CPU-only systems.";
      const warning2 =
        "Are you absolutely sure? This action cannot be undone.";
      if (!window.confirm(warning1)) {
        return;
      }
      if (!window.confirm(warning2)) {
        return;
      }
      statusEl.textContent = "Clearing embedding cache...";
      logRankingConsole("Clearing embedding cache...");
      try {
        const payload = await fetchJson("/api/cache/embeddings/clear", {
          method: "POST",
        });
        statusEl.textContent = `Cleared embedding cache (${payload.removed_files} files).`;
        logRankingConsole(
          `Embedding cache cleared. Removed <strong>${payload.removed_files}</strong> files.`
        );
      } catch (error) {
        statusEl.textContent = error.message;
        logRankingConsole(`Cache clear failed: <em>${error.message}</em>`);
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
