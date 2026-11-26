const clinicalInput = document.getElementById("clinicalInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const errorBanner = document.getElementById("errorBanner");
const resultsArea = document.getElementById("resultsArea");
const highlightedText = document.getElementById("highlightedText");
const drugCountBadge = document.getElementById("drugCountBadge");
const eventCountBadge = document.getElementById("eventCountBadge");
const classLabel = document.getElementById("classLabel");
const classConfText = document.getElementById("classConfText");
const classConfBar = document.getElementById("classConfBar");
const sentenceBreakdown = document.getElementById("sentenceBreakdown");
const pairsContainer = document.getElementById("pairsContainer");
const apiStatus = document.getElementById("apiStatus");
const apiStatusText = document.getElementById("apiStatusText");
const refreshStatsBtn = document.getElementById("refreshStatsBtn");

// Tabs
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "analytics") {
      loadStats();
    }
  });
});

// Sample chips
document.querySelectorAll(".sample-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    clinicalInput.value = chip.dataset.sample;
    analyzeText();
  });
});

analyzeBtn.addEventListener("click", analyzeText);
clearBtn.addEventListener("click", () => {
  clinicalInput.value = "";
  errorBanner.textContent = "";
  resultsArea.classList.add("hidden");
});
refreshStatsBtn.addEventListener("click", loadStats);

async function checkHealth() {
  try {
    const res = await fetch("/healthcheck");
    if (!res.ok) throw new Error("Unhealthy");
    const data = await res.json();
    if (data.status === "ok") {
      apiStatus.classList.add("online");
      apiStatus.classList.remove("offline");
      apiStatusText.textContent = "Backend Connected";
      return;
    }
  } catch (_) {
    // handled below
  }
  apiStatus.classList.add("offline");
  apiStatus.classList.remove("online");
  apiStatusText.textContent = "Backend Disconnected";
}

function renderHighlightedSpans(text, spans) {
  highlightedText.textContent = "";
  if (!spans || spans.length === 0) {
    highlightedText.textContent = text;
    return;
  }

  let cursor = 0;
  const sorted = [...spans].sort((a, b) => a.start - b.start);

  for (const span of sorted) {
    if (span.start < cursor) continue;
    if (span.start > cursor) {
      highlightedText.appendChild(document.createTextNode(text.slice(cursor, span.start)));
    }
    const mark = document.createElement("span");
    mark.className = span.label === "drug" ? "hl-drug" : "hl-event";
    mark.textContent = text.slice(span.start, span.end);

    const tag = document.createElement("span");
    tag.className = "hl-tag";
    tag.textContent = span.label === "drug" ? "DRUG" : "ADE";
    mark.appendChild(tag);

    highlightedText.appendChild(mark);
    cursor = span.end;
  }

  if (cursor < text.length) {
    highlightedText.appendChild(document.createTextNode(text.slice(cursor)));
  }
}

function renderClassification(classifyData) {
  const isAde = classifyData.label === "ADE";
  const pct = Math.round((classifyData.confidence || 0) * 100);

  classLabel.textContent = classifyData.label;
  classLabel.className = `class-badge ${isAde ? "ade" : "non-ade"}`;
  classConfText.textContent = `${pct}% confidence`;
  classConfBar.style.width = `${pct}%`;
  classConfBar.className = `progress-fill ${isAde ? "ade" : "non-ade"}`;

  sentenceBreakdown.textContent = "";
  const sentences = classifyData.sentences || [];
  if (sentences.length > 1) {
    sentences.forEach((item) => {
      const row = document.createElement("div");
      row.className = "sentence-item";

      const txt = document.createElement("span");
      txt.textContent = item.sentence;

      const badge = document.createElement("span");
      badge.className = `class-badge ${item.label === "ADE" ? "ade" : "non-ade"}`;
      badge.style.fontSize = "11px";
      badge.style.flexShrink = "0";
      badge.textContent = `${item.label} (${Math.round(item.confidence * 100)}%)`;

      row.appendChild(txt);
      row.appendChild(badge);
      sentenceBreakdown.appendChild(row);
    });
  }
}

function renderPairs(extractData) {
  pairsContainer.textContent = "";
  const pairs = extractData.extractions || [];
  const drugs = extractData.drugs || [];
  const events = extractData.events || [];

  drugCountBadge.textContent = `${drugs.length} Drug${drugs.length === 1 ? "" : "s"}`;
  eventCountBadge.textContent = `${events.length} ADE${events.length === 1 ? "" : "s"}`;

  if (pairs.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    if (drugs.length > 0 && events.length === 0) {
      empty.textContent = `Detected drug(s): ${drugs.join(", ")}. No adverse drug events detected in the text.`;
    } else if (events.length > 0 && drugs.length === 0) {
      empty.textContent = `Detected event(s): ${events.join(", ")}. No drug names matched in the text.`;
    } else {
      empty.textContent = "No drug–ADE pairs detected in this text.";
    }
    pairsContainer.appendChild(empty);
    return;
  }

  const table = document.createElement("table");
  table.className = "data-table";
  table.innerHTML = `
    <thead>
      <tr>
        <th>Drug</th>
        <th>Adverse Event</th>
        <th>Relation</th>
      </tr>
    </thead>
  `;
  const tbody = document.createElement("tbody");
  pairs.forEach((pair) => {
    const tr = document.createElement("tr");

    const tdDrug = document.createElement("td");
    const drugPill = document.createElement("span");
    drugPill.className = "pill pill-drug";
    drugPill.textContent = pair.drug;
    tdDrug.appendChild(drugPill);

    const tdEvent = document.createElement("td");
    const eventPill = document.createElement("span");
    eventPill.className = "pill pill-event";
    eventPill.textContent = pair.event;
    tdEvent.appendChild(eventPill);

    const tdRel = document.createElement("td");
    tdRel.textContent = pair.relation === "adverse_effect" ? "Causal ADE cue" : "Co-occurrence";

    tr.appendChild(tdDrug);
    tr.appendChild(tdEvent);
    tr.appendChild(tdRel);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  pairsContainer.appendChild(table);
}

async function analyzeText() {
  const text = clinicalInput.value.trim();
  errorBanner.textContent = "";
  if (!text) {
    errorBanner.textContent = "Please enter some clinical or biomedical text to analyze.";
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  try {
    const [extractRes, classifyRes] = await Promise.all([
      fetch("/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      }),
      fetch("/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      }),
    ]);

    if (!extractRes.ok || !classifyRes.ok) {
      throw new Error("Backend returned an error response.");
    }

    const extractData = await extractRes.json();
    const classifyData = await classifyRes.json();

    renderHighlightedSpans(text, extractData.spans || []);
    renderClassification(classifyData);
    renderPairs(extractData);

    resultsArea.classList.remove("hidden");
  } catch (err) {
    errorBanner.textContent = `Analysis failed: ${err.message}`;
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze Text";
  }
}

function renderBarChart(container, items, labelKey, fillClass) {
  container.textContent = "";
  if (!items || items.length === 0) {
    container.innerHTML = `<div class="empty-state">No data available.</div>`;
    return;
  }
  const maxVal = Math.max(...items.map((i) => i.count), 1);

  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const label = document.createElement("span");
    label.className = "bar-label";
    label.title = item[labelKey];
    label.textContent = item[labelKey];

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = fillClass;
    fill.style.width = `${Math.max(4, Math.round((item.count / maxVal) * 100))}%`;
    track.appendChild(fill);

    const count = document.createElement("span");
    count.className = "bar-count";
    count.textContent = item.count;

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(count);
    container.appendChild(row);
  });
}

async function loadStats() {
  try {
    const res = await fetch("/stats");
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById("kpiTotalDrugs").textContent = data.summary.total_drugs;
    document.getElementById("kpiTotalMentions").textContent = data.summary.total_ade_mentions;
    document.getElementById("kpiUniqueAdes").textContent = data.summary.unique_ades;
    document.getElementById("kpiCorpusSents").textContent = data.summary.corpus_sentences_extracted;

    renderBarChart(
      document.getElementById("topDrugsChart"),
      data.top_drugs,
      "drug",
      "bar-fill-drug"
    );
    renderBarChart(
      document.getElementById("topEventsChart"),
      data.top_events,
      "event",
      "bar-fill-event"
    );

    const drugsBody = document.getElementById("topDrugsTableBody");
    drugsBody.textContent = "";
    (data.top_drugs || []).forEach((item, idx) => {
      const tr = document.createElement("tr");
      const tdRank = document.createElement("td");
      tdRank.textContent = String(idx + 1);

      const tdDrug = document.createElement("td");
      const pill = document.createElement("span");
      pill.className = "pill pill-drug";
      pill.textContent = item.drug;
      tdDrug.appendChild(pill);

      const tdCount = document.createElement("td");
      tdCount.textContent = String(item.count);

      const tdEvents = document.createElement("td");
      tdEvents.textContent = (item.sample_events || []).join("; ");

      tr.appendChild(tdRank);
      tr.appendChild(tdDrug);
      tr.appendChild(tdCount);
      tr.appendChild(tdEvents);
      drugsBody.appendChild(tr);
    });

    const corpusBody = document.getElementById("corpusTableBody");
    corpusBody.textContent = "";
    (data.corpus_extractions || []).forEach((item) => {
      const tr = document.createElement("tr");

      const tdFile = document.createElement("td");
      tdFile.textContent = item.file || "corpus";

      const tdSent = document.createElement("td");
      tdSent.textContent = item.sentence || "";

      const tdDrugs = document.createElement("td");
      (item.drugs || []).forEach((d) => {
        const p = document.createElement("span");
        p.className = "pill pill-drug";
        p.style.marginRight = "4px";
        p.textContent = d;
        tdDrugs.appendChild(p);
      });

      const tdEvents = document.createElement("td");
      (item.events || []).forEach((e) => {
        const p = document.createElement("span");
        p.className = "pill pill-event";
        p.style.marginRight = "4px";
        p.textContent = e;
        tdEvents.appendChild(p);
      });

      tr.appendChild(tdFile);
      tr.appendChild(tdSent);
      tr.appendChild(tdDrugs);
      tr.appendChild(tdEvents);
      corpusBody.appendChild(tr);
    });
  } catch (_) {
    // ignore background load failure
  }
}

checkHealth();
loadStats();
