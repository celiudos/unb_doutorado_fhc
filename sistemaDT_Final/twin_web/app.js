/* Painel do Digital Twin — SPA leve (perfil piloto, SPEC 10.4). */
"use strict";

const S = { token: null, role: null, user: null, modelId: null, model: null, charts: {} };
const $ = (sel) => document.querySelector(sel);
const NAMES = { SV: "Valores Compartilhados", SG: "Estratégia", SU: "Estrutura", SM: "Sistemas",
                SF: "Equipe", SY: "Estilo de Liderança", SK: "Habilidades", TP: "Proteção Tecnológica" };
const EXO = ["SV", "SG", "SU", "SM", "SF", "SY", "SK"];

async function api(path, opts = {}) {
  const headers = { ...(opts.headers || {}) };
  if (S.token) headers["Authorization"] = `Bearer ${S.token}`;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.json);
  }
  const res = await fetch(path, { ...opts, headers });
  if (res.status === 401) { logout(); throw new Error("Sessão expirada"); }
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))).detail;
    throw new Error(detail || `Erro ${res.status}`);
  }
  return res.headers.get("content-type")?.includes("json") ? res.json() : res.text();
}

function fmt(x, d = 2) { return Number(x).toFixed(d); }
function fmtCI(ci, d = 2) { return `[${fmt(ci[0], d)}; ${fmt(ci[1], d)}]`; }

/* ---------- login ---------- */

$("#login-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("#login-error").textContent = "";
  try {
    const body = new URLSearchParams({ username: $("#login-user").value, password: $("#login-pass").value });
    const res = await fetch("/auth/token", { method: "POST", body });
    if (!res.ok) throw new Error("Usuário ou senha inválidos");
    const data = await res.json();
    S.token = data.access_token; S.role = data.role; S.user = $("#login-user").value;
    $("#login-view").hidden = true; $("#app-view").hidden = false;
    $("#user-info").textContent = `${S.user} (${S.role})`;
    document.querySelectorAll("[data-analyst]").forEach(b => { b.disabled = S.role === "viewer"; });
    $("#card-import").hidden = S.role !== "admin";
    await loadModels();
  } catch (e) { $("#login-error").textContent = e.message; }
});

function logout() {
  S.token = null;
  $("#app-view").hidden = true; $("#login-view").hidden = false;
}
$("#logout").addEventListener("click", logout);

$("#login-info-btn").addEventListener("click", () => {
  const pop = $("#login-info-pop");
  pop.hidden = !pop.hidden;
  $("#login-info-btn").setAttribute("aria-expanded", String(!pop.hidden));
});

/* ---------- tabs ---------- */

document.querySelectorAll("nav#tabs button").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll("nav#tabs button").forEach(b => b.classList.remove("active"));
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  btn.classList.add("active");
  $(`#tab-${btn.dataset.tab}`).classList.add("active");
  // gráficos criados com a aba oculta medem 0x0; redimensiona ao exibir
  requestAnimationFrame(() => Object.values(S.charts).forEach(c => c.resize()));
}));

/* ---------- modelos ---------- */

async function loadModels() {
  const models = await api("/models");
  const sel = $("#model-select");
  sel.innerHTML = "";
  if (!models.length) {
    sel.innerHTML = "<option>Nenhum modelo — rode scripts/seed.py</option>";
    return;
  }
  for (const m of models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.name} · N=${m.n} · R²=${fmt(m.r2, 3)}`;
    sel.appendChild(opt);
  }
  sel.onchange = () => selectModel(sel.value);
  await selectModel(models[0].id);
}

async function selectModel(id) {
  S.modelId = id;
  S.model = await api(`/models/${id}`);
  await Promise.all([renderOverview(), renderPredictive(), renderPrescriptive(), renderValidation(), renderGovernance()]);
}

/* ---------- gráficos ---------- */

function makeChart(id, cfg) {
  if (S.charts[id]) S.charts[id].destroy();
  cfg.options = { responsive: true, maintainAspectRatio: false, ...(cfg.options || {}) };
  S.charts[id] = new Chart(document.getElementById(id), cfg);
}

/* ---------- Visão Geral ---------- */

async function renderOverview() {
  const base = await api(`/models/${S.modelId}/baseline`);
  const cs = [...EXO, "TP"];
  makeChart("chart-constructs", {
    type: "bar",
    data: {
      labels: cs.map(c => NAMES[c]),
      datasets: [{ data: cs.map(c => base.constructs[c].mean_1_7),
                   backgroundColor: cs.map(c => c === "TP" ? "#1f5eff" : "#9db4e8") }],
    },
    options: { indexAxis: "y", plugins: { legend: { display: false } },
               scales: { x: { min: 1, max: 7, title: { display: true, text: "média (escala 1–7)" } } } },
  });
  const kpis = [
    ["R² (TP)", fmt(S.model.r2, 3)],
    ["TP média", fmt(base.constructs.TP.mean_1_7)],
    ["N (onda 1)", S.model.n],
    ["Maior gargalo", NAMES[EXO.reduce((a, b) =>
      base.constructs[a].mean_1_7 < base.constructs[b].mean_1_7 ? a : b)]],
  ];
  $("#kpis").innerHTML = kpis.map(([l, v]) =>
    `<div class="kpi"><div class="val">${v}</div><div class="lbl">${l}</div></div>`).join("");

  await renderReadingGuide(base);
}

/* faixa interpretativa de um índice 1–7 (semáforo) */
function band(m) {
  if (m < 3.5) return { label: "Crítico", cls: "b-crit" };
  if (m < 4.5) return { label: "Atenção", cls: "b-warn" };
  if (m < 5.5) return { label: "Bom", cls: "b-ok" };
  return { label: "Forte", cls: "b-strong" };
}

async function renderReadingGuide(base) {
  const sens = await api(`/models/${S.modelId}/sensitivity`);
  const delta = {};                    // ΔTP (pontos 1–7) por +1 DP no construto
  sens.tornado.forEach(t => { delta[t.construct] = t.delta_up; });

  const legend = [
    ["1 – 3,5", "Crítico", "b-crit"], ["3,5 – 4,5", "Atenção", "b-warn"],
    ["4,5 – 5,5", "Bom", "b-ok"], ["5,5 – 7", "Forte", "b-strong"],
  ];
  $("#scale-legend").innerHTML = legend.map(([faixa, lbl, cls]) =>
    `<span class="scale-chip ${cls}"><strong>${lbl}</strong> ${faixa}</span>`).join("");

  // ordena exógenos por índice (do mais fraco ao mais forte) e põe TP no fim
  const exoSorted = [...EXO].sort((a, b) => base.constructs[a].mean_1_7 - base.constructs[b].mean_1_7);
  const rows = exoSorted.map(c => {
    const m = base.constructs[c].mean_1_7, b = band(m), d = delta[c] ?? 0;
    let valor, leitura;
    if (Math.abs(d) < 0.05) {
      valor = "≈ 0 (efeito quase nulo)";
      leitura = "Praticamente não move a proteção tecnológica no modelo.";
    } else if (d > 0) {
      valor = `+1 DP ⇒ +${fmt(d)} pts em TP`;
      leitura = m < 4.5
        ? "Fraco e influente: <strong>prioridade</strong> — cada avanço aqui rende muito em TP."
        : m < 5.5
          ? "Influente e com folga: ainda há ganho a extrair elevando este tema."
          : "Força que já sustenta a proteção — o foco é manter o patamar.";
    } else {
      valor = `+1 DP ⇒ ${fmt(d)} pts em TP`;
      leitura = "Efeito negativo/não significativo no modelo — não é alavanca de melhoria " +
                "(provável supressão estatística; ver Governança).";
    }
    const beta = S.model.beta[c];
    const betaCls = beta > 0.05 ? "beta-pos" : beta < -0.05 ? "beta-neg" : "beta-zero";
    return `<tr>
      <td>${NAMES[c]}</td>
      <td><strong>${fmt(m)}</strong></td>
      <td><span class="tag ${b.cls}">${b.label}</span></td>
      <td class="${betaCls}"><strong>${beta >= 0 ? "+" : ""}${fmt(beta, 3)}</strong></td>
      <td>${valor}</td>
      <td>${leitura}</td></tr>`;
  });

  const tp = base.constructs.TP.mean_1_7, tpb = band(tp);
  rows.push(`<tr class="row-tp">
    <td><strong>Proteção Tecnológica</strong><br><span class="note">(resultado)</span></td>
    <td><strong>${fmt(tp)}</strong></td>
    <td><span class="tag ${tpb.cls}">${tpb.label}</span></td>
    <td>—</td>
    <td>—</td>
    <td>É o <strong>alvo</strong> que o modelo explica; todo o resto existe para elevá-lo.</td></tr>`);
  $("#tbl-reading tbody").innerHTML = rows.join("");

  const gargalo = NAMES[exoSorted[0]];
  $("#value-chain").innerHTML =
    `<strong>Por que aumentar ou diminuir importa.</strong> A Proteção Tecnológica (hoje ${fmt(tp)}/7) é o ` +
    `resultado final — o quanto a organização percebe que protege sua tecnologia. Os sete temas são as ` +
    `<em>alavancas</em>: elevá-los desloca a TP para cima (a coluna “quanto vale melhorar” mostra em quantos ` +
    `pontos). O maior valor está em melhorar um tema <strong>ao mesmo tempo fraco e influente</strong> — no ` +
    `momento, <strong>${gargalo}</strong> é o maior gargalo. Ganhar 1 ponto na escala não é cosmético: é a ` +
    `organização inteira movendo sua percepção de proteção rumo ao topo, o que na prática significa menos ` +
    `exposição percebida a incidentes e vazamentos. É esse ganho que o modo Preditivo simula e o Prescritivo ` +
    `prioriza.`;
}

/* ---------- Preditivo ---------- */

function renderPredictive() {
  const wrap = $("#sliders");
  wrap.innerHTML = EXO.map(c => `
    <div class="slider-row">
      <span class="name">${NAMES[c]}</span>
      <input type="range" id="sl-${c}" min="-1.5" max="1.5" step="0.1" value="0"
             oninput="document.getElementById('out-${c}').value = (+this.value).toFixed(1) + ' DP'">
      <output id="out-${c}">0.0 DP</output>
    </div>`).join("");
}

function setSlider(c, v) {
  $(`#sl-${c}`).value = v;
  $(`#out-${c}`).value = `${(+v).toFixed(1)} DP`;
}

const SIM_PLACEHOLDER = '<p class="note">Ajuste os sliders e simule.</p>';

$("#btn-reset").addEventListener("click", () => {
  EXO.forEach(c => setSlider(c, 0));
  $("#sim-result").innerHTML = SIM_PLACEHOLDER;   // limpa resultado órfão da simulação anterior
});
$("#btn-wargame").addEventListener("click", () => {
  EXO.forEach(c => setSlider(c, 0));
  setSlider("SK", -0.6); setSlider("SF", -0.4);
});

$("#btn-simulate").addEventListener("click", async () => {
  const interventions = EXO
    .map(c => ({ construct: c, kind: "shift", value: +$(`#sl-${c}`).value }))
    .filter(iv => iv.value !== 0);
  const btn = $("#btn-simulate");
  btn.disabled = true;
  try {
    const out = await api(`/models/${S.modelId}/simulate`,
      { method: "POST", json: { n: 200, k: 2000, seed: 42, interventions } });
    const r = out.results;
    // % da escala (0% = mínimo TP=1, 100% = máximo TP=7) — bounded, é a métrica de desempenho do IPMA
    const clip = x => Math.min(7, Math.max(1, x));
    const perf = tp => (tp - 1) / 6 * 100;
    const base = r.tp_index_baseline;
    const projRaw = r.tp_index_projected;
    const proj = clip(projRaw);
    const ci = r.delta_tp_points.ci95;
    const basePerf = perf(base), projPerf = perf(proj);
    const dPerf = projPerf - basePerf;
    const projLo = perf(clip(base + ci[0])), projHi = perf(clip(base + ci[1]));
    const overshoot = projRaw > 7 + 1e-9, undershoot = projRaw < 1 - 1e-9;
    // variação abaixo de 0,5 pp é ruído do Monte Carlo -> tratar como neutra (nem verde nem vermelho, sem "-0")
    const neutral = Math.abs(dPerf) < 0.5;
    const cls = neutral ? "neutral" : (dPerf > 0 ? "pos" : "neg");
    const varTxt = neutral
      ? "sem variação relevante (0 pontos percentuais)"
      : `variação ${dPerf > 0 ? "+" : "−"}${fmt(Math.abs(dPerf), 0)} pontos percentuais`;
    $("#sim-result").innerHTML = `
      <div class="delta ${cls}">${fmt(projPerf, 0)}%</div>
      <p>nível de proteção projetado — <strong>0% = mínimo</strong> (TP 1), <strong>100% = máximo</strong> (TP 7) ·
         IC 95% [${fmt(projLo, 0)}%; ${fmt(projHi, 0)}%]</p>
      <p class="delta-points">Sai de <strong>${fmt(basePerf, 0)}%</strong> (atual) para
         <strong>${fmt(projPerf, 0)}%</strong> · ${varTxt}</p>
      <p class="delta-points">No índice TP (escala 1–7): ${fmt(base)} → <strong>${fmt(proj)}</strong></p>
      ${overshoot ? `<p class="ceiling-note">⚠ A combinação de intervenções ultrapassaria o teto da escala
         (índice projetado ${fmt(projRaw)} &gt; 7). O modelo estrutural é linear e extrapola além do máximo;
         o valor foi limitado a 100% (TP 7). Cenário extremo — interpretar com cautela.</p>` : ""}
      ${undershoot ? `<p class="ceiling-note">⚠ A combinação ultrapassaria o piso da escala
         (índice ${fmt(projRaw)} &lt; 1); limitado a 0% (TP 1).</p>` : ""}
      <p>P(Δ&gt;0) = ${fmt(r.p_delta_positive * 100, 1)}%</p>
      <p class="note">Monte Carlo: K=${r.k} réplicas de N=${r.n} · semente ${r.seed}</p>`;
  } catch (e) {
    $("#sim-result").innerHTML = `<p class="error">${e.message}</p>`;
  } finally { btn.disabled = false; }
});

/* ---------- Prescritivo ---------- */

async function renderPrescriptive() {
  const sens = await api(`/models/${S.modelId}/sensitivity`);
  makeChart("chart-tornado", {
    type: "bar",
    data: {
      labels: sens.tornado.map(t => t.name),
      datasets: [
        { label: "+1 DP", data: sens.tornado.map(t => t.delta_up), backgroundColor: "#14804a" },
        { label: "−1 DP", data: sens.tornado.map(t => t.delta_down), backgroundColor: "#b3261e" },
      ],
    },
    options: { indexAxis: "y",
               scales: { x: { title: { display: true, text: "ΔTP (pontos no índice 1–7)" } } } },
  });
  makeChart("chart-ipma", {
    type: "scatter",
    data: { datasets: sens.ipma.map(p => ({
      label: p.name, data: [{ x: p.importance, y: p.performance }], pointRadius: 6 })) },
    options: { scales: {
      x: { title: { display: true, text: "Importância (efeito total β)" } },
      y: { title: { display: true, text: "Desempenho (0–100)" }, min: 0, max: 100 } } },
  });

  const palette = { SV: "#5b9bd5", SG: "#e15d80", SU: "#ed9c46", SM: "#e8c34b",
                    SF: "#57bfb1", SY: "#9a6fd8", SK: "#9aa2ad" };
  const byConstruct = {};
  for (const it of sens.ipma_items) (byConstruct[it.construct] ??= []).push(it);
  makeChart("chart-ipma-items", {
    type: "scatter",
    data: { datasets: Object.entries(byConstruct).map(([c, items]) => ({
      label: items[0].construct_name,
      data: items.map(it => ({ x: it.importance, y: it.performance, item: it.item, label: it.label })),
      backgroundColor: palette[c], pointRadius: 5,
    })) },
    options: {
      scales: {
        x: { title: { display: true, text: "Importância (efeito total do item em TP)" } },
        y: { title: { display: true, text: "Desempenho (0–100)" }, min: 0, max: 100 },
      },
      plugins: { tooltip: { callbacks: {
        label: (ctx) => `[${ctx.raw.item}] ${ctx.raw.label}`.slice(0, 90),
        afterLabel: (ctx) => `importância ${ctx.raw.x.toFixed(3)} · desempenho ${ctx.raw.y.toFixed(0)}/100`,
      } } },
    },
  });
  await refreshRecs();
}

async function refreshRecs() {
  const recs = await api(`/models/${S.modelId}/recommendations`);
  $("#recs").innerHTML = recs.length ? recs.map(r => `
    <div class="rec ${r.status}">
      <strong>${NAMES[r.construct]}</strong> — shift de ${r.intervention.value} DP
      <span class="badge">${r.status === "emitted" ? "aguardando decisão" : r.status}</span>
      <p class="note">${r.expected.rationale}</p>
      <p>Efeito esperado: <strong>+${fmt(r.expected.expected_delta_tp_points)}</strong> pontos em TP ·
         IC 95% ${fmtCI(r.expected.ci95_points)} · P(Δ&gt;0)=${fmt(r.expected.p_positive * 100, 0)}%</p>
      ${(r.expected.levers || []).length ? `
        <div class="levers">
          <span class="levers-title">Onde atuar (itens-alavanca pelo IPMA de indicador):</span>
          ${r.expected.levers.map(l => `
            <div class="lever">
              <code>${l.item}</code> ${l.label} —
              desempenho <strong>${fmt(l.performance, 0)}/100</strong>;
              +1 ponto neste item ≈ <strong>+${fmt(l.delta_tp_per_point)}</strong> em TP
            </div>`).join("")}
        </div>` : ""}
      ${r.status === "emitted" && S.role !== "viewer" ? `
        <div class="row">
          <button onclick="decide('${r.id}','accepted')">Aceitar</button>
          <button class="ghost" onclick="decide('${r.id}','rejected')">Rejeitar</button>
        </div>` : r.decided_by ? `<p class="note">Decidido por ${r.decided_by}</p>` : ""}
    </div>`).join("")
    : `<p class="note">Nenhuma recomendação emitida ainda.</p>`;
}

window.decide = async (id, status) => {
  await api(`/recommendations/${id}`, { method: "PATCH", json: { status } });
  await refreshRecs();
};

$("#btn-report").addEventListener("click", async () => {
  const btn = $("#btn-report");
  btn.disabled = true;
  try {
    const res = await fetch(`/models/${S.modelId}/recommendations/report.pdf`,
                            { headers: { Authorization: `Bearer ${S.token}` } });
    if (!res.ok) throw new Error(`Erro ${res.status} ao gerar o relatório`);
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (res.headers.get("Content-Disposition") || "").match(/filename="(.+)"/)?.[1]
                 || "recomendacoes_aceitas.pdf";
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e) { alert(e.message); }
  finally { btn.disabled = false; }
});

$("#btn-clear-accepted").addEventListener("click", async () => {
  if (!confirm("Zerar as recomendações aceitas? Elas serão arquivadas (saem do relatório e da lista), " +
               "a trilha de auditoria é preservada, e um novo conjunto de sugestões será emitido " +
               "para um novo ciclo de decisão.")) return;
  const btn = $("#btn-clear-accepted");
  btn.disabled = true;
  try {
    const out = await api(`/models/${S.modelId}/recommendations/clear-accepted`, { method: "POST" });
    await api(`/models/${S.modelId}/recommendations?top=3`, { method: "POST" });  // novo ciclo
    alert(`${out.archived} recomendação(ões) arquivada(s). Novas sugestões emitidas para decisão.`);
    await refreshRecs();
  } catch (e) { alert(e.message); }
  finally { btn.disabled = false; }
});

$("#btn-emit-recs").addEventListener("click", async () => {
  const btn = $("#btn-emit-recs");
  btn.disabled = true;
  try {
    const novas = await api(`/models/${S.modelId}/recommendations?top=3`, { method: "POST" });
    if (!novas.length) {
      alert("Nenhuma recomendação nova a emitir: todos os construtos elegíveis já estão " +
            "pendentes, aceitos ou rejeitados. Decida os pendentes ou use 'Zerar aceitas'.");
    }
    await refreshRecs();
  } catch (e) { alert(e.message); }
  finally { btn.disabled = false; }
});

/* ---------- Validações ---------- */

function statusCell(ok, warnLabel = "revisar") {
  return ok ? `<td class="status-ok">OK</td>` : `<td class="status-bad">${warnLabel}</td>`;
}

async function renderValidation() {
  const b = await api(`/models/${S.modelId}/battery`);
  const rows = Object.entries(b.reliability).map(([c, m]) => `
    <tr><td>${NAMES[c]}</td><td>${fmt(m.alpha, 3)}</td><td>${fmt(m.rho_a, 3)}</td>
        <td>${fmt(m.rho_c, 3)}</td><td>${fmt(m.ave, 3)}</td>
        ${statusCell(m.alpha >= 0.7 && m.rho_c >= 0.7 && m.ave >= 0.5)}</tr>`);
  $("#tbl-reliability tbody").innerHTML = rows.join("");
  const ok = b.htmt_max < 0.85;
  $("#htmt-summary").innerHTML =
    `HTMT máximo: <strong>${fmt(b.htmt_max, 3)}</strong> — validade discriminante ` +
    (ok ? `<span class="status-ok">OK (&lt; 0.85)</span>`
        : `<span class="status-bad">violada — revisar modelo de medida (G7)</span>`);
}

$("#btn-equivalence").addEventListener("click", async () => {
  const btn = $("#btn-equivalence");
  btn.disabled = true;
  $("#equiv-result").innerHTML = `<p class="note">Gerando sintético e rodando MGA/MICOM (~1 min)…</p>`;
  try {
    const gen = await api(`/models/${S.modelId}/generate`, { method: "POST",
      json: { n: 142, seed: 42, mode: "empirical",
              noise: { p_careless: 0, p_straight: 0, sigma_acq: 0 } } });
    const rep = await api(`/models/${S.modelId}/equivalence`, { method: "POST",
      json: { dataset_id: gen.dataset_id, n_perm: 200, seed: 42 } });
    const cong = Object.entries(rep.loading_congruence);
    $("#equiv-result").innerHTML = `
      <p>Veredito: ${rep.equivalent
        ? '<span class="status-ok">EQUIVALENTE</span>'
        : '<span class="status-bad">NÃO EQUIVALENTE</span>'}</p>
      <table><thead><tr><th>Critério</th><th>Resultado</th><th>Status</th></tr></thead><tbody>
        <tr><td>Δ máx. de médias (itens)</td><td>${fmt(rep.descriptives.max_abs_diff_mean, 3)}</td>
            ${statusCell(rep.descriptives.max_abs_diff_mean < 0.5)}</tr>
        <tr><td>Congruência de cargas (mín.)</td>
            <td>${fmt(Math.min(...cong.map(([, v]) => v)), 3)}</td>
            ${statusCell(Math.min(...cong.map(([, v]) => v)) > 0.95)}</tr>
        <tr><td>MGA por permutação (caminhos iguais)</td>
            <td>${rep.mga.all_equivalent_at_5pct ? "todos p > 0.05" : "há caminho divergente"}</td>
            ${statusCell(rep.mga.all_equivalent_at_5pct)}</tr>
        <tr><td>MICOM passo 2 (invariância composicional)</td>
            <td>${rep.micom_step2.all_invariant ? "todos os construtos invariantes" : "violação"}</td>
            ${statusCell(rep.micom_step2.all_invariant)}</tr>
      </tbody></table>
      <p class="note">Modo empírico, N=142, semente 42, sem ruído.</p>`;
  } catch (e) {
    $("#equiv-result").innerHTML = `<p class="error">${e.message}</p>`;
  } finally { btn.disabled = false; }
});

/* ---------- Governança ---------- */

async function renderGovernance() {
  const m = S.model;
  $("#gov-model").innerHTML = `
    <dt>Modelo</dt><dd>${m.name} (${m.id})</dd>
    <dt>Hash de proveniência</dt><dd>${m.provenance_hash}</dd>
    <dt>Fonte</dt><dd>${m.provenance.source} · N=${m.provenance.n}</dd>
    <dt>Calibrado em</dt><dd>${m.provenance.calibrated_at}</dd>
    <dt>Estimador</dt><dd>${m.provenance.estimator}</dd>
    <dt>Maturidade (CA9)</dt><dd>${m.maturity_level}</dd>
    <dt>Criado por</dt><dd>${m.created_by}</dd>`;
  $("#tbl-paths tbody").innerHTML =
    EXO.map(c => `<tr><td>${NAMES[c]}</td><td>${fmt(m.beta[c], 3)}</td></tr>`).join("");
}

/* ---------- Calibração (RF1, dois caminhos) ---------- */

function importMsg(text, isError = false) {
  const el = $("#import-msg");
  el.textContent = text;
  el.className = isError ? "error" : "note";
}

$("#btn-import-raw").addEventListener("click", async () => {
  const file = $("#raw-file").files[0];
  if (!file) return importMsg("Selecione o CSV de respostas.", true);
  const fd = new FormData();
  fd.append("file", file);
  const btn = $("#btn-import-raw");
  btn.disabled = true;
  try {
    const m = await api(`/models?name=${encodeURIComponent($("#raw-name").value)}`,
                        { method: "POST", body: fd });
    importMsg(`Modelo calibrado do bruto: ${m.name} (N=${m.n}, R²=${fmt(m.r2, 3)}).`);
    await loadModels();
  } catch (e) { importMsg(e.message, true); }
  finally { btn.disabled = false; }
});

$("#btn-import-pls").addEventListener("click", async () => {
  const cl = $("#pls-cl").files[0], fl = $("#pls-fl").files[0], desc = $("#pls-desc").files[0];
  if (!cl || !fl || !desc) return importMsg("Os três exports obrigatórios precisam ser selecionados.", true);
  let betaJson;
  try { betaJson = JSON.stringify(JSON.parse($("#pls-beta").value)); }
  catch { return importMsg("β inválido: precisa ser JSON (ex.: {\"SV\":0.02, ...}).", true); }
  const fd = new FormData();
  fd.append("cross_loadings", cl);
  fd.append("fornell_larcker", fl);
  fd.append("descriptives", desc);
  if ($("#pls-corr").files[0]) fd.append("item_correlations", $("#pls-corr").files[0]);
  fd.append("beta", betaJson);
  fd.append("r2", $("#pls-r2").value);
  fd.append("name", $("#pls-name").value);
  const btn = $("#btn-import-pls");
  btn.disabled = true;
  try {
    const m = await api("/models/smartpls", { method: "POST", body: fd });
    importMsg(`Modelo importado do SmartPLS: ${m.name} (R²=${fmt(m.r2, 3)}). ` +
              "What-if usa os parâmetros aferidos pelo SmartPLS.");
    await loadModels();
  } catch (e) { importMsg(e.message, true); }
  finally { btn.disabled = false; }
});

$("#btn-bootstrap").addEventListener("click", async () => {
  const btn = $("#btn-bootstrap");
  btn.disabled = true; btn.textContent = "Rodando bootstrap…";
  try {
    const boot = await api(`/models/${S.modelId}/bootstrap?n_boot=500`);
    $("#tbl-paths tbody").innerHTML = EXO.map(c => {
      const p = boot.paths[c];
      const sig = p.p < 0.05;
      return `<tr><td>${NAMES[c]}</td><td>${fmt(p.estimate, 3)}</td>
        <td>${fmtCI(p.ci95, 3)}</td><td>${fmt(p.p, 3)}</td>
        <td class="${sig ? "status-ok" : "status-warn"}">${sig ? "sim" : "não"}</td></tr>`;
    }).join("");
  } catch (e) { alert(e.message); }
  finally { btn.disabled = false; btn.textContent = "Rodar bootstrap (500 reamostras)"; }
});
