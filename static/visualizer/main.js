import * as THREE from
    "https://unpkg.com/three@0.160.0/build/three.module.js";

import { OrbitControls } from
    "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js?module";

const API_BASE = "http://localhost:5000";
let backendOnline = false;
let lastActivations = null;  


const scene = new THREE.Scene();
scene.background = new THREE.Color(0x020408);
scene.fog = new THREE.FogExp2(0x020408, 0.018);

const camera = new THREE.PerspectiveCamera(
    65, innerWidth / innerHeight, 0.1, 2000
);
camera.position.set(0, 10, 45);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 5;
controls.maxDistance = 120;

scene.add(new THREE.AmbientLight(0x0a1020, 2));

const keyLight = new THREE.PointLight(0x4488ff, 6, 80);
keyLight.position.set(15, 25, 20);
scene.add(keyLight);

const rimLight = new THREE.PointLight(0x00ffcc, 3, 60);
rimLight.position.set(-15, 10, -10);
scene.add(rimLight);


const gridHelper = new THREE.GridHelper(200, 60, 0x0a1a2a, 0x0a1a2a);
scene.add(gridHelper);


const MODEL_STRUCTURE = [
    "Tokenizer",
    "Embedding",
    "Positional Encoding",
    "Transformer 0",
    "Transformer 1",
    "Transformer 2",
    "Transformer 3",
    "Transformer 4",
    "Transformer 5",
    "Sentence Meaning",
    "Output"
];

const STAGE_COLORS = {
    Tokenizer:    0x00ffcc,
    Embedding:    0x00ff88,
    Positional:   0xaa55ff,
    Transformer:  0x3366ff,
    Sentence:     0xffff44,
    Output:       0xff4444
};

const STAGE_DESCRIPTIONS = {
    Tokenizer:               "Splits text into sub-word tokens using BPE vocabulary",
    Embedding:               "Maps each token ID → 768-dim dense vector",
    "Positional Encoding":   "Adds sinusoidal position signals to each embedding",
    "Transformer 0":         "Layer 0: Multi-head self-attention + FFN (syntactic patterns)",
    "Transformer 1":         "Layer 1: Refines token relationships, builds phrase structure",
    "Transformer 2":         "Layer 2: Semantic composition begins",
    "Transformer 3":         "Layer 3: Long-range dependencies resolved",
    "Transformer 4":         "Layer 4: Contextual meaning crystallises",
    "Transformer 5":         "Layer 5: Final deep representation",
    "Sentence Meaning":      "Pooled CLS token → sentence-level representation",
    Output:                  "Softmax over vocabulary → next-token probability distribution"
};

const TRANSFORMER_INTERNALS = [
    { id: "ln1",  label: "LayerNorm 1",    color: 0xaaaaff, x: -4, y:  3 },
    { id: "qkv",  label: "Q / K / V proj", color: 0x5599ff, x:  0, y:  5 },
    { id: "attn", label: "Multi-Head Attn",color: 0x0088ff, x:  0, y:  3 },
    { id: "add1", label: "Residual Add",   color: 0x44ffaa, x:  4, y:  3 },
    { id: "ln2",  label: "LayerNorm 2",    color: 0xaaaaff, x: -4, y:  0 },
    { id: "ffn1", label: "FFN Linear 1",   color: 0xff8833, x:  0, y:  0 },
    { id: "gelu", label: "GELU Activation",color: 0xffcc00, x:  4, y:  0 },
    { id: "ffn2", label: "FFN Linear 2",   color: 0xff5533, x:  0, y: -2 },
    { id: "add2", label: "Residual Add",   color: 0x44ffaa, x:  4, y: -2 },
];

const INTERNAL_FLOW = ["ln1","qkv","attn","add1","ln2","ffn1","gelu","ffn2","add2"];

const blocks = [];
const connections = [];
let internalGroup = null;
let expandedTransformer = null;

let currentTokens  = [];
let outputTokens   = [];
let processingInput = "";
let stepLog        = [];

function createLabel(text, fontSize = 40, color = "white") {
    const canvas = document.createElement("canvas");
    canvas.width  = 512;
    canvas.height = 100;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = color;
    ctx.font = `${fontSize}px 'Courier New', monospace`;
    ctx.fillText(text, 16, 65);
    const tex = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(
        new THREE.SpriteMaterial({ map: tex, transparent: true })
    );
    sprite.scale.set(7, 1.4, 1);
    return sprite;
}

const floatingLabels = {};

function createFloatingLabel(blockMesh, text, yOffset = 4.5) {
    const name = blockMesh.userData.name;
    if (floatingLabels[name]) scene.remove(floatingLabels[name]);

    const canvas = document.createElement("canvas");
    canvas.width  = 768;
    canvas.height = 120;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, 768, 120);
    ctx.fillStyle = "rgba(0,0,0,0.7)";
    roundRect(ctx, 0, 0, 768, 120, 14);
    ctx.fill();
    ctx.strokeStyle = "#3366ff88";
    ctx.lineWidth = 2;
    roundRect(ctx, 0, 0, 768, 120, 14);
    ctx.stroke();
    ctx.fillStyle = "#aaddff";
    ctx.font = "bold 30px 'Courier New'";
    ctx.fillText(text.slice(0, 36), 16, 42);
    ctx.fillStyle = "#667799";
    ctx.font = "22px 'Courier New'";
    const desc = STAGE_DESCRIPTIONS[name] || "";
    ctx.fillText(desc.slice(0, 56), 16, 80);

    const tex = new THREE.CanvasTexture(canvas);
    const sprite = new THREE.Sprite(
        new THREE.SpriteMaterial({ map: tex, transparent: true })
    );
    sprite.scale.set(12, 1.9, 1);
    sprite.position.set(
        blockMesh.position.x,
        blockMesh.position.y + yOffset,
        blockMesh.position.z
    );
    scene.add(sprite);
    floatingLabels[name] = sprite;
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}


function createBlock(name, z) {
    const key   = name.split(" ")[0];
    const color = STAGE_COLORS[key] || 0xffffff;

    const outlineGeo = new THREE.BoxGeometry(7.6, 3.6, 1.6);
    const outlineMat = new THREE.MeshBasicMaterial({
        color, transparent: true, opacity: 0.15, side: THREE.BackSide
    });
    const outline = new THREE.Mesh(outlineGeo, outlineMat);
    outline.position.set(0, 0, z);
    scene.add(outline);

    const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(7, 3, 1.2),
        new THREE.MeshStandardMaterial({
            color: 0x060e1a,
            emissive: color,
            emissiveIntensity: 0.18,
            roughness: 0.3,
            metalness: 0.8
        })
    );
    mesh.position.set(0, 0, z);
    mesh.userData.name    = name;
    mesh.userData.color   = color;
    mesh.userData.outline = outline;
    scene.add(mesh);
    blocks.push(mesh);

    const label = createLabel(name);
    label.position.set(0, 2.8, z);
    scene.add(label);

    return mesh;
}


function connect(a, b, opacity = 0.35) {
    const points = [];
    const steps  = 20;
    for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        points.push(new THREE.Vector3(
            a.position.x * (1 - t) + b.position.x * t,
            Math.sin(t * Math.PI) * 0.6,
            a.position.z * (1 - t) + b.position.z * t
        ));
    }
    const geo  = new THREE.BufferGeometry().setFromPoints(points);
    const line = new THREE.Line(
        geo,
        new THREE.LineBasicMaterial({ color: 0x1a3366, transparent: true, opacity })
    );
    scene.add(line);
    connections.push(line);
}

function buildOverview() {
    let prev = null, z = 0;
    MODEL_STRUCTURE.forEach(name => {
        const block = createBlock(name, z);
        if (prev) connect(prev, block, 0.5);
        blocks.forEach(old => {
            if (old !== block && Math.random() < 0.2)
                connect(old, block, 0.04);
        });
        prev = block;
        z += 5;
    });
}

buildOverview();

function highlight(name) {
    blocks.forEach(b => {
        const active = b.userData.name === name || b.userData.name.includes(name);
        b.material.emissiveIntensity = active ? 2.5 : 0.08;
        b.material.opacity  = expandedTransformer ? (active ? 1 : 0.08) : 1;
        b.material.transparent = !!expandedTransformer;
        if (b.userData.outline)
            b.userData.outline.material.opacity = active ? 0.5 : 0.04;
        b.scale.lerp(
            active ? new THREE.Vector3(1.3, 1.3, 1.3) : new THREE.Vector3(1, 1, 1),
            0.15
        );
    });
}

function buildTransformerInternals(blockMesh) {
    if (internalGroup) { scene.remove(internalGroup); internalGroup = null; }

    expandedTransformer = blockMesh.userData.name;
    internalGroup = new THREE.Group();
    internalGroup.position.copy(blockMesh.position);
    internalGroup.position.x += 12;
    internalGroup.position.y += 1;

    const layerIdx  = parseInt(blockMesh.userData.name.replace("Transformer ", ""), 10);
    const layerData = lastActivations?.layers?.[layerIdx] || null;

    const nodes = {};
    TRANSFORMER_INTERNALS.forEach(info => {
        let emissiveInt = 0.4;
        if (layerData) {
            if (info.id === "attn" || info.id === "qkv") {
                emissiveInt = Math.min(3.0, 0.3 + Math.abs(layerData.attn_mean) * 0.6);
            } else if (info.id === "ffn1" || info.id === "ffn2" || info.id === "gelu") {
                emissiveInt = Math.min(3.0, 0.3 + Math.abs(layerData.mlp_mean) * 0.6);
            }
        }

        const geo  = new THREE.BoxGeometry(3.8, 1.4, 0.7);
        const mat  = new THREE.MeshStandardMaterial({
            color: 0x030810,
            emissive: info.color,
            emissiveIntensity: emissiveInt,
            roughness: 0.2,
            metalness: 0.9
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(info.x, info.y, 0);
        mesh.userData.id = info.id;
        internalGroup.add(mesh);
        nodes[info.id] = mesh;

        const lbl = createLabel(info.label, 28, "#aaddff");
        lbl.scale.set(5, 0.9, 1);
        lbl.position.set(info.x, info.y + 1.2, 0.5);
        internalGroup.add(lbl);
    });

    for (let i = 0; i < INTERNAL_FLOW.length - 1; i++) {
        const a = nodes[INTERNAL_FLOW[i]];
        const b = nodes[INTERNAL_FLOW[i + 1]];
        const geo  = new THREE.BufferGeometry().setFromPoints([a.position, b.position]);
        const line = new THREE.Line(geo, new THREE.LineBasicMaterial({
            color: 0x4466aa, transparent: true, opacity: 0.6
        }));
        internalGroup.add(line);
    }

    scene.add(internalGroup);

    blocks.forEach(b => {
        b.material.transparent = true;
        b.material.opacity = b.userData.name === expandedTransformer ? 1 : 0.1;
        if (b.userData.outline)
            b.userData.outline.material.opacity =
                b.userData.name === expandedTransformer ? 0.4 : 0.02;
    });

    addStepLog(`🔍 Expanded: ${blockMesh.userData.name} — click elsewhere to close`);
}

function closeInternals() {
    if (internalGroup) { scene.remove(internalGroup); internalGroup = null; }
    expandedTransformer = null;
    blocks.forEach(b => {
        b.material.transparent = false;
        b.material.opacity = 1;
        if (b.userData.outline) b.userData.outline.material.opacity = 0.15;
    });
}

let internalFlowTimer = 0;
let internalFlowIdx   = 0;

function animateInternals(dt) {
    if (!internalGroup) return;
    internalFlowTimer += dt;
    if (internalFlowTimer > 0.35) {
        internalFlowTimer = 0;
        internalGroup.children.forEach(child => {
            if (child.material?.emissive)
                child.material.emissiveIntensity = 0.25;
        });
        const id   = INTERNAL_FLOW[internalFlowIdx % INTERNAL_FLOW.length];
        const node = internalGroup.children.find(c => c.userData.id === id);
        if (node) node.material.emissiveIntensity = 3;
        internalFlowIdx++;
    }
}

const particles = [];

function spawnFlow(fromIdx, toIdx, color, label = "") {
    const fromZ = fromIdx * 5;
    const toZ   = toIdx   * 5;

    const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.22, 12, 12),
        new THREE.MeshBasicMaterial({ color })
    );
    const halo = new THREE.Mesh(
        new THREE.SphereGeometry(0.44, 12, 12),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.18 })
    );
    mesh.add(halo);
    scene.add(mesh);

    let labelSprite = null;
    if (label) {
        labelSprite = createLabel(label, 26, "#ffffaa");
        labelSprite.scale.set(4, 0.8, 1);
        scene.add(labelSprite);
    }

    particles.push({ mesh, labelSprite, t: 0, fromZ, toZ });
}

function addStepLog(msg) {
    stepLog.push(msg);
    if (stepLog.length > 14) stepLog.shift();
    const el = document.getElementById("step-log");
    if (el) {
        el.innerHTML = stepLog.map((s, i) =>
            `<div style="opacity:${0.35 + 0.65 * (i / stepLog.length)}">${s}</div>`
        ).join("");
        el.scrollTop = el.scrollHeight;
    }
}

function updateOutputArea() {
    const el = document.getElementById("output-area");
    if (!el || !processingInput) return;

    el.innerHTML = `
        <div class="out-section">
            <div class="out-label">Input</div>
            <div class="out-text">${processingInput}</div>
        </div>
        <div class="out-section">
            <div class="out-label">Tokens (${currentTokens.length})</div>
            <div class="out-tokens">
                ${currentTokens.map(t => {
                    const display = t
                        .replace(/^Ġ/, " ")
                        .replace(/Ċ/g, "↵");
                    return `<span class="token" title="${t}">${display}</span>`;
                }).join("")}
            </div>
        </div>
        <div class="out-section">
            <div class="out-label">Generated</div>
            <div class="out-text out-generated">${outputTokens.join(" ")}<span class="cursor">▌</span></div>
        </div>
    `;
}


function showActivationBars(summary) {
    const el = document.getElementById("token-prob");
    if (!el) return;

    const maxMag = Math.max(...summary.layers.map(l =>
        Math.abs(l.attn_mean) + Math.abs(l.mlp_mean)
    ), 0.001);

    el.innerHTML = `<div class="out-label">Layer Activation Magnitudes</div>` +
        summary.layers.map(l => {
            const mag  = Math.abs(l.attn_mean) + Math.abs(l.mlp_mean);
            const pct  = ((mag / maxMag) * 100).toFixed(1);
            const col  = mag > 2  ? "#ff4444"
                       : mag > 1  ? "#ffcc00"
                       :            "#00ffaa";
            return `
            <div class="token-row" title="attn=${l.attn_mean.toFixed(3)} mlp=${l.mlp_mean.toFixed(3)}">
                <span class="token-word">L${l.layer}</span>
                <div class="token-bar-wrap">
                    <div class="token-bar" style="width:${pct}%;background:${col}"></div>
                </div>
                <span class="token-pct">${mag.toFixed(2)}</span>
            </div>`;
        }).join("");
}

function applyActivationsToBlocks(summary) {
    if (!summary?.layers) return;

    summary.layers.forEach((layer, i) => {
        const name  = `Transformer ${i}`;
        const block = blocks.find(b => b.userData.name === name);
        if (!block) return;

        const mag  = Math.abs(layer.attn_mean) + Math.abs(layer.mlp_mean);
        const glow = Math.min(2.5, 0.1 + mag * 0.45);

        block.material.emissiveIntensity = glow;

        if (mag > 2.0)      block.material.emissive.setHex(0xff2222);
        else if (mag > 1.2) block.material.emissive.setHex(0xff8833);
        else                block.material.emissive.setHex(block.userData.color);

        if (block.userData.outline)
            block.userData.outline.material.opacity = Math.min(0.6, 0.1 + mag * 0.15);
    });
}

async function pingBackend() {
    try {
        const res = await fetch(`${API_BASE}/timeline`, { method: "GET" });
        backendOnline = res.ok;
    } catch {
        backendOnline = false;
    }
    updateConnectionBadge();
}

function updateConnectionBadge() {
    let badge = document.getElementById("backend-badge");
    if (!badge) {
        badge = document.createElement("div");
        badge.id = "backend-badge";
        badge.style.cssText = `
            position:fixed; bottom:14px; right:16px;
            font-family:'Share Tech Mono',monospace; font-size:10px;
            padding:4px 10px; border-radius:4px; z-index:99;
            pointer-events:none; letter-spacing:1px;
        `;
        document.body.appendChild(badge);
    }
    if (backendOnline) {
        badge.textContent  = "● BACKEND LIVE";
        badge.style.color  = "#00ffaa";
        badge.style.background = "rgba(0,30,15,0.85)";
        badge.style.border = "1px solid #00ffaa44";
    } else {
        badge.textContent  = "○ BACKEND OFFLINE — simulation mode";
        badge.style.color  = "#ff5533";
        badge.style.background = "rgba(30,5,0,0.85)";
        badge.style.border = "1px solid #ff553344";
    }
}

pingBackend();

async function runRealPipeline(msg) {
    // --- STEP 1: tokenize immediately ---
    addStepLog("Tokenising…");
    highlight("Tokenizer");
    try {
        const r = await fetch(`${API_BASE}/tokenize`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: msg })
        });
        const tok = await r.json();
        currentTokens = tok.tokens || [];
        addStepLog(`✂️  Tokenizer → ${tok.count} tokens`);
        updateOutputArea();
        spawnFlow(0, 1, STAGE_COLORS.Tokenizer, currentTokens[0] || "");
    } catch (e) {
        addStepLog(" Tokenizer request failed");
    }

    await delay(500);
    highlight("Embedding");
    createFloatingLabel(blocks[1], `${currentTokens.length} tokens × 768 dims`);
    addStepLog(` Embedding: ${currentTokens.length} tokens → ℝ⁷⁶⁸`);
    spawnFlow(1, 2, STAGE_COLORS.Embedding);

    await delay(500);
    highlight("Positional Encoding");
    createFloatingLabel(blocks[2], "sin/cos position signals injected");
    addStepLog(" Positional Encoding: offsets added");
    spawnFlow(2, 3, STAGE_COLORS.Positional);

    const chatPromise = fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg })
    });

    for (let i = 0; i < 6; i++) {
        await delay(350);
        const stageName = `Transformer ${i}`;
        highlight(stageName);
        createFloatingLabel(blocks[3 + i], `12 heads × 64 dims`);
        addStepLog(` Transformer ${i}: self-attention running…`);
        spawnFlow(3 + i, 4 + i, STAGE_COLORS.Transformer, currentTokens[i % currentTokens.length] || "");
    }

    addStepLog(" Generating response…");
    let aiResponse = "";
    try {
        const chatRes  = await chatPromise;
        const chatData = await chatRes.json();
        aiResponse     = chatData.response || "";
        outputTokens   = aiResponse.split(/\s+/).filter(Boolean);
        addStepLog(`Generated: "${aiResponse.slice(0, 60)}${aiResponse.length > 60 ? "…" : ""}"`);
    } catch (e) {
        addStepLog("Chat request failed");
    }
    try {
        const actRes  = await fetch(`${API_BASE}/activations_summary`);
        const summary = await actRes.json();
        lastActivations = summary;
        applyActivationsToBlocks(summary);
        showActivationBars(summary);
        addStepLog("Real activations applied to 3-D blocks");
    } catch (e) {
        addStepLog(" Could not load activations");
    }

    try {
        const tlRes  = await fetch(`${API_BASE}/timeline`);
        const frames = await tlRes.json();
        frames.forEach(f => addStepLog(` stage: ${f.stage}`));
    } catch {}

    await delay(300);
    highlight("Sentence Meaning");
    createFloatingLabel(blocks[9], "CLS token pooled → 768-d vector");
    spawnFlow(9, 10, STAGE_COLORS.Sentence);

    await delay(300);
    highlight("Output");
    createFloatingLabel(blocks[10], `"${outputTokens[0] || "…"}" generated`);
    updateOutputArea();
}

function tokenizeSim(text) {
    return text.trim().split(/\s+/).flatMap(w =>
        w.length > 6
            ? [w.slice(0, Math.ceil(w.length / 2)), w.slice(Math.ceil(w.length / 2))]
            : [w]
    );
}

const VOCAB = ["the","a","and","is","are","was","were","will","can","I","you","it","that",
               "this","be","have","do","say","get","make","go","know","think","see","come","want",
               "use","find","give","tell","work","call","try","ask","need","feel","become","leave"];

function generateNextTokenCandidates() {
    let probs = Array.from({length:5}, () => Math.random());
    const sum = probs.reduce((a, b) => a + b, 0);
    probs = probs.map(p => p / sum).sort((a, b) => b - a);
    const used = new Set(), words = [];
    while (words.length < 5) {
        const w = VOCAB[Math.floor(Math.random() * VOCAB.length)];
        if (!used.has(w)) { used.add(w); words.push(w); }
    }
    return words.map((w, i) => ({ word: w, prob: probs[i] }));
}

let timeline_sim = [];
let playing      = false;
let stageIndex   = 0;
let timer        = 0;
const STEP_TIME  = 1.8;

function buildSimTimeline(tokens) {
    return [
        { stage: "tokenizer",          meta: { tokens } },
        { stage: "embedding",          meta: { dims: 768 } },
        { stage: "positional_encoding",meta: {} },
        ...Array.from({length:6}, (_, i) => ({ stage: "transformer", meta: { index: i } })),
        { stage: "sentence_meaning",   meta: {} },
        { stage: "output",             meta: {} },
    ];
}

window.playThinking = async function () {
    const input = document.getElementById("prompt");
    const msg   = input.value.trim();
    if (!msg) return;

    processingInput = msg;
    currentTokens   = [];
    outputTokens    = [];
    stepLog         = [];
    playing         = false;

    addStepLog(`Processing: "${msg}"`);
    updateOutputArea();

    await pingBackend();

    if (backendOnline) {
        addStepLog(" Backend connected — using real model");
        await runRealPipeline(msg);
    } else {
        addStepLog("Backend offline — simulation mode");
        currentTokens = tokenizeSim(msg);
        timeline_sim  = buildSimTimeline(currentTokens);
        playing       = true;
        stageIndex    = 0;
        timer         = 0;
        updateOutputArea();
    }
};

window.send = window.playThinking;

function updateSimulation(dt) {
    animateInternals(dt);

    if (!playing || !timeline_sim.length) return;

    timer += dt;
    if (timer < STEP_TIME) return;
    timer = 0;

    const frame = timeline_sim[stageIndex];
    let stageName = "Embedding", logMsg = "", floatText = "";

    if (frame.stage === "tokenizer") {
        stageName = "Tokenizer";
        floatText = frame.meta.tokens.join(" | ");
        logMsg    = `Tokenizer → [${frame.meta.tokens.join(", ")}]`;
        currentTokens = frame.meta.tokens;
    } else if (frame.stage === "embedding") {
        stageName = "Embedding";
        floatText = `${currentTokens.length} tokens × 768 dims`;
        logMsg    = `Embedding: ${currentTokens.length} tokens → ℝ⁷⁶⁸`;
    } else if (frame.stage === "positional_encoding") {
        stageName = "Positional Encoding";
        floatText = "sin/cos position signals injected";
        logMsg    = "Positional Encoding: sin/cos offsets added";
    } else if (frame.stage === "transformer") {
        stageName = "Transformer " + frame.meta.index;
        floatText = "12 heads × 64 dims each";
        logMsg    = `Transformer ${frame.meta.index}: self-attention over ${currentTokens.length} tokens`;
    } else if (frame.stage === "sentence_meaning") {
        stageName = "Sentence Meaning";
        floatText = "CLS token pooled → 768-d vector";
        logMsg    = "Sentence Meaning: pooled to single representation";
    } else if (frame.stage === "output") {
        stageName = "Output";
        const candidates = generateNextTokenCandidates();
        const chosen     = candidates[0];
        outputTokens.push(chosen.word);
        floatText = `Next token → "${chosen.word}" (${(chosen.prob * 100).toFixed(1)}%)`;
        logMsg    = `Output: chose "${chosen.word}" (p=${chosen.prob.toFixed(3)})`;

        const el = document.getElementById("token-prob");
        if (el) {
            el.innerHTML = `<div class="out-label">Next Token Candidates</div>` +
                candidates.map(t =>
                    `<div class="token-row">
                        <span class="token-word">${t.word}</span>
                        <div class="token-bar-wrap">
                            <div class="token-bar" style="width:${(t.prob*100).toFixed(1)}%;background:${t.prob>0.4?"#00ffaa":t.prob>0.2?"#ffcc00":"#ff6644"}"></div>
                        </div>
                        <span class="token-pct">${(t.prob*100).toFixed(1)}%</span>
                    </div>`
                ).join("");
        }

        updateOutputArea();
    }

    const idx = MODEL_STRUCTURE.indexOf(stageName);
    highlight(stageName);
    createFloatingLabel(blocks[Math.max(idx, 0)], floatText);

    if (idx > 0)
        spawnFlow(Math.max(idx-1,0), idx,
            STAGE_COLORS[stageName.split(" ")[0]] || 0xffffff,
            currentTokens[0] || "");

    addStepLog(logMsg);
    updateOutputArea();

    stageIndex++;
    if (stageIndex >= timeline_sim.length) {
        playing = false;
        addStepLog(`Done. Output: "${outputTokens.join(" ")}"`);
    }
}


function updateParticles(dt) {
    for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.t += dt * 0.65;

        p.mesh.position.z = p.fromZ + (p.toZ - p.fromZ) * p.t;
        p.mesh.position.y = Math.sin(p.t * Math.PI) * 1.8;
        p.mesh.position.x = Math.sin(p.t * Math.PI * 2) * 0.5;

        if (p.labelSprite) {
            p.labelSprite.position.copy(p.mesh.position);
            p.labelSprite.position.y += 1.2;
        }

        if (p.t >= 1) {
            scene.remove(p.mesh);
            if (p.labelSprite) scene.remove(p.labelSprite);
            particles.splice(i, 1);
        }
    }
}

const raycaster = new THREE.Raycaster();
const mouse     = new THREE.Vector2();

window.addEventListener("click", e => {
    if (e.target !== renderer.domElement) return;

    mouse.x = (e.clientX / window.innerWidth)  * 2 - 1;
    mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const hit = raycaster.intersectObjects(blocks);

    if (!hit.length) { closeInternals(); return; }

    const block = hit[0].object;
    if (block.userData.name.includes("Transformer")) {
        if (expandedTransformer === block.userData.name) closeInternals();
        else { buildTransformerInternals(block); highlight(block.userData.name); }
    } else {
        closeInternals();
        highlight(block.userData.name);
        addStepLog(`${block.userData.name}: ${STAGE_DESCRIPTIONS[block.userData.name] || ""}`);
    }
});

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

let last = performance.now();

function animate() {
    requestAnimationFrame(animate);
    const now = performance.now();
    const dt  = Math.min((now - last) / 1000, 0.05);
    last = now;

    updateSimulation(dt);
    updateParticles(dt);

    keyLight.intensity = 5 + Math.sin(now * 0.001) * 1.5;
    controls.update();
    renderer.render(scene, camera);
}

animate();

window.addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
});

document.addEventListener("keydown", e => {
    if (e.key === "Enter" && document.getElementById("prompt") === document.activeElement)
        window.playThinking();
});