/* UI glue: render the board, handle clicks, drive the bot's turns. */
(function () {
  "use strict";
  var SVGNS = "http://www.w3.org/2000/svg";
  var M = 40, STEP = 100, DOT_R = 6;

  var MODEL = null, state = null, humanPlayer = 1, sims = 400, busy = false;
  var gen = 0; // bumped on New game; a running bot loop aborts if it changes
  var P1 = "#2563EB", P2 = "#F97316";            // player colors
  var edgeOwner = new Array(24).fill(0);          // which player drew each edge
  function pColor(p) { return p === 1 ? P1 : P2; }
  var boardEl = document.getElementById("board");
  var statusEl = document.getElementById("status");
  var scoreEl = document.getElementById("score");

  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  // endpoints of an edge index, in SVG coords
  function edgeCoords(e) {
    if (e < 12) { var r = Math.floor(e / 3), c = e % 3;
      return [M + c * STEP, M + r * STEP, M + (c + 1) * STEP, M + r * STEP]; }
    var i = e - 12, rr = Math.floor(i / 4), cc = i % 4;
    return [M + cc * STEP, M + rr * STEP, M + cc * STEP, M + (rr + 1) * STEP];
  }
  function el(tag, attrs) {
    var n = document.createElementNS(SVGNS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }

  function render() {
    while (boardEl.firstChild) boardEl.removeChild(boardEl.firstChild);

    // boxes (fill claimed cells in the owner's color, label You/Bot)
    for (var b = 0; b < 9; b++) {
      var br = Math.floor(b / 3), bc = b % 3, owner = state.box_owner[b];
      if (owner !== 0) {
        var col = pColor(owner);
        boardEl.appendChild(el("rect", { x: M + bc * STEP, y: M + br * STEP,
          width: STEP, height: STEP, fill: col + "22" }));
        var t = el("text", { x: M + bc * STEP + STEP / 2, y: M + br * STEP + STEP / 2 + 7,
          "text-anchor": "middle", "font-size": 26, "font-weight": 700, fill: col });
        t.textContent = owner === humanPlayer ? "You" : "Bot";
        boardEl.appendChild(t);
      }
    }

    // edges: the wide transparent hit target is appended BEFORE its own line so
    // the CSS `.hit:hover + .edge-line` targets the correct edge (not the next).
    var myTurn = !busy && !DAB.isTerminal(state) && state.current_player === humanPlayer;
    for (var e = 0; e < 24; e++) {
      var p = edgeCoords(e), drawn = state.edges[e] === 1;
      if (!drawn && myTurn) {
        var hit = el("line", { x1: p[0], y1: p[1], x2: p[2], y2: p[3], class: "hit hoverable" });
        (function (edge) { hit.addEventListener("click", function () { onPlay(edge); }); })(e);
        boardEl.appendChild(hit);
      }
      var line = el("line", { x1: p[0], y1: p[1], x2: p[2], y2: p[3],
        class: "edge-line " + (drawn ? "drawn" : "undrawn") });
      if (drawn) line.setAttribute("stroke", pColor(edgeOwner[e] || 1));
      boardEl.appendChild(line);
    }

    // dots on top
    for (var r = 0; r < 4; r++) for (var c = 0; c < 4; c++)
      boardEl.appendChild(el("circle", { cx: M + c * STEP, cy: M + r * STEP, r: DOT_R, class: "dot" }));

    updateScore();
  }

  function updateScore() {
    var p = [0, 0, 0];
    for (var i = 0; i < 9; i++) p[state.box_owner[i]]++;
    var you = p[humanPlayer], bot = p[humanPlayer === 1 ? 2 : 1];
    var youCol = pColor(humanPlayer), botCol = pColor(humanPlayer === 1 ? 2 : 1);
    scoreEl.innerHTML = 'Boxes — <b style="color:' + youCol + '">You ' + you +
      '</b> · <b style="color:' + botCol + '">Bot ' + bot + "</b>";
  }

  function setStatus(msg) { statusEl.textContent = msg; }

  function updateTurnStatus() {
    if (DAB.isTerminal(state)) {
      var w = DAB.winner(state);
      if (w === 0) setStatus("It's a tie!");
      else setStatus(w === humanPlayer ? "You win! 🎉" : "Bot wins.");
      return;
    }
    setStatus(state.current_player === humanPlayer ? "Your move" : "Bot is thinking…");
  }

  function onPlay(edge) {
    if (busy || DAB.isTerminal(state)) return;
    if (state.current_player !== humanPlayer || state.edges[edge] !== 0) return;
    edgeOwner[edge] = state.current_player;
    state = DAB.applyMove(state, edge);
    render();
    if (!DAB.isTerminal(state) && state.current_player !== humanPlayer) botTurn();
    else updateTurnStatus();
  }

  async function botTurn() {
    var myGen = gen;
    busy = true; render();
    while (!DAB.isTerminal(state) && state.current_player !== humanPlayer) {
      updateTurnStatus();
      await sleep(40); // let the UI paint the "thinking" state
      if (gen !== myGen) return; // a New game superseded this loop
      var move = DAB.netMctsMove(MODEL, state, sims, 1.5);
      if (gen !== myGen) return;
      edgeOwner[move] = state.current_player;
      state = DAB.applyMove(state, move);
      render();
      await sleep(350); // pace multi-move chains so you can watch them
      if (gen !== myGen) return;
    }
    busy = false;
    render();
    updateTurnStatus();
  }

  function newGame() {
    gen++; // invalidate any bot loop still running from a previous game
    humanPlayer = +document.getElementById("side").value;
    sims = +document.getElementById("sims").value;
    edgeOwner = new Array(24).fill(0);
    boardEl.style.setProperty("--hover", pColor(humanPlayer)); // preview in your color
    state = DAB.initialState();
    busy = false;
    render();
    if (state.current_player !== humanPlayer) botTurn();
    else updateTurnStatus();
  }

  // Weights are embedded (model.js sets window.DAB_MODEL), so this works even
  // when opened as a local file:// — no server required.
  if (window.DAB_MODEL) {
    MODEL = window.DAB_MODEL;
    document.getElementById("controls").style.display = "flex";
    document.getElementById("newgame").addEventListener("click", newGame);
    newGame();
  } else {
    setStatus("Could not load model.js — make sure it sits next to index.html.");
  }
})();
