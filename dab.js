/* Dots and Boxes (3x3 boxes) engine + net + net-guided MCTS, in plain JS.
 * Ported 1:1 from the Python project so the trained weights play identically.
 * Works in the browser and in Node (for the self-test). */

(function (root) {
  "use strict";

  var NUM_EDGES = 24, NUM_BOXES = 9;

  // --- board geometry (matches game.py) ---
  function hEdge(r, c) { return r * 3 + c; }            // horizontal, r 0..3 c 0..2
  function vEdge(r, c) { return 12 + r * 4 + c; }       // vertical,   r 0..2 c 0..3

  var BOX_EDGES = [];
  for (var r = 0; r < 3; r++) for (var c = 0; c < 3; c++)
    BOX_EDGES.push([hEdge(r, c), hEdge(r + 1, c), vEdge(r, c), vEdge(r, c + 1)]);

  var EDGE_BOXES = [];
  for (var e = 0; e < NUM_EDGES; e++) {
    var bs = [];
    for (var b = 0; b < NUM_BOXES; b++) if (BOX_EDGES[b].indexOf(e) >= 0) bs.push(b);
    EDGE_BOXES.push(bs);
  }

  // --- state: {edges:[24], box_owner:[9], current_player:1|2} ---
  function initialState() {
    return { edges: new Array(NUM_EDGES).fill(0),
             box_owner: new Array(NUM_BOXES).fill(0), current_player: 1 };
  }
  function legalMoves(s) {
    var m = []; for (var i = 0; i < NUM_EDGES; i++) if (s.edges[i] === 0) m.push(i); return m;
  }
  function sidesDrawn(edges, b) {
    var be = BOX_EDGES[b]; return edges[be[0]] + edges[be[1]] + edges[be[2]] + edges[be[3]];
  }
  function applyMove(s, edge) {
    var edges = s.edges.slice(); edges[edge] = 1;
    var owner = s.box_owner.slice(); var completed = 0;
    for (var b = 0; b < NUM_BOXES; b++)
      if (owner[b] === 0 && sidesDrawn(edges, b) === 4) { owner[b] = s.current_player; completed++; }
    var next = completed > 0 ? s.current_player : (s.current_player === 1 ? 2 : 1);
    return { edges: edges, box_owner: owner, current_player: next };
  }
  function isTerminal(s) { for (var i = 0; i < NUM_EDGES; i++) if (s.edges[i] === 0) return false; return true; }
  function winner(s) {
    var p1 = 0, p2 = 0;
    for (var i = 0; i < NUM_BOXES; i++) { if (s.box_owner[i] === 1) p1++; else if (s.box_owner[i] === 2) p2++; }
    return p1 > p2 ? 1 : (p2 > p1 ? 2 : 0);
  }
  function terminalValue(s, player) { var w = winner(s); return w === 0 ? 0 : (w === player ? 1 : -1); }

  // --- encoding (matches network.encode_state, length 83) ---
  function encodeState(s) {
    var me = s.current_player, f = [], i, b;
    for (i = 0; i < NUM_EDGES; i++) f.push(s.edges[i]);                         // 24 edge bits
    for (b = 0; b < NUM_BOXES; b++)                                             // 9 box values
      f.push(s.box_owner[b] === 0 ? 0 : (s.box_owner[b] === me ? 1 : -1));
    var capture = new Array(NUM_EDGES).fill(0), gift = new Array(NUM_EDGES).fill(0);
    for (i = 0; i < NUM_EDGES; i++) {
      if (s.edges[i] === 1) continue;
      var bx = EDGE_BOXES[i];
      for (var k = 0; k < bx.length; k++) {
        var sd = sidesDrawn(s.edges, bx[k]);
        if (sd === 3) capture[i] = 1; else if (sd === 2) gift[i] = 1;
      }
    }
    for (i = 0; i < NUM_EDGES; i++) f.push(capture[i]);                         // 24 capture flags
    for (i = 0; i < NUM_EDGES; i++) f.push(gift[i]);                            // 24 gift flags
    var neutral = 0;
    for (i = 0; i < NUM_EDGES; i++) if (s.edges[i] === 0 && capture[i] === 0 && gift[i] === 0) neutral++;
    f.push(neutral / NUM_EDGES); f.push(neutral % 2);                          // 2 safe scalars
    return f;
  }

  // --- network forward pass (matches PolicyValueNet) ---
  function linRelu(W, bias, x) {
    var out = new Array(W.length);
    for (var i = 0; i < W.length; i++) {
      var row = W[i], acc = bias[i];
      for (var j = 0; j < row.length; j++) acc += row[j] * x[j];
      out[i] = acc > 0 ? acc : 0;   // ReLU
    }
    return out;
  }
  function linear(W, bias, x) {
    var out = new Array(W.length);
    for (var i = 0; i < W.length; i++) {
      var row = W[i], acc = bias[i];
      for (var j = 0; j < row.length; j++) acc += row[j] * x[j];
      out[i] = acc;
    }
    return out;
  }
  function forward(model, x) {
    var h1 = linRelu(model.trunk0_w, model.trunk0_b, x);
    var h2 = linRelu(model.trunk2_w, model.trunk2_b, h1);
    var logits = linear(model.policy_w, model.policy_b, h2);
    var v = linear(model.value_w, model.value_b, h2)[0];
    return { logits: logits, value: Math.tanh(v) };
  }

  // --- net-guided MCTS (PUCT + value-head leaf eval) ---
  function Node(s) {
    this.state = s; this.player = s.current_player; this.prior = 0;
    this.children = {}; this.visits = 0; this.valueSum = 0; this.terminal = isTerminal(s);
  }
  Node.prototype.q = function () { return this.visits ? this.valueSum / this.visits : 0; };
  Node.prototype.expanded = function () { for (var k in this.children) return true; return false; };

  function softmaxLegal(logits, legal) {
    var mx = -Infinity, i;
    for (i = 0; i < legal.length; i++) if (logits[legal[i]] > mx) mx = logits[legal[i]];
    var sum = 0, ex = [];
    for (i = 0; i < legal.length; i++) { var e = Math.exp(logits[legal[i]] - mx); ex.push(e); sum += e; }
    var p = {};
    for (i = 0; i < legal.length; i++) p[legal[i]] = ex[i] / sum;
    return p;
  }
  function evaluateLeaf(model, node) {
    var out = forward(model, encodeState(node.state));
    var legal = legalMoves(node.state);
    var priors = softmaxLegal(out.logits, legal);
    for (var i = 0; i < legal.length; i++) {
      var a = legal[i], child = new Node(applyMove(node.state, a));
      child.prior = priors[a]; node.children[a] = child;
    }
    return out.value;   // from node.player's perspective
  }
  function puctSelect(node, cPuct) {
    var best = -Infinity, bestA = -1, bestChild = null, sqrtN = Math.sqrt(node.visits);
    for (var a in node.children) {
      var ch = node.children[a];
      var q = ch.player === node.player ? ch.q() : -ch.q();
      var u = cPuct * ch.prior * sqrtN / (1 + ch.visits);
      var score = q + u;
      if (score > best) { best = score; bestA = a; bestChild = ch; }
    }
    return bestChild;
  }
  function backup(path, leafPlayer, value) {
    for (var i = 0; i < path.length; i++) {
      path[i].visits += 1;
      path[i].valueSum += (path[i].player === leafPlayer) ? value : -value;
    }
  }
  function simulate(model, root, cPuct) {
    var node = root, path = [root];
    while (node.expanded() && !node.terminal) { node = puctSelect(node, cPuct); path.push(node); }
    var leafPlayer = node.player, value;
    if (node.terminal) value = terminalValue(node.state, leafPlayer);
    else value = evaluateLeaf(model, node);
    backup(path, leafPlayer, value);
  }
  function netMctsMove(model, state, sims, cPuct) {
    if (cPuct === undefined) cPuct = 1.5;
    var root = new Node(state);
    for (var i = 0; i < sims; i++) simulate(model, root, cPuct);
    var best = -1, bestA = -1;
    for (var a in root.children) if (root.children[a].visits > best) { best = root.children[a].visits; bestA = +a; }
    return bestA;
  }
  function greedyMove(model, state) {   // net alone (no search), for reference
    var out = forward(model, encodeState(state)), legal = legalMoves(state);
    var best = -Infinity, bestA = legal[0];
    for (var i = 0; i < legal.length; i++) if (out.logits[legal[i]] > best) { best = out.logits[legal[i]]; bestA = legal[i]; }
    return bestA;
  }

  var API = {
    NUM_EDGES: NUM_EDGES, NUM_BOXES: NUM_BOXES, BOX_EDGES: BOX_EDGES,
    hEdge: hEdge, vEdge: vEdge,
    initialState: initialState, legalMoves: legalMoves, applyMove: applyMove,
    isTerminal: isTerminal, winner: winner, encodeState: encodeState,
    forward: forward, netMctsMove: netMctsMove, greedyMove: greedyMove,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = API;
  else root.DAB = API;
})(typeof window !== "undefined" ? window : globalThis);
