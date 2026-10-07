// Run these checks with node test_control_room.js.
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
class Element {
  constructor() {
    this.children = [];
    this.classes = new Set();
    this.style = { setProperty() {} };
    this.attributes = {};
    this.classList = {
      add: (c) => this.classes.add(c),
      remove: (c) => this.classes.delete(c),
      toggle: (c, on) => (on ? this.classes.add(c) : this.classes.delete(c)),
    };
  }
  append(...items) {
    this.children.push(...items);
  }
  replaceChildren() {
    this.children = [];
  }
  setAttribute(key, value) {
    this.attributes[key] = value;
  }
}
async function test(side, pauseSource = false) {
  const elements = new Map(
    ["wall", "summary", "previous", "next", "page", "pause", "fullscreen"].map(
      (id) => [id, new Element()],
    ),
  );
  const document = {
    getElementById: (id) => elements.get(id),
    createElement: () => new Element(),
    referrer: "http://localhost:8501/",
    visibilityState: "visible",
    documentElement: new Element(),
  };
  const feeds = Array.from({ length: 10 }, (_, i) => ({
    id: String(i),
    name: "Camera " + i,
    paused: pauseSource && i === 9,
    location: "",
  }));
  let states = feeds.map((feed) => ({
    id: feed.id,
    sequence: 1,
    status: "Live",
    jpeg: "test-jpeg",
    live: true,
    searching: feed.id === "9",
    query: "red car",
  }));
  let requested;
  const context = vm.createContext({
    document,
    window: { location: { href: "about:srcdoc" } },
    URL,
    Map,
    JSON,
    encodeURIComponent,
    AbortSignal,
    setTimeout() {},
    fetch: async (url) => {
      requested = url;
      return { ok: true, json: async () => states };
    },
  });
  const html = fs.readFileSync("control_room.html", "utf8");
  const script = html
    .match(/<script>([\s\S]*?)<\/script>/)[1]
    .replace(
      "__CONFIG__",
      JSON.stringify({ side, feeds, port: 8123, token: "private-room" }),
    );
  vm.runInContext(script, context);
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(elements.get("wall").children.length, Math.min(side * side, feeds.length - Math.floor(9 / (side * side)) * side * side));
  const active = elements
    .get("wall")
    .children.find((tile) => tile.classes.has("searching"));
  assert.ok(active, "active camera follows the visible page");
  assert.equal(
    active.children[1].children[1].textContent,
    "SEARCHING: red car",
  );
  assert.equal(
    active.children[0].children[0].src,
    "data:image/jpeg;base64,test-jpeg",
  );
  assert.ok(
    requested.startsWith("http://localhost:8123/frames?token=private-room"),
  );
  if (pauseSource) {
    states = states.map(state => ({ ...state, jpeg: "new-frame", sequence: 2 }));
    await context.poll();
    assert.equal(active.children[0].children[0].src, "data:image/jpeg;base64,test-jpeg");
    assert.ok(!new URL(requested).searchParams.get("visible").split(",").includes("9"));
  }
  elements.get("pause").onclick();
  states = states.map((state) => ({ ...state, searching: false, jpeg: null }));
  await context.poll();
  assert.equal(
    elements.get("wall").children.some((tile) => tile.classes.has("searching")),
    false,
  );
  assert.equal(elements.get("pause").textContent, "Resume playback");
  assert.ok(requested.endsWith("&visible="));
}
(async () => {
  for (const side of [1, 2, 3, 4, 9]) await test(side);
  await test(4, true);
  console.log(
    "Control room layouts, playback, highlighting, paging, and pause checks passed.",
  );
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
