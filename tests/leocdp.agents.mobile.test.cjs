const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const source = fs.readFileSync(
  path.join(__dirname, "../resources/js/leocdp.agents.js"),
  "utf8"
);

function setupAgents({ width, reducedMotion, chatVisible = true }) {
  let quickActionClick;
  const handlers = {};
  const scrollCalls = [];
  const sentQuestions = [];
  const humanMessages = [];
  const ajaxRequests = [];
  const reloadedTouchpoints = [];
  const styleProperties = {};
  let decodedQr = null;
  const chat = {
    getClientRects: () => chatVisible ? [{}] : [],
    scrollIntoView: options => scrollCalls.push(["chat", options]),
  };
  const chatCard = {
    scrollIntoView: options => scrollCalls.push(["card", options]),
  };

  function jquery(subject) {
    if (typeof subject === "function") {
      subject();
      return;
    }

    let proxy;
    const stub = {
      length: 0,
      attr(name, value) {
        if (arguments.length === 1 && subject && subject.attributes) {
          return subject.attributes[name];
        }
        return proxy;
      },
      on(event, selector, handler) {
        if (subject === "#leobot_quick_actions") {
          quickActionClick = handler;
        }
        const callback = typeof selector === "function" ? selector : handler;
        if (typeof callback === "function") {
          handlers[subject + ":" + event] = callback;
        }
        return proxy;
      },
    };
    proxy = new Proxy(stub, {
      get(target, property) {
        if (property in target) return target[property];
        return function () { return proxy; };
      },
    });
    return proxy;
  }

  jquery.getScript = function () {};
  jquery.ajax = function (options) {
    const request = { options };
    ajaxRequests.push(request);
    return {
      done(callback) {
        request.success = callback;
        return this;
      },
      fail(callback) {
        request.failure = callback;
        return this;
      },
    };
  };
  jquery.getJSON = function () {
    return {
      fail(callback) {
        return this;
      },
    };
  };

  const document = {
    documentElement: {
      style: {
        setProperty(name, value) {
          styleProperties[name] = value;
        },
      },
    },
    getElementById(id) {
      if (id === "LEO_ChatBot_Container") return chat;
      if (id === "leobot_chat_container") return chatCard;
      return null;
    },
  };
  const window = {
    LEO_UI_LANGUAGE: "en",
    clearTimeout() {},
    currentUserProfile: {
      visitorId: "visitor",
      displayName: "Ada",
      touchpointId: "",
      touchpointLocked: false,
      touchpointName: "Web visitor",
      touchpointDescription: "",
      touchpointType: "web",
      touchpointKeywords: [],
      latitude: 10,
      longitude: 106,
    },
    getBotUI() {
      return { message: { human: message => humanMessages.push(message) } };
    },
    innerWidth: width,
    innerHeight: 844,
    localStorage: { setItem() {} },
    location: { hostname: "localhost", protocol: "http:" },
    Html5Qrcode: class {
      start(config, options, success) {
        decodedQr = success;
        return Promise.resolve();
      }
      stop() {
        return Promise.resolve();
      }
      clear() {
        return Promise.resolve();
      }
    },
    matchMedia(query) {
      return {
        matches: query.includes("prefers-reduced-motion")
          ? reducedMotion
          : width <= 991,
      };
    },
    sendQuestionToLeoAI(context, question) {
      sentQuestions.push({ context, question });
    },
    setTimeout() {
      return 1;
    },
    startLeoChatBot() {},
    reloadLeoChatBotSession(touchpoint) {
      reloadedTouchpoints.push(touchpoint);
    },
    leoBotStarted: true,
  };
  const context = vm.createContext({
    CDP_TRACKING: false,
    HOSTNAME: "localhost",
    BASE_URL_GET_VISITOR_INFO: "/visitor-info",
    BASE_URL_QR_TOUCHPOINT: "/touchpoint/web-qr",
    currentUserProfile: window.currentUserProfile,
    document,
    jQuery: jquery,
    lscache: { set() {} },
    touchpointCacheKey(visitorId) {
      return "touchpoint_id_" + visitorId;
    },
    URL,
    URLSearchParams,
    window,
  });
  vm.runInContext(source, context);

  return {
    clickQuickAction(question) {
      assert.equal(typeof quickActionClick, "function");
      quickActionClick.call({ attributes: { "data-question": question } });
    },
    humanMessages,
    scanQr(value) {
      assert.equal(typeof decodedQr, "function");
      decodedQr(value);
    },
    clickQrStart() {
      handlers["#leobot_qr_start_button:click"]();
    },
    clickQrChat() {
      handlers["#leobot_qr_chat_button:click"]();
    },
    ajaxRequests,
    reloadedTouchpoints,
    scrollCalls,
    sentQuestions,
    styleProperties,
  };
}

test("quick actions scroll mobile users to the chat and send the prompt", () => {
  const app = setupAgents({ width: 390, reducedMotion: false });
  assert.equal(app.styleProperties["--leobot-chat-height"], "540px");
  app.clickQuickAction("Find coffee nearby");

  assert.equal(JSON.stringify(app.scrollCalls), JSON.stringify([
    ["chat", { behavior: "smooth", block: "start" }],
  ]));
  assert.equal(JSON.stringify(app.sentQuestions), JSON.stringify([
    { context: "ask", question: "Find coffee nearby" },
  ]));
  assert.equal(
    JSON.stringify(app.humanMessages),
    JSON.stringify([{ content: "Find coffee nearby" }])
  );
});

test("QR URL is crawled on start-chat click and reloads the touchpoint session", () => {
  const app = setupAgents({ width: 390, reducedMotion: false });
  app.clickQrStart();
  app.scanQr("https://example.com/temple-guide");
  assert.equal(app.ajaxRequests.length, 0);

  app.clickQrChat();
  assert.equal(app.ajaxRequests.length, 1);
  assert.equal(app.ajaxRequests[0].options.url, "/touchpoint/web-qr");
  assert.equal(
    JSON.parse(app.ajaxRequests[0].options.data).full_url,
    "https://example.com/temple-guide"
  );

  app.ajaxRequests[0].success({
    touchpoint_id: "qr-temple",
    full_url: "https://example.com/temple-guide",
    name: "Temple Guide",
    description: "A guide scanned from QR",
    type: "web_qr",
    keywords: ["temple"],
  });

  assert.equal(app.reloadedTouchpoints.length, 1);
  assert.equal(app.reloadedTouchpoints[0].touchpointId, "qr-temple");
  assert.equal(app.reloadedTouchpoints[0].type, "web_qr");
  assert.equal(
    app.reloadedTouchpoints[0].fullUrl,
    "https://example.com/temple-guide"
  );

  app.clickQrChat();
  assert.equal(app.ajaxRequests.length, 1);
  assert.equal(app.reloadedTouchpoints.length, 2);
});

test("uses the visible chat card if the chat surface is still hidden", () => {
  const app = setupAgents({
    width: 390,
    reducedMotion: true,
    chatVisible: false,
  });
  app.clickQuickAction("Find coffee nearby");

  assert.equal(JSON.stringify(app.scrollCalls), JSON.stringify([
    ["card", { behavior: "auto", block: "start" }],
  ]));
});

test("does not change page scroll on desktop", () => {
  const app = setupAgents({ width: 1280, reducedMotion: false });
  app.clickQuickAction("Find coffee nearby");

  assert.equal(app.scrollCalls.length, 0);
});
