(function ($, window, document) {
  "use strict";

  window.LEO_UI_TEXTS = {
    vi: {
      help: "Trợ giúp",
      exploreWithContext: "Khám phá theo ngữ cảnh",
      heroTitle: "Khám phá địa điểm cùng LEO",
      heroDescription: "Hỏi về các địa điểm gần bạn, nhận gợi ý địa phương hoặc tiếp tục trò chuyện về địa điểm bạn đã chọn.",
      heroDescriptionMobile: "Khám phá địa điểm gần bạn cùng LEO.",
      locationHelps: "Vị trí giúp LEO hữu ích hơn",
      locationPrivacy: "Trình duyệt sẽ hỏi quyền truy cập. Chúng tôi chỉ dùng vị trí để tìm địa điểm gần bạn, không dùng để theo dõi.",
      chatWithLeo: "Trò chuyện với LEO",
      online: "Đang trực tuyến",
      chatHint: "Thử “quán cà phê gần tôi” hoặc chọn một gợi ý.",
      locationOff: "Đã tắt vị trí",
      preparingAssistant: "Đang chuẩn bị trợ lý địa phương…",
      yourContext: "Ngữ cảnh của bạn",
      currentLocation: "Vị trí hiện tại",
      checkingLocation: "Đang kiểm tra khả năng truy cập vị trí…",
      coordinates: "Tọa độ",
      notAvailable: "Chưa có dữ liệu",
      accuracy: "Độ chính xác",
      available: "Có dữ liệu",
      language: "Ngôn ngữ",
      useMyLocation: "Dùng vị trí của tôi",
      refreshLocation: "Làm mới vị trí",
      gettingLocation: "Đang lấy vị trí…",
      permissionHint: "Bạn có thể thay đổi quyền này trong cài đặt trình duyệt.",
      startExploring: "Bắt đầu khám phá",
      whatLookingFor: "Bạn đang tìm gì?",
      loadingRecommendations: "Đang tải gợi ý gần bạn…",
      locationSuggestionsUnavailable: "Cho phép vị trí để tải gợi ý gần bạn.",
      nearbyHint: "LEO sẽ dùng vị trí của bạn khi tìm kiếm địa điểm gần đó.",
      madeBy: "Phát triển bởi",
      privateByDesign: "· Riêng tư theo thiết kế",
      locationOptional: "Dịch vụ vị trí là tùy chọn.",
      helpTitle: "Sử dụng LEO hiệu quả hơn",
      helpDescription: "Hãy hỏi tự nhiên. Để khám phá địa điểm, hãy thêm “gần tôi” và cho phép truy cập vị trí khi được hỏi.",
      helpExampleOne: "“Tìm nhà thờ gần nhất”",
      helpExampleTwo: "“Gần đây có gì thú vị?”",
      helpExampleThree: "Chọn số của một địa điểm để đặt làm chủ đề cho câu hỏi tiếp theo.",
      locationRequesting: "Đang chờ quyền truy cập từ trình duyệt…",
      locationReady: "Đã sẵn sàng sử dụng vị trí. Câu trả lời gần bạn sẽ phù hợp hơn.",
      locationDenied: "Bạn chưa cấp quyền vị trí. Bạn vẫn có thể trò chuyện hoặc thử lại.",
      locationUnavailable: "Trình duyệt này không cung cấp được vị trí.",
      locationError: "Không thể chuẩn bị vị trí. Vui lòng thử lại.",
      locationReadyBadge: "Đã bật vị trí",
      chatPlaceholder: "Bạn muốn tìm gì?",
      emptyQuestion: "Vui lòng nhập câu hỏi hoặc số của địa điểm.",
      networkError: "Không thể gửi tin nhắn. Vui lòng thử lại.",
      assistantLoadError: "Không thể tải LEO. Vui lòng tải lại trang và thử lại.",
      googleSignedInAs: "Đã đăng nhập: ",
      scanTouchpoint: "Quét điểm chạm",
      scanTouchpointTitle: "Bắt đầu từ mã QR",
      scanTouchpointDescription: "Quét mã QR của điểm chạm để LEO dùng đúng ngữ cảnh và nội dung liên quan khi trò chuyện.",
      scanTouchpointHint: "Cho phép camera để quét mã QR.",
      scanTouchpointButton: "Quét mã QR",
      stopScanning: "Dừng quét",
      touchpointReady: "Đã chọn điểm chạm",
      startTouchpointChat: "Bắt đầu trò chuyện",
      qrScannerUnavailable: "Không thể tải trình quét QR trên trình duyệt này.",
      qrScannerStarting: "Đang mở camera…",
      qrScannerScanning: "Đưa mã QR vào khung hình.",
      qrScannerInvalid: "Mã QR không chứa điểm chạm hợp lệ.",
      qrScannerError: "Không thể mở camera. Vui lòng kiểm tra quyền truy cập và thử lại.",
      qrScannerFetching: "Đang tải nội dung từ liên kết QR…",
      qrScannerFetchError: "Không thể tải nội dung liên kết QR. Vui lòng thử lại.",
      qrScannerSelected: "Đã chọn điểm chạm. Bạn có thể bắt đầu trò chuyện.",
      qrScannerNoVisitor: "Chưa tạo được phiên khách truy cập. Vui lòng thử lại."
    },
    en: {
      help: "Help",
      exploreWithContext: "Explore with context",
      heroTitle: "Explore Places with LEO",
      heroDescription: "Ask about nearby places, get local ideas, or continue a conversation about a place you selected.",
      heroDescriptionMobile: "Discover nearby places with a little help from LEO.",
      locationHelps: "Location helps LEO be useful",
      locationPrivacy: "Your browser asks for permission. We use your location to find nearby places, not to track you.",
      chatWithLeo: "Chat with LEO",
      online: "Online",
      chatHint: "Try “coffee near me” or choose a suggestion.",
      locationOff: "Location off",
      preparingAssistant: "Preparing your local assistant…",
      yourContext: "Your context",
      currentLocation: "Current location",
      checkingLocation: "Checking whether location is available…",
      coordinates: "Coordinates",
      notAvailable: "Not available",
      accuracy: "Accuracy",
      available: "Available",
      language: "Language",
      useMyLocation: "Use my location",
      refreshLocation: "Refresh location",
      gettingLocation: "Getting location…",
      permissionHint: "You can change this permission in your browser settings.",
      startExploring: "Start exploring",
      whatLookingFor: "What are you looking for?",
      loadingRecommendations: "Loading nearby suggestions…",
      locationSuggestionsUnavailable: "Allow location access to load nearby suggestions.",
      nearbyHint: "LEO will use your location when a nearby search needs it.",
      madeBy: "Made by",
      privateByDesign: "· Private by design",
      locationOptional: "Location services are optional.",
      helpTitle: "Get more from LEO",
      helpDescription: "Ask naturally. For local discovery, include “near me” and allow location access when prompted.",
      helpExampleOne: "“Find the nearest church”",
      helpExampleTwo: "“What can I do around here?”",
      helpExampleThree: "Choose a place number to make it the focus of your next questions.",
      locationRequesting: "Waiting for browser permission…",
      locationReady: "Location ready. Nearby answers will be more relevant.",
      locationDenied: "Location access was not granted. You can still chat, or try again.",
      locationUnavailable: "Location is unavailable in this browser.",
      locationError: "We could not prepare location. Please try again.",
      locationReadyBadge: "Location ready",
      chatPlaceholder: "What would you like to find?",
      emptyQuestion: "Please enter a question or a place number.",
      networkError: "Unable to send your message. Please try again.",
      assistantLoadError: "LEO could not load. Please refresh the page and try again.",
      googleSignedInAs: "Signed in as ",
      scanTouchpoint: "Scan a touchpoint",
      scanTouchpointTitle: "Start from a QR code",
      scanTouchpointDescription: "Scan a touchpoint QR code so LEO can use the right context and related knowledge while chatting.",
      scanTouchpointHint: "Allow camera access to scan a QR code.",
      scanTouchpointButton: "Scan QR code",
      stopScanning: "Stop scanning",
      touchpointReady: "Touchpoint selected",
      startTouchpointChat: "Start chatting",
      qrScannerUnavailable: "The QR scanner is not available in this browser.",
      qrScannerStarting: "Opening camera…",
      qrScannerScanning: "Place the QR code inside the frame.",
      qrScannerInvalid: "This QR code does not contain a valid touchpoint.",
      qrScannerError: "Unable to open the camera. Check permission and try again.",
      qrScannerFetching: "Loading content from the QR link…",
      qrScannerFetchError: "Unable to load the QR link content. Please try again.",
      qrScannerSelected: "Touchpoint selected. You can start chatting.",
      qrScannerNoVisitor: "A visitor session is not ready yet. Please try again."
    }
  };

  window.leoUiText = function (key) {
    var language = window.LEO_UI_LANGUAGE === "en" ? "en" : "vi";
    return (window.LEO_UI_TEXTS[language] && window.LEO_UI_TEXTS[language][key])
      || window.LEO_UI_TEXTS.vi[key]
      || key;
  };

  var latestLocationDetail = { state: "pending" };
  var recommendedActions = null;
  var qrScanner = null;
  var qrScannerRunning = false;
  var qrTouchpointLoading = false;
  var pendingQrTouchpoint = null;

  function setQrStatus(key, className) {
    var $status = $("#leobot_qr_status");
    if (!$status.length) return;
    $status.attr("class", "small mb-3 " + (className || "text-secondary"))
      .empty()
      .append($("<i>", {
        class: "fa " + (className === "text-danger" ? "fa-exclamation-circle" : "fa-info-circle") + " me-1",
        "aria-hidden": "true"
      }))
      .append($("<span>").text(window.leoUiText(key)));
  }

  function normalizeQrTouchpoint(rawValue) {
    var raw = typeof rawValue === "string" ? rawValue.trim() : "";
    if (!raw) return null;

    var parsed = null;
    try {
      parsed = JSON.parse(raw);
    } catch (error) {
      parsed = null;
    }

    var source = parsed && typeof parsed === "object" ? parsed : {};
    var nestedTouchpoint =
      source.touchpoint && typeof source.touchpoint === "object"
        ? source.touchpoint
        : {};
    var nestedData =
      source.data && typeof source.data === "object" ? source.data : {};
    var sourceUrl =
      source.full_url ||
      source.url ||
      source.uri ||
      source.href ||
      nestedData.full_url ||
      nestedData.url ||
      nestedData.uri;
    if (!sourceUrl && /^https?:\/\//i.test(raw)) {
      sourceUrl = raw;
    }
    var normalizedUrl = null;
    if (typeof sourceUrl === "string" && sourceUrl.trim()) {
      try {
        var parsedUrl = new URL(sourceUrl.trim(), window.location.origin);
        if (parsedUrl.protocol === "http:" || parsedUrl.protocol === "https:") {
          normalizedUrl = parsedUrl.toString();
        }
      } catch (error) {
        normalizedUrl = null;
      }
    }
    var touchpointId =
      source.touchpoint_id ||
      source.touchpointId ||
      (typeof source.touchpoint === "string" ? source.touchpoint : null) ||
      source.id ||
      source.code ||
      nestedTouchpoint.touchpoint_id ||
      nestedTouchpoint.touchpointId ||
      nestedTouchpoint.id ||
      nestedData.touchpoint_id ||
      nestedData.touchpointId ||
      null;
    if (!touchpointId && typeof parsed === "string") {
      touchpointId = parsed;
    }
    if (!touchpointId) {
      try {
        var url = new URL(raw, window.location.origin);
        var hashParams = new URLSearchParams(url.hash.replace(/^#\/?/, ""));
        touchpointId =
          url.searchParams.get("touchpoint_id") ||
          url.searchParams.get("touchpointId") ||
          url.searchParams.get("touchpoint") ||
          url.searchParams.get("tp") ||
          url.searchParams.get("id") ||
          hashParams.get("touchpoint_id") ||
          hashParams.get("touchpointId") ||
          hashParams.get("touchpoint") ||
          hashParams.get("tp") ||
          hashParams.get("id");

        if (!touchpointId) {
          var pathParts = url.pathname.split("/").filter(function (part) {
            return part.trim();
          });
          var touchpointPathIndex = pathParts.findIndex(function (part) {
            return /^(?:touchpoint|touchpoints|tp|web_qr)$/i.test(part);
          });
          if (touchpointPathIndex >= 0) {
            touchpointId = pathParts[touchpointPathIndex + 1] || null;
          }
        }
      } catch (error) {
        touchpointId = null;
      }
    }
    if (!touchpointId) {
      var prefixedId = raw.match(
        /^(?:touchpoint|touchpoints|tp|web_qr)[:/]+(.+)$/i
      );
      touchpointId = prefixedId ? prefixedId[1] : null;
    }
    if (!touchpointId && /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$/.test(raw)) {
      touchpointId = raw;
    }
    if (typeof touchpointId !== "string" && touchpointId !== null) return null;
    if (touchpointId) {
      try {
        touchpointId = decodeURIComponent(touchpointId.trim());
      } catch (error) {
        return null;
      }
      touchpointId = touchpointId.trim();
      if (!/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$/.test(touchpointId)) return null;
    }
    if (!touchpointId && !normalizedUrl) return null;

    var latitude = Number(source.latitude);
    var longitude = Number(source.longitude);
    return {
      touchpointId: touchpointId,
      fullUrl: normalizedUrl,
      url: normalizedUrl,
      name: typeof source.name === "string" ? source.name.trim() : "",
      description: typeof source.description === "string" ? source.description.trim() : "",
      type: typeof source.type === "string" ? source.type.trim() : "web_qr",
      keywords: Array.isArray(source.keywords)
        ? source.keywords.filter(function (keyword) {
            return typeof keyword === "string" && keyword.trim();
          }).map(function (keyword) {
            return keyword.trim();
          })
        : [],
      latitude: Number.isFinite(latitude) && latitude >= -90 && latitude <= 90
        ? latitude
        : null,
      longitude: Number.isFinite(longitude) && longitude >= -180 && longitude <= 180
        ? longitude
        : null
    };
  }

  function stopQrScanner() {
    if (!qrScanner || !qrScannerRunning) return Promise.resolve();
    qrScannerRunning = false;
    return qrScanner.stop().then(function () {
      return qrScanner.clear();
    }).catch(function (error) {
      console.warn("Unable to stop QR scanner cleanly.", error);
    }).then(function () {
      $("#leobot_qr_reader").addClass("d-none").empty();
      $("#leobot_qr_start_button").removeClass("d-none");
      $("#leobot_qr_stop_button").addClass("d-none");
    });
  }

  function persistQrTouchpoint(touchpoint) {
    var visitorId = window.currentUserProfile && window.currentUserProfile.visitorId;
    if (!visitorId) {
      setQrStatus("qrScannerNoVisitor", "text-danger");
      return;
    }
    lscache.set(touchpointCacheKey(visitorId), touchpoint.touchpointId);
    $.getJSON(BASE_URL_GET_VISITOR_INFO, {
      visitor_id: visitorId,
      touchpoint_id: touchpoint.touchpointId,
      touchpoint_name: touchpoint.name || "QR touchpoint",
      touchpoint_description: touchpoint.description || "",
      touchpoint_type: touchpoint.type || "web_qr",
      touchpoint_full_url:
        touchpoint.fullUrl || touchpoint.full_url || touchpoint.url || "",
      _: Date.now()
    }).fail(function () {
      console.warn("Unable to persist scanned touchpoint.");
      setQrStatus("qrScannerError", "text-danger");
    });
  }

  function activateQrTouchpoint(touchpoint) {
    pendingQrTouchpoint = touchpoint;
    pendingQrTouchpoint.isLoaded = true;
    currentUserProfile.touchpointId = touchpoint.touchpointId;
    currentUserProfile.touchpointLocked = true;
    currentUserProfile.touchpointName = touchpoint.name || "QR touchpoint";
    currentUserProfile.touchpointDescription = touchpoint.description;
    currentUserProfile.touchpointType = touchpoint.type;
    currentUserProfile.touchpointFullUrl =
      touchpoint.fullUrl || touchpoint.full_url || touchpoint.url || "";
    currentUserProfile.touchpointKeywords = touchpoint.keywords;
    currentUserProfile.latitude = touchpoint.latitude;
    currentUserProfile.longitude = touchpoint.longitude;
    if (touchpoint.fullUrl || touchpoint.full_url || touchpoint.url) {
      lscache.set(touchpointCacheKey(currentUserProfile.visitorId), touchpoint.touchpointId);
    } else {
      persistQrTouchpoint(touchpoint);
    }

    var label = touchpoint.name
      ? touchpoint.name + " (" + touchpoint.touchpointId + ")"
      : touchpoint.touchpointId;
    $("#leobot_qr_result_name").text(label);
    $("#leobot_qr_result_url").text(
      touchpoint.fullUrl || touchpoint.full_url || touchpoint.url || ""
    );
    $("#leobot_qr_result").removeClass("d-none");
    setQrStatus("qrScannerSelected", "text-success");
  }

  function queueQrTouchpoint(touchpoint) {
    pendingQrTouchpoint = touchpoint;
    var label = touchpoint.name
      || touchpoint.fullUrl
      || touchpoint.url
      || touchpoint.touchpointId;
    $("#leobot_qr_result_name").text(label);
    $("#leobot_qr_result_url").text(
      touchpoint.fullUrl || touchpoint.full_url || touchpoint.url || ""
    );
    $("#leobot_qr_result").removeClass("d-none");
    setQrStatus("qrScannerSelected", "text-success");
    stopQrScanner();
  }

  function applyQrTouchpoint(touchpoint, onReady) {
    var visitorId = window.currentUserProfile && window.currentUserProfile.visitorId;
    if (!visitorId) {
      setQrStatus("qrScannerNoVisitor", "text-danger");
      return false;
    }
    stopQrScanner();
    if (!touchpoint.fullUrl && !touchpoint.full_url && !touchpoint.url) {
      if (!touchpoint.touchpointId) {
        setQrStatus("qrScannerInvalid", "text-danger");
        return false;
      }
      activateQrTouchpoint(touchpoint);
      if (typeof onReady === "function") onReady(touchpoint);
      return true;
    }

    qrTouchpointLoading = true;
    $("#leobot_qr_chat_button").prop("disabled", true);
    setQrStatus("qrScannerFetching");
    $.ajax({
      url: BASE_URL_QR_TOUCHPOINT,
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify({
        visitor_id: visitorId,
        full_url: touchpoint.fullUrl || touchpoint.full_url || touchpoint.url,
        touchpoint_id: touchpoint.touchpointId || null
      })
    }).done(function (data) {
      if (!data || typeof data.touchpoint_id !== "string") {
        qrTouchpointLoading = false;
        $("#leobot_qr_chat_button").prop("disabled", false);
        setQrStatus("qrScannerFetchError", "text-danger");
        return;
      }
      touchpoint.touchpointId = data.touchpoint_id;
      touchpoint.name = data.name || touchpoint.name;
      touchpoint.description = data.description || touchpoint.description;
      touchpoint.type = data.type || "web_qr";
      touchpoint.fullUrl =
        data.full_url || data.url || touchpoint.fullUrl || touchpoint.url;
      touchpoint.keywords = Array.isArray(data.keywords)
        ? data.keywords
        : touchpoint.keywords;
      activateQrTouchpoint(touchpoint);
      qrTouchpointLoading = false;
      $("#leobot_qr_chat_button").prop("disabled", false);
      if (typeof onReady === "function") onReady(touchpoint);
    }).fail(function (jqXHR) {
      console.warn("Unable to crawl QR URL.", jqXHR.status, jqXHR.responseText);
      qrTouchpointLoading = false;
      $("#leobot_qr_chat_button").prop("disabled", false);
      setQrStatus("qrScannerFetchError", "text-danger");
    });
    return true;
  }

  function startQrScanner() {
    if (qrScannerRunning) return;
    if (typeof window.Html5Qrcode !== "function") {
      setQrStatus("qrScannerUnavailable", "text-danger");
      return;
    }

    $("#leobot_qr_reader").removeClass("d-none");
    $("#leobot_qr_start_button").addClass("d-none");
    $("#leobot_qr_stop_button").removeClass("d-none");
    $("#leobot_qr_result").addClass("d-none");
    pendingQrTouchpoint = null;
    setQrStatus("qrScannerStarting");
    qrScanner = new window.Html5Qrcode("leobot_qr_reader");
    qrScannerRunning = true;
    qrScanner.start(
      { facingMode: "environment" },
      { fps: 10, qrbox: { width: 220, height: 220 } },
      function (decodedText) {
        if (!qrScannerRunning || qrTouchpointLoading) return;
        var touchpoint = normalizeQrTouchpoint(decodedText);
        if (!touchpoint) {
          setQrStatus("qrScannerInvalid", "text-danger");
          return;
        }
        queueQrTouchpoint(touchpoint);
      },
      function () {}
    ).then(function () {
      setQrStatus("qrScannerScanning");
    }).catch(function (error) {
      console.warn("Unable to start QR scanner.", error);
      qrScannerRunning = false;
      $("#leobot_qr_reader").addClass("d-none").empty();
      $("#leobot_qr_start_button").removeClass("d-none");
      $("#leobot_qr_stop_button").addClass("d-none");
      setQrStatus("qrScannerError", "text-danger");
    });
  }

  function renderRecommendedActions(actions) {
    recommendedActions = Array.isArray(actions) ? actions : null;
    var $container = $("#leobot_quick_actions").empty();
    if (recommendedActions === null || recommendedActions.length === 0) {
      var messageKey = recommendedActions === null
        ? "loadingRecommendations"
        : "locationSuggestionsUnavailable";
      $("<p>", { class: "small text-secondary mb-0" })
        .text(window.leoUiText(messageKey))
        .appendTo($container);
      return;
    }

    recommendedActions.forEach(function (action) {
      if (!action || !action.label || !action.question) return;
      var label = action.label[window.LEO_UI_LANGUAGE];
      var question = action.question[window.LEO_UI_LANGUAGE];
      if (typeof label !== "string" || typeof question !== "string") return;

      var icon = typeof action.icon === "string" &&
        /^fa-[a-z0-9-]+$/.test(action.icon)
        ? action.icon
        : "fa-compass";
      var $button = $("<button>", {
        type: "button",
        class: "btn btn-light text-start rounded-3 leobot-quick-action"
      }).attr("data-question", question);
      $("<i>", {
        class: "fa " + icon + " text-primary me-2",
        "aria-hidden": "true"
      }).appendTo($button);
      $("<span>").text(label).appendTo($button);
      $button.appendTo($container);
    });

    if (!$container.find(".leobot-quick-action").length) {
      $("<p>", { class: "small text-secondary mb-0" })
        .text(window.leoUiText("locationSuggestionsUnavailable"))
        .appendTo($container);
    }
  }

  function applyLanguage(language) {
    window.LEO_UI_LANGUAGE = language === "en" ? "en" : "vi";
    try {
      window.localStorage.setItem("leobot_ui_language", window.LEO_UI_LANGUAGE);
    } catch (error) {
      console.warn("Unable to save LEO language preference.", error);
    }

    $("html").attr("lang", window.LEO_UI_LANGUAGE);
    $("#leobot_language").val(window.LEO_UI_LANGUAGE);
    $("#leobot_language").attr("aria-label", window.leoUiText("language"));
    $("#leobot_help_button")
      .attr("aria-label", window.leoUiText("help"))
      .attr("title", window.leoUiText("help"));
    $("[data-i18n]").each(function () {
      $(this).text(window.leoUiText($(this).data("i18n")));
    });
    renderRecommendedActions(recommendedActions);
    $("#LEO_ChatBot_Container .botui-actions-text-input")
      .attr("placeholder", window.leoUiText("chatPlaceholder"))
      .attr("aria-label", window.leoUiText("chatPlaceholder"));
    var submitLabel = window.LEO_UI_LANGUAGE === "en" ? "Send" : "Gửi";
    $("#LEO_ChatBot_Container .botui-actions-buttons-button")
      .attr("aria-label", submitLabel)
      .find("span")
      .text(submitLabel);

    var $greeting = $("#LEO_ChatBot_Container .leobot-answer .botui-message-content").first();
    if ($greeting.length && typeof getGreetingMessage === "function") {
      var profile = window.currentUserProfile;
      var touchpoint = profile && profile.touchpointLocked
        ? { name: profile.touchpointName }
        : null;
      $greeting.text(getGreetingMessage(
        profile ? profile.displayName : "",
        window.LEO_UI_LANGUAGE,
        touchpoint
      ));
    }
    updateLocationPanel(latestLocationDetail);
  }

  function updateLocationPanel(detail) {
    latestLocationDetail = detail || { state: "pending" };
    var $status = $("#leobot_location_status");
    var $button = $("#leobot_location_button");
    var $icon = $("#leobot_location_icon");
    var $badge = $("#leobot_chat_location_badge");
    var $coordinates = $("#leobot_location_coordinates");
    var $accuracy = $("#leobot_location_accuracy");
    if (!$status.length || !$button.length || !$icon.length) return;

    var state = latestLocationDetail.state || "pending";
    var messages = {
      pending: ["secondary", "fa-circle-o-notch fa-spin", "checkingLocation"],
      requesting: ["info", "fa-location-arrow", "locationRequesting"],
      ready: ["success", "fa-check-circle", "locationReady"],
      denied: ["warning", "fa-info-circle", "locationDenied"],
      unavailable: ["warning", "fa-exclamation-triangle", "locationUnavailable"],
      error: ["danger", "fa-exclamation-circle", "locationError"]
    };
    var message = messages[state] || messages.error;
    $status.attr("class", "alert alert-" + message[0] + " small mb-3")
      .empty()
      .append($("<i>", {
        class: "fa " + message[1] + " me-2",
        "aria-hidden": "true"
      }))
      .append($("<span>").text(window.leoUiText(message[2])));

    $icon.attr("class", "leobot-location-icon is-" + state);
    $button.prop("disabled", state === "requesting").empty();
    if (state === "requesting") {
      $("<span>", {
        class: "spinner-border spinner-border-sm me-2",
        "aria-hidden": "true"
      }).appendTo($button);
      $button.append(document.createTextNode(window.leoUiText("gettingLocation")));
    } else {
      $("<i>", {
        class: "fa fa-crosshairs me-2",
        "aria-hidden": "true"
      }).appendTo($button);
      $button.append(document.createTextNode(
        window.leoUiText(state === "ready" ? "refreshLocation" : "useMyLocation")
      ));
    }

    if (state === "ready" && latestLocationDetail.data) {
      var data = latestLocationDetail.data;
      var latitude = Number(data.latitude);
      var longitude = Number(data.longitude);
      if (Number.isFinite(latitude) && Number.isFinite(longitude)) {
        $coordinates.text(latitude.toFixed(5) + ", " + longitude.toFixed(5));
      }
      $accuracy.text(data.accuracy
        ? "±" + Math.round(data.accuracy) + " m"
        : window.leoUiText("available"));
      renderRecommendedActions(
        Array.isArray(data.recommended_actions) ? data.recommended_actions : []
      );
      $badge.attr(
        "class",
        "badge rounded-pill text-bg-success-subtle text-success-emphasis d-none d-sm-inline-flex align-items-center gap-1"
      ).empty()
        .append($("<i>", { class: "fa fa-location-arrow", "aria-hidden": "true" }))
        .append(document.createTextNode(window.leoUiText("locationReadyBadge")));
    } else if (state !== "ready") {
      if (["denied", "unavailable", "error"].indexOf(state) !== -1) {
        renderRecommendedActions([]);
      }
      $badge.attr(
        "class",
        "badge rounded-pill text-bg-light text-secondary d-none d-sm-inline-flex align-items-center gap-1"
      ).empty()
        .append($("<i>", { class: "fa fa-location-arrow", "aria-hidden": "true" }))
        .append(document.createTextNode(window.leoUiText("locationOff")));
    }
  }

  function scrollToChatOnMobile() {
    var isMobile =
      typeof window.matchMedia === "function"
        ? window.matchMedia("(max-width: 991.98px)").matches
        : window.innerWidth <= 991;
    if (!isMobile) return;

    var chat = document.getElementById("LEO_ChatBot_Container");
    if (!chat || chat.getClientRects().length === 0) {
      chat = document.getElementById("leobot_chat_container");
    }
    if (!chat) return;

    var prefersReducedMotion =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    chat.scrollIntoView({
      behavior: prefersReducedMotion ? "auto" : "smooth",
      block: "start",
    });
  }

  function isMobileViewport() {
    return typeof window.matchMedia === "function"
      ? window.matchMedia("(max-width: 991.98px)").matches
      : window.innerWidth <= 991;
  }

  function updateMobileChatViewport() {
    if (!isMobileViewport()) return;
    var viewport = window.visualViewport;
    var visibleHeight = viewport && Number.isFinite(viewport.height)
      ? viewport.height
      : window.innerHeight;
    if (!Number.isFinite(visibleHeight) || visibleHeight <= 0) return;

    var keyboardOpen =
      Number.isFinite(window.innerHeight) &&
      visibleHeight < window.innerHeight * 0.78;
    var chatHeight = keyboardOpen
      ? Math.max(300, visibleHeight - 24)
      : Math.min(560, visibleHeight * 0.64);
    document.documentElement.style.setProperty(
      "--leobot-chat-height",
      Math.round(chatHeight) + "px"
    );

    if (keyboardOpen) {
      window.setTimeout(function () {
        var input = document.querySelector
          ? document.querySelector(
              "#LEO_ChatBot_Container .botui-actions-text-input"
            )
          : null;
        if (input && input.scrollIntoView) {
          input.scrollIntoView({ block: "nearest" });
        }
      }, 50);
    }
  }

  function startChatbot() {
    var $loader = $("#LEO_ChatBot_Container_Loader");
    var $chat = $("#LEO_ChatBot_Container");
    var startupComplete = false;

    function startChat(visitorId) {
      if (startupComplete) return;
      startupComplete = true;
      window.clearTimeout(startupFallback);
      if (typeof window.startLeoChatBot === "function") {
        window.startLeoChatBot(visitorId);
        return;
      }

      $loader.removeClass("d-flex").addClass("d-none").hide();
      var $error = $("<div>", {
        class: "alert alert-danger m-3",
        role: "alert"
      }).append(
        $("<i>", {
          class: "fa fa-exclamation-circle me-2",
          "aria-hidden": "true"
        }),
        document.createTextNode(window.leoUiText("assistantLoadError"))
      );
      $chat.stop(true, true).show().css("opacity", "1").empty().append($error);
    }

    var startupFallback = window.setTimeout(function () {
      startChat();
    }, 8000);
    var observerScript = window.location.protocol + "//" + HOSTNAME
      + "/resources/js/leocdp.observer.js";
    var allowedHost = ["leobot.leocdp.com", "leobot.example.com"]
      .indexOf(window.location.hostname) !== -1;

    if (CDP_TRACKING && allowedHost) {
      $.getScript(observerScript)
        .done(function () {
          startChat();
        })
        .fail(function () {
          console.warn("LEO observer failed to load; starting chatbot without tracking.");
          startChat();
        });
    } else {
      startChat();
    }
  }

  $(function () {
    $(document).on("leo:location", function (event, detail) {
      updateLocationPanel(detail || {});
    });
    $("#leobot_location_button").on("click", function () {
      if (
        typeof window.requestUserGeolocation === "function"
        && window.currentUserProfile
      ) {
        window.requestUserGeolocation(window.currentUserProfile.visitorId, true);
      }
    });
    $("#leobot_qr_start_button").on("click", startQrScanner);
    $("#leobot_qr_stop_button").on("click", function () {
      stopQrScanner();
    });
    $("#leobot_qr_chat_button").on("click", function () {
      if (!pendingQrTouchpoint || qrTouchpointLoading) return;
      var visitorId = window.currentUserProfile &&
        window.currentUserProfile.visitorId;
      var startTouchpointChat = function (touchpoint) {
        if (!window.leoBotStarted &&
            typeof window.startLeoChatBot === "function") {
          window.startLeoChatBot(visitorId);
        }
        if (typeof window.reloadLeoChatBotSession === "function") {
          window.reloadLeoChatBotSession(touchpoint);
        }
        scrollToChatOnMobile();
        if (!isMobileViewport()) {
          window.setTimeout(function () {
            $("#LEO_ChatBot_Container .botui-actions-text-input")
              .last()
              .trigger("focus");
          }, 400);
        }
      };
      if (pendingQrTouchpoint.isLoaded) {
        startTouchpointChat(pendingQrTouchpoint);
        return;
      }
      if (!applyQrTouchpoint(pendingQrTouchpoint, startTouchpointChat)) {
        return;
      }
    });
    $("#leobot_language").on("change", function () {
      applyLanguage($(this).val());
    });
    $("#leobot_quick_actions").on("click", ".leobot-quick-action", function () {
      scrollToChatOnMobile();
      var question = $(this).attr("data-question");
      if (question && typeof window.sendQuestionToLeoAI === "function") {
        window.sendQuestionToLeoAI("ask", question);
        window.getBotUI().message.human({ content: question });
      }
    });

    applyLanguage(window.LEO_UI_LANGUAGE);
    updateMobileChatViewport();
    if (typeof window.addEventListener === "function") {
      window.addEventListener("resize", updateMobileChatViewport);
      window.addEventListener("orientationchange", updateMobileChatViewport);
    }
    if (
      window.visualViewport &&
      typeof window.visualViewport.addEventListener === "function"
    ) {
      window.visualViewport.addEventListener(
        "resize",
        updateMobileChatViewport
      );
    }
    startChatbot();
    if (typeof window.startGoogleOneTapPrompt === "function") {
      window.startGoogleOneTapPrompt();
    }
  });
})(jQuery, window, document);