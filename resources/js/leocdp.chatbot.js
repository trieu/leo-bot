var currentUserProfile = { visitorId: "", displayName: "friend" };


function loadChatSessionWithProfile() {
  let userProfile = {};
  const hashData = location.hash.substring(1);
  try {
    userProfile = hashData ? JSON.parse(decodeURIComponent(hashData)) : {};
    console.log("Loaded user profile from hash:", userProfile);
  } catch (e) {
    console.warn("Invalid profile data in hash:", e);
    userProfile = {};
  }

  // === You can now use `userProfile` to personalize the chatbot ===
  // Example:
  // if (userProfile.name) greetUser(userProfile.name);
}

// Call when location.hash changes (e.g., updated by parent page)
window.addEventListener("hashchange", loadChatSessionWithProfile);

window.leoBotUI = false;
window.leoBotContext = false;
function getBotUI() {
  if (window.leoBotUI === false) {
    window.leoBotUI = new BotUI("LEO_ChatBot_Container");
  }
  return window.leoBotUI;
}


function initLeoChatBot(context, visitorId, okCallback) {
  window.leoBotContext = context;

  loadChatSessionWithProfile()
  window.currentUserProfile.visitorId = visitorId;
  
  window.leoBotUI = new BotUI("LEO_ChatBot_Container");

  var url =
    BASE_URL_GET_VISITOR_INFO +
    "?visitor_id=" +
    visitorId +
    "&_=" +
    new Date().getTime();
  $.getJSON(url, function (data) {
    var error_code = data.error_code;
    var answer = data.answer;
    console.log(data);

    if (error_code === 0) {
      var name = currentUserProfile.displayName;
      name = answer.length > 0 ? answer : name;
      currentUserProfile.displayName = name;
      showLeoChatBot(currentUserProfile.displayName);
    } 
    else if (error_code === 404) {
      // askTheContactOfUser();
      currentUserProfile.displayName = '';
      showLeoChatBot(currentUserProfile.displayName);
    } 
    else {
      leoBotShowError(answer, leoBotPromptQuestion);
    }
  });

  if (typeof okCallback === "function") {
    okCallback();
  }
}

/**
 * Returns a greeting message in either English or Vietnamese.
 *
 * @param {string} displayName The name of the user to greet.
 * @param {string} language The language code ('en' for English, 'vi' for Vietnamese).
 * @returns {string} The formatted greeting message.
 */
function getGreetingMessage(displayName, language) {
  let msg;
  switch (language) {
    case 'vi':
      msg = "Chào " + displayName + ", bạn có thể hỏi tôi bất cứ điều gì";
      break;
    case 'en':
    default: // Default to English if the language is not recognized
      msg = "Hi " + displayName + ", you may ask me for anything";
      break;
  }
  return msg;
}

var showLeoChatBot = function (displayName) {
  var msg = getGreetingMessage(displayName, "vi");
  var msgObj = { content: msg, cssClass: "leobot-answer" };
  getBotUI().message.removeAll();
  getBotUI().message.bot(msgObj).then(leoBotPromptQuestion);
};

var leoBotPromptQuestion = function (delay) {
  getBotUI()
    .action.text({
      delay: typeof delay === "number" ? delay : 800,
      action: {
        icon: "question-circle",
        cssClass: "leobot-question-input",
        value: "", // show the prevous answer if any
        placeholder: "..................",
      },
    })
    .then(function (res) {
      sendQuestionToLeoAI("ask", res.value);
    });
};

var leoBotShowAnswer = function (answerInHtml, providedDelay) {
  getBotUI()
    .message.add({
      human: false,
      cssClass: "leobot-answer",
      content: answerInHtml,
      type: "html",
    })
    .then(function () {
      // format all href nodes in answer
      $("div.botui-message")
        .find("a")
        .each(function () {
          $(this).attr("target", "_blank");
          var href = $(this).attr("href");
          if (href.indexOf("google.com") < 0) {
            href =
              "https://www.google.com/search?q=" +
              encodeURIComponent($(this).text());
          }
          $(this).attr("href", href);
        });

      let delay;
      if (typeof providedDelay === "number") {
        delay = providedDelay;
      } else if (answerInHtml.length > 200) {
        delay = 3000;
      } else {
        delay = 1500;
      }
      leoBotPromptQuestion(delay);
    });
};

var leoBotShowError = function (error, nextAction) {
  getBotUI()
    .message.add({
      human: false,
      cssClass: "leobot-error",
      content: error,
      type: "html",
    })
    .then(nextAction || function () {});
};

function isEmailValid(email) {
  const regex =
    /^(([^<>()[\]\\.,;:\s@\"]+(\.[^<>()[\]\\.,;:\s@\"]+)*)|(\".+\"))@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\])|(([a-zA-Z\-0-9]+\.)+[a-zA-Z]{2,}))$/;
  return regex.test(email);
}

var askTheEmailOfUser = function (name) {
  getBotUI()
    .action.text({
      delay: 0,
      action: {
        icon: "envelope-o",
        cssClass: "leobot-question-input",
        value: "",
        placeholder: "Email của bạn",
      },
    })
    .then(function (res) {
      var email = res.value;
      if (isEmailValid(email)) {
        console.log(name, email);
        var profileData = {
          loginProvider: "leochatbot",
          firstName: name,
          email: email,
        };
        if(window.CDP_TRACKING === true) {
          LeoObserverProxy.updateProfileBySession(profileData);
        }

        setTimeout(function () {
          location.reload(true);
        }, 3000);

        var s = "Chào " +  name + ", hệ thống đang đăng ký thông tin cho bạn ...";
        leoBotShowAnswer(s, 6000);// delay 10 seconds to make chatbot do not show input box
      } else {
        leoBotShowError(email + " không là email hợp lệ", function () {
          askTheEmailOfUser(name);
        });
      }
    });
};

var askTheNameOfUser = function () {
  getBotUI()
    .action.text({
      delay: 0,
      action: {
        icon: "user-circle-o",
        cssClass: "leobot-question-input",
        value: "",
        placeholder: "Tên bạn",
      },
    })
    .then(function (res) {
      askTheEmailOfUser(res.value);
    });
};

var askTheContactOfUser = function () {
  var msg = "Chào bạn, vui lòng nhập tên và email để  cần hỗ trỡ";
  getBotUI()
    .message.add({
      human: false,
      cssClass: "leobot-question",
      content: msg,
      type: "html",
    })
    .then(askTheNameOfUser);
};

var sendQuestionToLeoAI = function (context, question) {
  if (question.length > 1 && question !== "exit") {
    var processAnswer = function (answer) {
      if ("ask" === context) {
        leoBotShowAnswer(answer);
      }
      // save event into CDP
      if (typeof LeoObserver === "object" && CDP_TRACKING === true) {
        var sAnswer = answer.slice(0, 1000);
        var eventData = { question: question, answer: sAnswer };
        LeoObserver.recordEventAskQuestion(eventData);
      } else {
        console.log("SKIP LeoObserver.recordEventAskQuestion")
      }
    };

    var callServer = function (index) {
      var serverCallback = function (data) {
        getBotUI().message.remove(index);
        var error_code = data.error_code;
        var answer = data.answer;
        if (error_code === 0) {
          currentUserProfile.displayName = data.name;
          processAnswer(answer);
        } else if (error_code === 404) {
          // askTheContactOfUser();
          currentUserProfile.displayName = "";
          processAnswer(answer);
        } else {
          leoBotShowError(answer, leoBotPromptQuestion);
        }
      };

      var context = $('#LEO_ChatBot_Container').find('.botui-message-content').slice(-3)
              .map(function () {
                return $(this).text();
              }).get().join(' ; ');

      var payload = {};
      payload["context"] = context;
      payload["question"] = question;
      payload["visitor_id"] = currentUserProfile.visitorId;
      payload["answer_in_language"] = "Vietnamese";
      payload["answer_in_format"] = "html";
      
      callPostApi(BASE_URL_LEOBOT, payload, serverCallback);
    };
    showChatBotLoader().then(callServer);
  }
};

var showChatBotLoader = function () {
  return getBotUI().message.add({ loading: true, content: "" });
};

var callPostApi = function (urlStr, data, okCallback, errorCallback) {
  $.ajax({
    url: urlStr,
    crossDomain: true,
    data: JSON.stringify(data),
    contentType: "application/json",
    type: "POST",
    error: function (jqXHR, exception) {
      console.error("WE GET AN ERROR AT URL:" + urlStr);
      console.error(exception);
      if (typeof errorCallback === "function") {
        errorCallback();
      }
    },
  }).done(function (json) {
    okCallback(json);
    console.log("callPostApi", urlStr, data, json);
  });
};

var startLeoChatBot = function (visitorId) {
  currentUserProfile.visitorId = visitorId;
  $("#LEO_ChatBot_Container_Loader").hide();
  $("#LEO_ChatBot_Container").show();
  initLeoChatBot("leobot_website", visitorId);
};
