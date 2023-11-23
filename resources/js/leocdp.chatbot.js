var currentUserProfile = { visitorId: "", displayName: "good friend" };

window.leoBotUI = false;
window.leoBotContext = false;
function getBotUI() {
  if (window.leoBotUI === false) {
    window.leoBotUI = new BotUI("LEO_ChatBot_Container");
  }
  return window.leoBotUI;
}

function initLeoChatBot(context, okCallback) {
  window.leoBotContext = context;
  window.leoBotUI = new BotUI("LEO_ChatBot_Container");
  showLeoChatBot();
  if (typeof okCallback === "function") {
    okCallback();
  }
}

var showLeoChatBot = function () {
  var msg =
    "Hi " + currentUserProfile.displayName + ", you may ask me for anything";
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
        placeholder: "Give me a question",
      },
    })
    .then(function (res) {
      sendQuestionToLeoAI("ask", res.value);
    });
};

var leoBotShowAnswer = function (answerInHtml) {
  getBotUI()
    .message.add({
      human: false,
      cssClass: "leobot-answer",
      content: answerInHtml,
      type: "html",
    })
    .then(function () {
      $("div.botui-message").find("a").attr("target", "_blank");
      var delay = answerInHtml.length > 120 ? 6000 : 2000;
      leoBotPromptQuestion(delay);
    });
};

var leoBotShowError = function (error) {
  getBotUI()
    .message.add({
      human: false,
      cssClass: "leobot-answer",
      content: error,
      type: "html",
    })
    .then(function () {
      // skip
    });
};

var askTheEmailOfUser = function(name){
  getBotUI()
  .action.text({
    delay:0,
    action: {
      icon: "envelope-o",
      cssClass: "leobot-question-input",
      value: "", 
      placeholder: "Input your email here",
    },
  })
  .then(function (res) {
      var email = res.value;
      console.log(name, email)
      var profileData = {'loginProvider': "leochatbot", 'firstName': name, 'email': email}
      LeoObserverProxy.updateProfileBySession(profileData);
      setTimeout(function(){
        location.reload(true)
      },5000)
  });
}

var askTheNameOfUser = function(){
  getBotUI()
  .action.text({
    delay:0,
    action: {
      icon: "user-circle-o",
      cssClass: "leobot-question-input",
      value: "", 
      placeholder: "Input your name here",
    },
  })
  .then(function (res) {
    askTheEmailOfUser(res.value);
  });
}

var askForContactInfo = function (visitor_id) {
  var msg = 'Our system need your name and your email to register new user';
  getBotUI()
    .message.add({
      human: false,
      cssClass: "leobot-answer",
      content: msg,
      type: "html",
    }).then(askTheNameOfUser);
};

var sendQuestionToLeoAI = function (context, question) {
  if (question.length > 1 && question !== "exit") {
    var processAnswer = function (answer) {
      if ("ask" === context) {
        leoBotShowAnswer(answer);
      }
      // save event into LEO CDP
      if (typeof window.LeoObserver === "object") {
        var encodedAnswer = encodeURIComponent(answer.slice(0, 1000));
        var eventData = { question: question, answer: encodedAnswer };
        window.LeoObserver.recordEventAskQuestion(eventData);
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
        } 
        else if (error_code === 404) {
          askForContactInfo();
        } 
        else {
          leoBotShowError(answer);
        }
      };

      var payload = { prompt: question, question: question };
      payload["visitor_id"] = currentUserProfile.visitorId;
      payload["answer_in_language"] = $("#leobot_answer_in_language").val();
      payload["answer_in_format"] = "html";
      payload["context"] = "leobotweb";
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
  $("#LEO_ChatBot_Container, #leobot_answer_in_language").show();

  initLeoChatBot("website_leobot_" + visitorId);
};
