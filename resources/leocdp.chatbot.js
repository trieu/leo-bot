const baseLeoBotUrl = location.protocol + '//' + dnsDomainLeoBot;
var currentUserProfile = {"userLogin":"demo", "displayName": "good friend"}		

const BASE_URL_LEOBOT = baseLeoBotUrl + '/ask';
const IS_LEO_BOT_READY = dnsDomainLeoBot !== "";

window.leoBotUI = false;
function getBotUI(){
	if(window.leoBotUI === false){
		window.leoBotUI = new BotUI('LEO_ChatBot_Container');	
	}
	return window.leoBotUI;
}

function initLeoChatBot(context) {
	$('#leoChatBotDialog').modal({backdrop: 'static', keyboard: false});
	getBotUI().message.removeAll();
	showLeoChatBot();
}

var showLeoChatBot = function() {
	var msg = 'Hi ' + currentUserProfile.displayName + ', you may ask me for anything';
	var msgObj = {content:msg, cssClass: 'leobot-answer'};
	getBotUI().message.bot(msgObj).then(leoBotPromptQuestion);
}

var leoBotPromptQuestion = function() {
	getBotUI().action.text({
		delay: 500,
		action: {
			icon: 'question',
			cssClass: 'leobot-question-input',
			value: '', // show the prevous answer if any
			placeholder: 'Give me a question'
		}
	}).then(function(res) {
		sendQuestionToLeoAI('ask', res.value);	
	});
}

var leoBotShowAnswer = function(answerInHtml){
	getBotUI().message.add({ 
		human: false, 
		cssClass: 'leobot-answer',
		content: answerInHtml, 
		type: 'html' 
	});
	setTimeout(function() {
		$('div.botui-message').find('a').attr('target', '_blank');
	}, 1500);
}

var sendQuestionToLeoAI = function(context, content) {
	if (content.length > 1 && content !== "exit") {
		
		var callServer = function (index) {
			var serverCallback = function(data) {
				getBotUI().message.remove(index);
	
				if (typeof data.answer === 'string') {
					var answerInRaw = data.answer.trim().replace(/(?:\r\n|\r|\n)/g, '<br>');
					var answerInHtml = marked.parse(answerInRaw);
	
					if ('ask' === context) {
						leoBotShowAnswer(answerInHtml);
						// next question
						leoBotPromptQuestion()
					}
					
					// save event into LEO CDP
					if(typeof window.LeoObserver === 'object') {
						var eventData = {"question":content,"answer":answerInRaw};
						window.LeoObserver.recordEventAskQuestion(eventData);
					}
				}
				else if (data.error) {
					alert(data.error)				
				}
				else {
					alert('LEO BOT is getting a system error !')
				}
			};

			var lang = $('#leobot_answer_in_language').val()
			var prompt = content;
			var userLogin = currentUserProfile.userLogin;		
			var payload = { 'prompt': prompt, 'content': content, 'usersession': getUserSession(), 'userlogin': userLogin, 'answer_in_language': lang };
			callPostApi(BASE_URL_LEOBOT, payload, serverCallback);
		};

		getBotUI().message.add({loading: true, content:''}).then(callServer);
	}
}

var callPostApi = function (urlStr, data, okCallback, errorCallback) {
	$.ajax({
		url: urlStr,
		crossDomain: true,
		data: JSON.stringify(data),
		contentType: 'application/json',
		type: 'POST',
		error: function (jqXHR, exception) {
			console.error('WE GET AN ERROR AT URL:' + urlStr);
			console.error(exception);
			if (typeof errorCallback === 'function') {
				errorCallback();
			}
		}
	}).done(function (json) {
		okCallback(json);
		console.log("callPostApi", urlStr, data, json);
	});
}

var getUserSession = function(){
	// In Redis, need: hset demo userlogin demo
	return "demo";
}