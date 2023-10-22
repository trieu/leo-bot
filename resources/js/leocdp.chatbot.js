var currentUserProfile = {"userLogin":"demo", "displayName": "good friend"}		

window.leoBotUI = false;
window.leoBotContext = false;
function getBotUI(){
	if(window.leoBotUI === false){
		window.leoBotUI = new BotUI('LEO_ChatBot_Container');	
	}
	return window.leoBotUI;
}

function initLeoChatBot(context, okCallback) {
	window.leoBotContext = context;
	window.leoBotUI = new BotUI('LEO_ChatBot_Container');
	showLeoChatBot();
	if(typeof okCallback === 'function'){
		okCallback();
	}
}

var showLeoChatBot = function() {
	var msg = 'Hi ' + currentUserProfile.displayName + ', you may ask me for anything';
	var msgObj = {content:msg, cssClass: 'leobot-answer'};
	getBotUI().message.removeAll();
	getBotUI().message.bot(msgObj).then(leoBotPromptQuestion);
}

var leoBotPromptQuestion = function(delay) {
	getBotUI().action.text({
		delay: typeof delay === 'number' ? delay : 800,
		action: {
			icon: 'question-circle',
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
	}).then(function(){
		$('div.botui-message').find('a').attr('target', '_blank');
		var delay = answerInHtml.length > 120 ? 6000 : 2000;
		leoBotPromptQuestion(delay);
	});
}

var sendQuestionToLeoAI = function(context, question) {
	if (question.length > 1 && question !== "exit") {

		var processAnswer = function(answer) {
			if ('ask' === context) {
				leoBotShowAnswer(answer);
			}
			// save event into LEO CDP
			if(typeof window.LeoObserver === 'object') {
				var encodedAnswer = encodeURIComponent(answer.slice(0, 1000));
				var eventData = {"question":question,"answer":encodedAnswer};
				window.LeoObserver.recordEventAskQuestion(eventData);
			}
		}
		
		var callServer = function (index) {
			var serverCallback = function(data) {
				getBotUI().message.remove(index);
				if (typeof data.answer === 'string') {
					var answer = data.answer;
					processAnswer(answer);
				}
				else if (data.error) {
					alert(data.error)				
				}
				else {
					alert('LEO BOT is getting a system error !')
				}
			};
	
			var payload = { 'prompt': question, 'question': question };
			payload['usersession'] = getUserSession();
			payload['userlogin'] = currentUserProfile.userLogin;
			payload['answer_in_language'] = $('#leobot_answer_in_language').val()
			payload['answer_in_format'] = 'html';
			payload['context'] = 'leobotweb';
			callPostApi(BASE_URL_LEOBOT, payload, serverCallback);
		}
		showChatBotLoader().then(callServer);
	}
}

var showChatBotLoader = function(){
	return getBotUI().message.add({loading: true, content:''});
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