// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
var userId         = null;
var scenarioId     = null;
var scenarioLang   = null;
var latestPendulumData    = null;
var latestExperimentLog   = null;

// ---------------------------------------------------------------------------
// Markdown → HTML converter
// ---------------------------------------------------------------------------
const converter = new showdown.Converter();

function scrollToBottomOfResults() {
  const el = document.getElementById('chats');
  el.scrollTop = el.scrollHeight;
}

// ---------------------------------------------------------------------------
// Render user message in the chat
// ---------------------------------------------------------------------------
function setUserResponse(message) {
  const html = `<img class="userAvatar" src="/static/chatbot_static/img/user_avatar.svg">
                <p class="userMsg">${message}</p>
                <div class="clearfix"></div>`;
  $(html).appendTo('.chats').show('slow');
  $('.usrInput').val('');
  scrollToBottomOfResults();
  showBotTyping();
  $('.suggestions').remove();
}

// ---------------------------------------------------------------------------
// Build a single bot bubble
// ---------------------------------------------------------------------------
function getBotResponse(html) {
  return `<img class="botAvatar" src="/static/chatbot_static/img/PlatoV2.png"/>
          <span class="botMsg">${html}</span>
          <div class="clearfix"></div>`;
}

// ---------------------------------------------------------------------------
// Render the full Rasa response array in the chat
// ---------------------------------------------------------------------------
function setBotResponse(response) {
  setTimeout(() => {
    hideBotTyping();

    if (!response || response.length < 1) {
      const fallback = `<img class="botAvatar" src="/static/chatbot_static/img/PlatoV2.png"/>
                        <p class="botMsg">I'm having some trouble right now. Please try again in a moment.</p>
                        <div class="clearfix"></div>`;
      $(fallback).appendTo('.chats').hide().fadeIn(1000);
      scrollToBottomOfResults();
      return;
    }

    for (let i = 0; i < response.length; i++) {
      const msg = response[i];

      // --- Text ---
      if (msg.text != null) {
        let html = converter.makeHtml(msg.text);
        html = html
          .replaceAll('<p>', '').replaceAll('</p>', '')
          .replaceAll('<strong>', '<b>').replaceAll('</strong>', '</b>');
        html = html.replace(/(?:\r\n|\r|\n)/g, '<br>');

        let botBubble;
        if (html.includes('<blockquote>') || html.includes('<ul') ||
            html.includes('<ol') || html.includes('<li') || html.includes('<h3')) {
          html = html.replaceAll('<br>', '');
          botBubble = getBotResponse(html);
        } else if (html.includes('<img')) {
          html = html.replaceAll('<img', '<img class="imgcard_mrkdwn" ');
          botBubble = getBotResponse(html);
        } else if (html.includes('<pre') || html.includes('<code>')) {
          botBubble = getBotResponse(html);
        } else {
          botBubble = `<img class="botAvatar" src="/static/chatbot_static/img/PlatoV2.png"/>
                       <p class="botMsg">${msg.text}</p>
                       <div class="clearfix"></div>`;
        }
        $(botBubble).appendTo('.chats').hide().fadeIn(1000);
      }

      // --- Image ---
      if (msg.image) {
        const imgHtml = `<div class="singleCard"><img class="imgcard" src="${msg.image}"></div>
                         <div class="clearfix">`;
        $(imgHtml).appendTo('.chats').hide().fadeIn(1000);
      }

      // --- Buttons ---
      if (msg.buttons && msg.buttons.length > 0) {
        addSuggestion(msg.buttons);
      }

      // --- Video attachment ---
      if (msg.attachment && msg.attachment.type === 'video') {
        const videoHtml = `<div class="video-container">
          <iframe src="${msg.attachment.payload.src}" frameborder="0" allowfullscreen></iframe>
        </div>`;
        $(videoHtml).appendTo('.chats').hide().fadeIn(1000);
      }

      // --- Custom payloads ---
      if (msg.custom) {
        const { payload } = msg.custom;
        if (payload === 'quickReplies')    { showQuickReplies(msg.custom.data); continue; }
        if (payload === 'pdf_attachment')  { renderPdfAttachment(msg);          continue; }
        if (payload === 'dropDown')        { renderDropDwon(msg.custom.data);   continue; }
        if (payload === 'location')        { $('#userInput').prop('disabled', true); getLocation(); continue; }
        if (payload === 'cardsCarousel')   { showCardsCarousel(msg.custom.data); continue; }
        if (payload === 'chart') {
          const { title, labels, backgroundColor, chartsData, chartType, displayLegend } = msg.custom.data;
          createChart(title, labels, backgroundColor, chartsData, chartType, displayLegend);
          $(document).on('click', '#expand', () =>
            createChartinModal(title, labels, backgroundColor, chartsData, chartType, displayLegend)
          );
          continue;
        }
        if (payload === 'collapsible') { createCollapsible(msg.custom.data); }
      }
    }

    scrollToBottomOfResults();
    $('.usrInput').focus();
  }, 500);
}

// ---------------------------------------------------------------------------
// postMessage listeners — unified, with origin check
// ---------------------------------------------------------------------------
window.addEventListener('message', function(event) {
  if (event.origin !== window.location.origin) return;

  const { type, data: evData } = event.data;

  if (type === 'initData') {
    userId      = event.data.userId;
    scenarioId  = event.data.scenarioId;
    scenarioLang = event.data.scenarioLang;
    window.dispatchEvent(new CustomEvent('userDataReceived', {
      detail: { userId, scenarioId, scenarioLang }
    }));
  }

  if (type === 'pendulumData') {
    latestPendulumData  = event.data.pendulumData;
    latestExperimentLog = event.data.experimentData || null;
  }

  if (type === 'sendReadyMessage') {
    const displayText = event.data.displayText || event.data.message;
    const rasaMsg     = event.data.rasaMessage  || event.data.message;
    setUserResponse(displayText);
    send(rasaMsg);
  }

  if (type === 'sendMessage') {
    const msg = event.data.message;
    setUserResponse(msg);
    send(msg);
  }
});

// ---------------------------------------------------------------------------
// Send message to Rasa (fetch-based, no artificial delay)
// ---------------------------------------------------------------------------
async function send(message) {
  try {
    const response = await fetch(rasa_server_url, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        message,
        sender:   sender_id,
        metadata: {
          user_id:         userId,
          scenario_id:     scenarioId,
          scenario_lang:   scenarioLang,
          pendulum_data:   latestPendulumData,
          experiment_data: latestExperimentLog,
        },
      }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const botResponse = await response.json();

    // Dispatch activity ID events before rendering (so parent page updates first)
    botResponse.forEach((msg) => {
      if (msg.custom?.activity_id !== undefined) {
        window.dispatchEvent(new CustomEvent('activityIdReceived', {
          detail: { activityId: msg.custom.activity_id }
        }));
      }
      if (msg.custom?.scenario_ended) {
        window.dispatchEvent(new CustomEvent('scenarioEnded'));
      }
      if (msg.buttons) {
        msg.buttons.forEach((btn) => {
          if (btn.activity_id !== undefined) {
            window.dispatchEvent(new CustomEvent('activityIdReceived', {
              detail: { activityId: btn.activity_id }
            }));
          }
        });
      }
    });

    if (message === '/restart') {
      $('#userInput').prop('disabled', false);
      return;
    }

    setBotResponse(botResponse);

  } catch (err) {
    setBotResponse([]);
    console.error('Rasa error:', err);
  }
}

// ---------------------------------------------------------------------------
// Restart conversation
// ---------------------------------------------------------------------------
function restartConversation() {
  $('#userInput').prop('disabled', true);
  $('.collapsible').remove();
  if (typeof chatChart  !== 'undefined') chatChart.destroy();
  if (typeof modalChart !== 'undefined') modalChart.destroy();
  $('.chart-container').remove();
  $('.chats').html('');
  $('.usrInput').val('');
  send('/restart');
}
$('#restart').click(() => restartConversation());

// ---------------------------------------------------------------------------
// Enter key / send button
// ---------------------------------------------------------------------------
function _clearBeforeSend() {
  $('.collapsible, .dropDownMsg, .chart-container, #paginated_cards, .suggestions, .quickReplies').remove();
  if (typeof chatChart  !== 'undefined') chatChart.destroy();
  if (typeof modalChart !== 'undefined') modalChart.destroy();
}

$('.usrInput').on('keyup keypress', (e) => {
  const keyCode = e.keyCode || e.which;
  if (keyCode !== 13) return true;

  const text = $('.usrInput').val().trim();
  if (!text) { e.preventDefault(); return false; }

  _clearBeforeSend();
  $('.usrInput').blur();
  setUserResponse(text);
  send(text);
  e.preventDefault();
  return false;
});

$('#sendButton').on('click', (e) => {
  const text = $('.usrInput').val().trim();
  if (!text) { e.preventDefault(); return false; }

  _clearBeforeSend();
  $('.usrInput').blur();
  setUserResponse(text);
  send(text);
  e.preventDefault();
  return false;
});

// ---------------------------------------------------------------------------
// Legacy trigger functions (kept for compatibility, not used in normal flow)
// ---------------------------------------------------------------------------
function actionTrigger() {
  console.warn('actionTrigger() is for Rasa 1.x and is no longer used.');
}
function customActionTrigger() {
  console.warn('customActionTrigger() is for Rasa 2.x and is no longer used.');
}
