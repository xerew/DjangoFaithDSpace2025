function include(file) {
    const script = document.createElement('script');
    script.src = file;
    script.type = 'text/javascript';
    script.defer = true;

    document.getElementsByTagName('head').item(0).appendChild(script);
}

/* include all the components js file */

include('/static/chatbot_static/js/components/chat.js');
include('/static/chatbot_static/js/constants.js');
include('/static/chatbot_static/js/components/cardsCarousel.js');
include('/static/chatbot_static/js/components/botTyping.js');
include('/static/chatbot_static/js/components/charts.js');
include('/static/chatbot_static/js/components/collapsible.js');
include('/static/chatbot_static/js/components/dropDown.js');
include('/static/chatbot_static/js/components/location.js');
include('/static/chatbot_static/js/components/pdfAttachment.js');
include('/static/chatbot_static/js/components/quickReplies.js');
include('/static/chatbot_static/js/components/suggestionButtons.js');
