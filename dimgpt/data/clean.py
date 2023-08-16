from unidecode import unidecode

from dimgpt.settings import *


AUTHORIZED_UNICODE = set(
	'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' \
	'0123456789' \
	' !"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~' \
	'ÀàÂâÄäÇçÉéÈèÊêËëÎîÏïÔôÖöÙùÛûÜüÆæŒœ' \
	'€£¥•·²³≠±×÷√π' \
	'😀😃😄😁😆😅😂🤣🥲🥹😊😇🙂🙃😉😌😍🥰😘😗😙😚😋😛😝😜🤪🤨🧐🤓😎🥸🤩🥳😏😒😞😔😟😕🙁😣😖😫😩🥺😢😭😤😠😡🤬🤯😳🥵🥶😱😨😰😥😓🫣🤗🫡🤔🫢🤭🤫🤥😶😐😑😬🫠🙄😯😦😧😮😲🥱😴🤤😪😵🫥🤐🥴🤢🤮🤧😷🤒🤕🤑🤠😈👿👹👺🤡💩👻💀👽👾🤖🎃😺😸😹😻😼😽🙀😿😾' \
	'👋🤚🖐✋🖖👌🤌🤏🤞🫰🤟🤘🤙🫵🫱🫲🫳🫴👈👉👆🖕👇👍👎✊👊🤛🤜👏🫶🙌👐🤲🤝🙏💅🤳💪🦾🦵🦿🦶👣👂🦻👃🫀🫁🧠🦷🦴👀👁👅👄🫦💋🩸' \
	'👶👧🧒👦👩🧑👨👱🧔👵🧓👴👲👳🧕👮👷💂👰🤵👸🫅🤴🥷🦸🦹🤶🎅🧙🧝🧛🧟🧞🧜🧚🧌👼🤰🤱🙇💁🙅🙆🙋🧏🤦🤷🙎🙍💇💆🧖💅🤳💃🕺👯🕴🚶🧎🏃🧍👭👬👫💑💏👪🗣👤👥🫂' \
	'🧳🌂🧵🪡🪢🧶👓🕶🥽🥼🦺👔👕👖🧣🧤🧥🧦👗👘🥻🩴🩱🩲🩳👙👚👛👜👝🎒👞👟🥾🥿👠👡🩰👢👑👒🎩🎓🧢⛑🪖💄💍💼' \
	'🐶🐱🐭🐹🐰🦊🐻🐼🐨🐯🦁🐮🐷🐽🐸🐵🙈🙉🙊🐒🐔🐧🐦🐤🐣🐥🦆🦅🦉🦇🐺🐗🐴🦄🐝🪱🐛🦋🐌🐞🐜🪰🪲🪳🦟🦗🕷🕸🦂🐢🐍🦎🦖🦕🐙🦑🦐🦞🦀🪸🐡🐠🐟🐬🐳🐋🦈🐊🐅🐆🦓🦍🦧🦣🐘🦛🦏🐪🐫🦒🦘🦬🐃🐂🐄🐎🐖🐏🐑🦙🐐🦌🐕🐩🦮🐈🪶🐓🦃🦤🦚🦜🦢🦩🕊🐇🦝🦨🦡🦫🦦🦥🐁🐀🐿🦔🐾🐉🐲🌵🎄🌲🌳🌴🪹🪺🪵🌱🌿🍀🎍🪴🎋🍃🍂🍁🍄🐚🪨🌾💐🌷🪷🌹🥀🌺🌸🌼🌻🌞🌝🌛🌜🌚🌕🌖🌗🌘🌑🌒🌓🌔🌙🌎🌍🌏🪐💫⭐🌟✨💥🔥🌪🌈🌤🌥🌦🌧⛈🌩🌨🌬💨💧💦🫧🌊🌫' \
	'🍏🍎🍐🍊🍋🍌🍉🍇🍓🫐🍈🍒🍑🥭🍍🥥🥝🍅🍆🥑🥦🥬🥒🌶🫑🌽🥕🫒🧄🧅🥔🍠🫘🥐🥯🍞🥖🥨🧀🥚🍳🧈🥞🧇🥓🥩🍗🍖🦴🌭🍔🍟🍕🫓🥪🥙🧆🌮🌯🫔🥗🥘🫕🥫🍝🍜🍲🍛🍣🍱🥟🦪🍤🍙🍚🍘🍥🥠🥮🍢🍡🍧🍨🍦🥧🧁🍰🎂🍮🍭🍬🍫🍿🍩🍪🌰🥜🍯🥛🍼🫖☕🍵🧃🥤🧋🫙🍶🍺🍻🥂🍷🫗🥃🍸🍹🧉🍾🧊🥄🍴🍽🥣🥡🥢🧂' \
	'⚽🏀🏈⚾🥎🎾🏐🏉🥏🎱🪀🏓🏸🏒🏑🥍🏏🪃🥅🪁🏹🎣🤿🥊🥋🎽🛹🛼🛷⛸🥌🎿⛷🏂🪂🤼🤸🤺🤾🏇🧘🏄🏊🤽🚣🧗🚵🚴🏆🥇🥈🥉🏅🎖🏵🎗🎫🎟🎪🤹🎭🩰🎨🎬🎤🎧🎼🎹🥁🪘🎷🎺🪗🎸🪕🎻🎲♟🎯🎳🎮🎰🧩' \
	'🚗🚕🚙🚌🚎🏎🚓🚑🚒🚐🛻🚚🚛🚜🦯🦽🦼🛴🚲🛵🏍🛺🚨🚔🚍🚘🚖🛞🚡🚠🚟🚃🚋🚞🚝🚄🚅🚈🚂🚆🚇🚊🚉🛫🛬🛩💺🛰🚀🛸🚁🛶⛵🚤🛥🛳⛴🚢🛟🪝🚧🚦🚥🚏🗺🗿🗽🗼🏰🏯🏟🎡🎢🛝🎠⛱🏖🏝🏜🌋⛰🏔🗻🏕🛖🏠🏡🏘🏚🏗🏭🏢🏬🏣🏤🏥🏦🏨🏪🏫🏩💒🏛🕌🕍🛕🕋⛩🛤🛣🗾🎑🏞🌅🌄🌠🎇🎆🌇🌆🏙🌃🌌🌉🌁' \
	'⌚📱📲💻🖥🖨🖱🖲🕹🗜💽💾💿📀📼📷📸📹🎥📽🎞📞📟📠📺📻🎙🎚🎛🧭⏱⏲⏰🕰⌛⏳📡🔋🪫🔌💡🔦🕯🪔🧯🛢💸💵💴💶💷🪙💰💳💎🪜🧰🪛🔧🔨⚒🛠⛏🪚🔩🪤🧱⛓🧲🔫💣🧨🪓🔪🗡🛡🚬🪦🏺🔮📿🧿🪬💈🔭🔬🕳🩹🩺🩻🩼💊💉🩸🧬🦠🧫🧪🌡🧹🪠🧺🧻🚽🚰🚿🛁🛀🧼🪥🪒🧽🪣🧴🛎🔑🗝🚪🪑🛋🛏🛌🧸🪆🖼🪞🪟🛍🛒🎁🎈🎏🎀🪄🪅🎊🎉🪩🎎🏮🎐🧧📩📨📧💌📥📤📦🏷🪧📪📫📬📭📮📯📜📃📄📑🧾📊📈📉🗒🗓📆📅🗑🪪📇🗃🗳🗄📋📁📂🗂🗞📰📓📔📒📕📗📘📙📚📖🔖🧷🔗📎🖇📐📏🧮📌📍🖊🖋🖌🖍📝🔍🔎🔏🔐🔒🔓' \
	'🧡💛💚💙💜🖤🤍🤎💔💕💞💓💗💖💘💝💟🔯🕎🛐⛎🆔🉑📴📳🈶🈸🈺🆚💮🉐🈴🈵🈹🈲🆎🆑🆘❌🛑⛔📛🚫💯💢🚷🚯🚳🚱🔞📵🚭🔅🔆🚸🔱🔰✅💹❎🌐💠🌀💤🏧🚾🛗🈳🛂🛃🛄🛅🚹🚺🚼⚧🚻🚮🎦📶🈁🔣🔤🔡🔠🆖🆗🆙🆒🆕🆓🔟🔢⏸⏯⏹⏺⏭⏮⏩⏪⏫⏬🔼🔽🔀🔁🔂🔄🔃🎵🎶➕➖➗🟰♾💲💱➰➿🔚🔙🔛🔝🔜🔘🔴🟠🟡🟢🔵🟣🟤🔺🔻🔸🔹🔶🔷🔳🔲🟥🟧🟨🟩🟦🟪🟫🔈🔇🔉🔊🔔🔕📣📢💬💭🗯🃏🎴🕐🕑🕒🕓🕔🕕🕖🕗🕘🕙🕚🕛🕜🕝🕞🕟🕠🕡🕢🕣🕤🕥🕦🕧' \
	'🏴🏁🚩🎌'
)

AUTHORIZED_ASCII = set(
	'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' \
	'0123456789' \
	' !"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~'
)

REPLACE_UNICODE = {
	'« ': '"',
	' »': '"',
	'«': '"',
	'»': '"',
	'❗️': '!',
	'❕': '!',
	'❓': '?',
	'❔': '?',
	'‼️': '!!',
	'⁉️': '!?',
	'✖️': '❌',
	'✔️': '✅',
	'☺': '😊',
	'☺️': '😊',
	'☹': '🙁',
	'☹️': '🙁'
}

ENCODE_STRING_EMOJIS = {
	'☂️': '☂',
	'☀️': '☀',
	'❄️': '❄',
	'✈️': '✈',
	'☎️': '☎',
	'⚙️': '⚙',
	'⚔️': '⚔',
	'✉️': '✉',
	'✂️': '✂',
	'✒️': '✒',
	'❤️': '❤',
	'☢️': '☢',
	'☣️': '☣',
	'⚠️': '⚠',
	'♻️': '♻',
	'🏳️': '🏳',
	'🏳️‍🌈': '①',
	'🏳️‍⚧️': '②',
	'🏴‍☠️': '③',
	'🇺🇸': '④',
	'🇨🇳': '⑤',
	'🇯🇵': '⑥',
	'🇩🇪': '⑦',
	'🇮🇳': '⑧',
	'🇬🇧': '⑨',
	'🇫🇷': '⑩',
	'🇮🇹': '⑪',
	'🇨🇦': '⑫',
	'🇧🇷': '⑬',
	'🇷🇺': '⑭',
	'🇰🇷': '⑮',
	'🇦🇺': '⑯',
	'🇲🇽': '⑰',
	'🇪🇸': '⑱'
}

DECODE_STRING_EMOJIS = {value: key for key, value in ENCODE_STRING_EMOJIS.items()}

ENCODE_CHARS = list('①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱')

REPLACE_ASCII_STRING = {
	'--': '-'
}

REPLACE_ASCII = {
	'\n': '↲',
	'\t': '⇥',
	'`': "'"
}

STRIP_REPLACE = {
	' ↲': '↲',
	'⇥↲': '↲',
	' ␃': '␃',
	'⇥␃': '␃',
	'↲␃': '␃',
	' ␄': '␄',
	'⇥␄': '␄',
	'↲␄': '␄'
}

POSSIBLE_CHARS = AUTHORIZED_UNICODE | set(DECODE_STRING_EMOJIS.keys()) | set('↲⇥␃␄�')


def clean_ascii(char: str) -> str:

	if char in REPLACE_ASCII:
		return REPLACE_ASCII[char]

	if char in AUTHORIZED_ASCII:
		return char

	return ''


def clean_unicode(char: str) -> str:

	if char in AUTHORIZED_UNICODE or char in DECODE_STRING_EMOJIS:
		return char

	text = unidecode(char)

	for key, value in REPLACE_ASCII_STRING.items():
		text = text.replace(key, value)

	return ''.join([clean_ascii(char) for char in text])


def clean_string(text: str) -> str:

	for key, value in REPLACE_UNICODE.items():
		text = text.replace(key, value)

	for char in ENCODE_CHARS:
		text = text.replace(char, unidecode(char))

	for key, value in ENCODE_STRING_EMOJIS.items():
		text = text.replace(key, value)

	text = ''.join([clean_unicode(char) for char in text])

	for key, value in STRIP_REPLACE.items():
		while key in text:
			text = text.replace(key, value)

	return text


def clean_document(document: str) -> str:

	return clean_string(document) + '␄'


def clean_message(message: str) -> str:

	message = clean_string(message).strip()

	while True:

		if message[0] == '>':
			start = 0
		elif message.find('↲>') != -1:
			start = message.find('↲>') + 1
		else:
			start = None

		if start is None:
			break

		end = message.find('↲', start + 1)

		if end == -1:
			break

		message = message[:start] + message[end:]

	while '↲↲↲' in message:
		message = message.replace('↲↲↲', '↲↲')

	while message.startswith('↲') or message.startswith('⇥') or message.startswith(' '):
		message = message[1:]

	while message.endswith('↲') or message.endswith('⇥') or message.endswith(' '):
		message = message[:-1]

	return message + '␃'


def clean_chat(chat: list[str]) -> str:

	for i in range(len(chat)):
		chat[i] = clean_message(chat[i])

	chat = ''.join(chat)

	if len(chat) == 0:
		return ''

	return chat[:-1] + '␄'


def decode_string(text: str) -> str:

	for key, value in DECODE_STRING_EMOJIS.items():
		text = text.replace(key, value)

	text = text.replace('↲', '\n')
	text = text.replace('⇥', '\t')
	text = text.replace('␃', '\n\n----- END OF MESSAGE -----\n\n')
	text = text.replace('␄', '\n\n----- END OF DOCUMENT -----\n\n')

	return text
