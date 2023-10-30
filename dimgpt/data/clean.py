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
	'🇪🇸': '⑱',
	'🏳️': '🏳'
}

DECODE_STRING_EMOJIS = {value: key for key, value in reversed(ENCODE_STRING_EMOJIS.items())}

ENCODE_CHARS = list('①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱')

REPLACE_ASCII_STRING = {
	'--': '-'
}

REPLACE_ASCII = {
	'`': "'"
}

STRIP_REPLACE = {
	' \n': '\n',
	'\t\n': '\n',
	'\n\n\n': '\n\n'
}

STRIP_CONTROLS = ['<sot>', '<som>', '<user>', '<bot>', '<eom>', '<eot>']

STRIP_SPACES = [' ', '\t', '\n']

CONTROL_REPLACE = {
	'\t': '<tab>',
	'\n': '<nl>'
}

POSSIBLE_CHARS = AUTHORIZED_UNICODE | set(DECODE_STRING_EMOJIS.keys())


def clean_ascii(char: str) -> str:

	if char in REPLACE_ASCII:
		return REPLACE_ASCII[char]

	if char in AUTHORIZED_ASCII or char in CONTROL_REPLACE.keys():
		return char

	return ''


def clean_unicode(char: str) -> str:

	if char in AUTHORIZED_UNICODE or char in DECODE_STRING_EMOJIS or char in CONTROL_REPLACE.keys():
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

	for control in STRIP_CONTROLS:
		for space in STRIP_SPACES:
			while space + control in text:
				text = text.replace(space + control, control)
			while control + space in text:
				text = text.replace(control + space, control)

	text = text.strip()

	for key, value in CONTROL_REPLACE.items():
		text = text.replace(key, value)

	return text


def decode_string(text: str, keep_control: bool = False) -> str:

	for key, value in DECODE_STRING_EMOJIS.items():
		text = text.replace(key, value)

	if keep_control:
		return text

	text = text.replace('<tab>', '\t')
	text = text.replace('<nl>', '\n')
	text = text.replace('<sot>', '\n\n---------- START OF DOCUMENT ----------\n\n')
	text = text.replace('<som>', '\n\n--- Start of message ---\n\n')
	text = text.replace('<user>', '\n\n--- User ---\n\n')
	text = text.replace('<bot>', '\n\n--- Bot ---\n\n')
	text = text.replace('<eom>', '\n\n--- End of message ---\n\n')
	text = text.replace('<eot>', '\n\n---------- END OF DOCUMENT ----------\n\n')
	text = text.replace('<unk>', '�')

	return text
