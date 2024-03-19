import regex
from unidecode import unidecode

from dimgpt.settings import *


AUTHORIZED_UNICODE = set(
	'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' \
	'0123456789' \
	' !"#$%&\'`()*+,-./:;<=>?@[\\]^_{|}~' \
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
	' !"#$%&\'`()*+,-./:;<=>?@[\\]^_{|}~'
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

STRIP_REPLACE = {
	' \n': '\n',
	'\t\n': '\n',
	'\n\n\n': '\n\n'
}

CONTROL_REPLACE = {
	'\t': '⮜tab⮞',
	'\n': '⮜new-line⮞'
}

POSSIBLE_CHARS = AUTHORIZED_UNICODE | set(DECODE_STRING_EMOJIS.keys())


def clean_ascii(char: str) -> str:

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


def clean_string(text: str, keep_control_tokens: bool = False) -> str:

	if len(text) == 0:
		return ''

	text = text.replace('\r', '')

	if keep_control_tokens:

		safe_control_tokens = [regex.escape(c) for c in CONTROL_TOKENS]
		reg = r'(' + r'|'.join(safe_control_tokens) + r''.join([f'[{i}]' for i in safe_control_tokens]) + r']+)'
		parts = regex.split(reg, text, flags = regex.UNICODE, concurrent = False)
		parts = list(filter(None, parts))

		return ''.join([part if part in CONTROL_TOKENS else clean_string(part) for part in parts])

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

	text = text.strip()

	for key, value in CONTROL_REPLACE.items():
		text = text.replace(key, value)

	return text


def unclean_string(text: str, keep_control_tokens: bool = False) -> str:

	for key, value in DECODE_STRING_EMOJIS.items():
		text = text.replace(key, value)

	if keep_control_tokens:
		return text

	text = text.replace('⮜unknown⮞', '�')
	text = text.replace('⮜padding⮞', '')
	text = text.replace('⮜start-of-text⮞', '\n\n---------- START OF TEXT ----------\n\n')
	text = text.replace('⮜tab⮞', '\t')
	text = text.replace('⮜new-line⮞', '\n')
	text = text.replace('⮜human⮞', '\n\n--- Human ---\n\n')
	text = text.replace('⮜system⮞', '\n\n--- System ---\n\n')
	text = text.replace('⮜user⮞', '\n\n--- User ---\n\n')
	text = text.replace('⮜assistant⮞', '\n\n--- Assistant ---\n\n')
	text = text.replace('⮜end-of-text⮞', '\n\n---------- END OF TEXT ----------\n\n')

	return text
