from unidecode import unidecode


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

POSSIBLE_CHARS = AUTHORIZED_UNICODE | set(DECODE_STRING_EMOJIS.keys()) | set('↲⇥␃␄')


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

	return ''.join([clean_unicode(char) for char in text])


def clean_document(document: str) -> str:

	return clean_string(document) + '␄'


def decode_string(text: str) -> str:

	for key, value in DECODE_STRING_EMOJIS.items():
		text = text.replace(key, value)

	return text
