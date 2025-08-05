import jsonyx as json

annotation_rules = {
    "品牌": "商品品牌名稱，如 華碩、LG",
    "系列名稱": "產品中所有的“產品系列”，以及商人為了特殊目的所特別額外創造出的商品名稱，使用者可能會利用該名稱搜尋商品，不包含特殊主題或是產品類型，不含廣告詞。如 Iphone 12、ROG 3060Ti",
    "產品類型": "實際產品名稱",
    "產品序號": "產品序號，該產品所擁有的唯一英數符號組合序號，不含系列名。",
    "顏色": "顏色資訊，包含化妝品色調以及明亮，如 花朵紅、藍色系、晶亮",
    "材質": "產品的製造材料，一般情況下不能食用，較接近原物料，不是產品成分，請注意，“紙”尿褲的材質是棉，不是紙，“皮革”外套的皮革是材質。如 木製、PVC材質、304不鏽鋼",
    "對象與族群": "人與動物的族群，如 新生兒用、寵物用、高齡族群",
    "適用物體、事件與場所": "適用的物品、事件、場所。如 手部用、騎車用、廚房用",
    "特殊主題": "該物品富含特殊人為創造創作、人物，並且該創作名有一定知名度。如 航海王、J.K.Rowling",
    "形狀": "商品形狀，囊括簡單的幾何圖形，以及明確從名稱中可以知道該商品屬於該形狀的詞。包含衣服版型。如 圓形、紐扣形、可愛熊造型、寬版、無袖、長筒、窄邊框",
    "圖案": "商品上的圖案，囊括簡單的幾何圖形，以及明確從名稱中可以知道該商品屬於該圖案的詞",
    "尺寸": "商品大小，常以數字與單位或特殊規格形式出現，如 120x80x10cm (長寬高)、XL、ATX (主機板)",
    "重量": "商品重量，常以數字與單位或特殊規格形式出現，如 10g、極輕",
    "容量": "商品容量，常以數字與單位或特殊規格形式出現，如 128G (電腦)、大容量",
    "包裝組合": "產品包裝方式、包裝分量以及贈品，如 10入、10g/包、鍵盤滑鼠桌墊組合、送電池",
    "功能與規格": "產品的功用、與其特殊規格、以及產品額外的特性。如 USB3.0、防臭、太陽能",
}

all_atts = list(annotation_rules.keys())

all_atts_json = json.dumps(all_atts, ensure_ascii=False)

PROMPT_TEMPLATE = (
    "商品名稱：{ctx}\n"
    "請根據以下屬性列表，從商品名稱中抽取相應實體：\n"
    f"{all_atts_json}\n"
    "請以純 JSON 回傳，格式範例：\n"
    "{{\n"
    '  "品牌": [...],\n'
    '  ...\n'
    '  "功能與規格": [...]\n'
    "}}\n"
    "若無對應實體，請以空陣列表示"
)

conll2003_atts = ["ORG", "MISC", "PER","LOC"]
conll2003_atts_json = json.dumps(conll2003_atts, ensure_ascii=False)

PROMPT_TEMPLATE_CONLL2003 = (
    "Sentence：{ctx}\n"
    "Based on the following attribute list, extract the corresponding entities from the sentence:\n"
    f"{conll2003_atts_json}\n"
    "Please return in pure JSON format, example format:\n"
    "{{\n"
    '  "ORG": [...],\n'
    '  "MISC": [...],\n'
    '  "PER": [...],\n'
    '  "LOC": [...]\n'
    "}}\n"
    "If there are no corresponding entities, please represent them with an empty array."
)

SEED = 2025

def set_seed(seed=SEED):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import re

find_all   = lambda txt, sub: [m.start() for m in re.finditer(re.escape(sub), txt)]
make_O     = lambda n: ["O"] * n
add_prefix = lambda tags, attr: [f"{t}-{attr}" if t in ("B", "I") else "O" for t in tags]

def fill(tags, pos, length, attr):
    tags[pos] = f"B-{attr}"
    for i in range(1, length):
        if pos + i < len(tags) and tags[pos + i] == "O":
            tags[pos + i] = f"I-{attr}"

ALIAS = {
    "產品": "產品類型",
    "名稱": "系列名稱",
}
