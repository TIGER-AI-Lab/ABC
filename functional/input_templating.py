
def text_only_input(text):
    return [{
            "role": "user",
            "content": [
                {"type": "text", "text" : text}
            ]
            }]

def image_only_input(img):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text" : ""}
        ]
    }]

def image_and_inst_input(text, img):
    return [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text" : f"Instruction: {text}"}
            ]
        }]