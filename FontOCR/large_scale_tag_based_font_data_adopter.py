from ifontDataAdopter import IFontDataAdopter

class LargeScaleTagBasedFontDataAdopter(IFontDataAdopter):
  def __init__(self) -> None:
    pass
  def alphabet_to_img_file_name(self, font_name, alphabet:str) -> str:
    if (alphabet.isupper()):
      alphabet = alphabet*2
    return f"{font_name}_{alphabet}.png"
