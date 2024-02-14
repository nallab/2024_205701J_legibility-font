import os
import string
from ifontDataAdopter import IFontDataAdopter

class Attr2FontDataAdopter(IFontDataAdopter):
  def __init__(self) -> None:
    charList = list(string.digits+string.ascii_lowercase+string.ascii_uppercase)
    self.char_to_num = {charList[i]: i for i in range(0, len(charList))}

  def alphabet_to_img_file_name(self, font_name, alphabet:str) -> str:
    return os.path.join(font_name, f"{self.char_to_num[alphabet]}.png")
