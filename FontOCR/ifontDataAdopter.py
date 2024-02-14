import abc

class IFontDataAdopter(metaclass=abc.ABCMeta):
  @abc.abstractmethod
  def alphabet_to_img_file_name(self, font_name, alphabet) -> str:
    raise NotImplementedError()