from string import ascii_lowercase, digits, ascii_uppercase

class VariableNameHelper(object):
    __permittedCharacter = ascii_lowercase + ascii_uppercase + digits + "_"

    @staticmethod
    def parse(name: str):
        newName = ""
        for char in name:
            if char not in VariableNameHelper.__permittedCharacter:
                newName += "_"
            else:
                newName += char
            
        if newName[0] == "_":
            return newName[1:]
        
        return newName
