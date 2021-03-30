from string import ascii_lowercase, digits, ascii_uppercase

class VariableNameHelper(object):
    __permittedCharacter = ascii_lowercase + ascii_uppercase + digits + "_"
    __permittedStartingCharacter = ascii_lowercase + ascii_uppercase + "_"

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
    
    @staticmethod
    def check(name: str) -> bool:
        if name[0] not in VariableNameHelper.__permittedStartingCharacter:
            return False
        
        for char in name:
            if char not in VariableNameHelper.__permittedCharacter:
                return False
        
        return True

if __name__ == "__main__":
    assert VariableNameHelper.check("_bnu#s") == False
    assert VariableNameHelper.check("ascd") == True