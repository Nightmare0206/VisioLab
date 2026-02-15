import django

class securePassword:
    def __init__(self, password):
        self.password = password

    def is_strong(self):
        # Check if the password is strong
        if len(self.password) < 8:
            return False
        if not any(char.isdigit() for char in self.password):
            return False
        if not any(char.isupper() for char in self.password):
            return False
        if not any(char.islower() for char in self.password):
            return False
        return True
    
class add_Image_to_Database:
    def __init__(self, image):
        self.image = image

    def save_to_database(self):
        # Code to save the image to the database
        pass